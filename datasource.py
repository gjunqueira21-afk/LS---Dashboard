"""Camada de dados do LS Dashboard: resolucao de ticker + download resiliente.

Fonte de dados: **brapi.dev** (API nativa da B3). Preços históricos vêm do
endpoint /api/quote/{ticker}?range=...&interval=1d. Usa-se o token PRO
(secrets BRAPI_TOKEN ou variável de ambiente) para histórico completo e sem
limite de requisições.

Por que classificar o erro: uma resposta vazia pode ser ticker inexistente,
token inválido, brapi fora do ar ou rate limit -- indistinguíveis se tratados
como "DataFrame vazio". Aqui cada caso vira um DataError tipado com mensagem
útil ao usuário.
"""

from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from tickers import Symbol, normalize_ticker

TZ_BR = ZoneInfo("America/Sao_Paulo")

BRAPI_URL = "https://brapi.dev/api/quote/{ticker}"
BRAPI_TIMEOUT = 20

# Baixa sempre a janela maxima e recorta em memoria: 1 entrada de cache por
# ticker em vez de 5 (uma por opcao do seletor). Trocar "2 anos" -> "5 anos"
# fica instantaneo e sem rede.
MAX_PERIOD = "5y"
PERIOD_DAYS = {"6mo": 183, "1y": 365, "2y": 730, "3y": 1095, "5y": 1826}

# range aceitos pelo brapi; period fora disso cai no MAX_PERIOD.
_BRAPI_RANGES = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}


class DataError(Exception):
    """Falha de obtencao de dados ja classificada e com mensagem para o usuario."""

    def __init__(self, kind: str, message: str, symbol: str = "", retryable: bool = False):
        super().__init__(message)
        self.kind = kind            # not_found | rate_limit | source_down | network | empty | auth
        self.message = message
        self.symbol = symbol
        self.retryable = retryable


@dataclass
class SeriesResult:
    """Serie de fechamentos + tudo que a UI precisa saber sobre a origem dela."""
    series: pd.Series
    symbol: str                     # simbolo resolvido efetivamente usado
    display: str                    # rotulo curto para a UI
    resolution_note: str = ""       # "petr4 -> PETR4.SA" (vazio se trivial)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------
def _brapi_token() -> str:
    """Token do brapi.dev, em ordem de prioridade:
    campo colado na UI (session_state) -> secrets BRAPI_TOKEN -> env BRAPI_TOKEN.
    """
    try:
        ui = st.session_state.get("_brapi_token_ui", "")
        if ui:
            return str(ui).strip()
    except Exception:
        pass
    try:
        tok = st.secrets.get("BRAPI_TOKEN", "")
        if tok:
            return str(tok).strip()
    except Exception:
        pass
    return os.environ.get("BRAPI_TOKEN", "").strip()


def _brapi_ticker(symbol: str) -> str:
    """Converte o simbolo interno (PETR4.SA) no ticker B3 que o brapi usa (PETR4)."""
    s = symbol.strip().upper()
    if s.endswith(".SA"):
        s = s[:-3]
    return s


# ---------------------------------------------------------------------------
# TTL adaptativo
# ---------------------------------------------------------------------------
def cache_bucket(now: datetime | None = None) -> int:
    """Balde de tempo que invalida o cache na granularidade certa.

    Dado diario so muda de verdade durante o pregao (a barra de hoje e viva).
    Fora do pregao a serie esta congelada e recarregar e desperdicio.

      pregao B3 (seg-sex, 10:00-18:30)  -> balde de 60s
      resto do dia util                 -> balde de 15min
      fim de semana                     -> balde de 6h

    Passado como argumento de uma funcao @st.cache_data, muda a chave de cache
    exatamente quando deve -- o que `ttl=` fixo nao consegue fazer.

    Limitacao conhecida: usa dia-da-semana e horario, nao o calendario de
    feriados da B3. Em feriado nacional ainda opera na granularidade de 60s.
    """
    now = now or datetime.now(TZ_BR)
    if now.weekday() >= 5:
        gran = 21600
    elif time(10, 0) <= now.time() <= time(18, 30):
        gran = 60
    else:
        gran = 900
    return int(now.timestamp()) // gran


# ---------------------------------------------------------------------------
# Download (brapi.dev)
# ---------------------------------------------------------------------------
def _classify_http(status: int, symbol: str, body: str = "") -> DataError:
    """Traduz status HTTP do brapi em erro classificado com mensagem util."""
    if status in (401, 403):
        return DataError(
            "auth",
            "O brapi.dev recusou a autenticacao (token invalido ou ausente). "
            "Confira o BRAPI_TOKEN em .streamlit/secrets.toml.",
            symbol,
        )
    if status == 404:
        return DataError(
            "not_found",
            f"O ticker **{symbol}** nao foi encontrado no brapi.dev.",
            symbol,
        )
    if status == 429:
        return DataError(
            "rate_limit",
            "O brapi.dev bloqueou temporariamente as requisicoes (rate limit). "
            "Aguarde cerca de 1 minuto e tente de novo.",
            symbol, retryable=True,
        )
    if status >= 500:
        return DataError(
            "source_down", "O brapi.dev esta instavel (erro 5xx). Tente em alguns minutos.",
            symbol, retryable=True,
        )
    return DataError("source_down",
                     f"Resposta inesperada do brapi.dev para {symbol} (HTTP {status}). {body[:120]}",
                     symbol, retryable=True)


def _download_once(symbol: str, period: str) -> pd.Series:
    """Uma tentativa contra o brapi.dev, com excecao tipada."""
    tk = _brapi_ticker(symbol)
    rng = period if period in _BRAPI_RANGES else MAX_PERIOD
    params = {"range": rng, "interval": "1d"}
    token = _brapi_token()
    if token:
        params["token"] = token

    try:
        resp = requests.get(BRAPI_URL.format(ticker=tk), params=params, timeout=BRAPI_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise DataError("network",
                        "Falha de rede ao falar com o brapi.dev. Verifique a conexao.",
                        symbol, retryable=True) from exc

    if resp.status_code != 200:
        # brapi devolve JSON de erro com mensagem util em alguns casos
        body = ""
        try:
            body = resp.json().get("message", "") or resp.text
        except Exception:
            body = resp.text
        raise _classify_http(resp.status_code, symbol, body)

    try:
        data = resp.json()
    except ValueError as exc:
        raise DataError("source_down", f"brapi.dev devolveu resposta nao-JSON para {symbol}.",
                        symbol, retryable=True) from exc

    if data.get("error"):
        raise DataError("not_found",
                        f"brapi.dev: {data.get('message', 'ticker nao encontrado')} ({symbol}).",
                        symbol)

    results = data.get("results") or []
    if not results:
        raise DataError("not_found", f"brapi.dev nao retornou dados para **{symbol}**.", symbol)

    hist = results[0].get("historicalDataPrice") or []
    if not hist:
        raise DataError("not_found",
                        f"brapi.dev nao devolveu historico de precos para **{symbol}**.", symbol)

    dates, closes = [], []
    for row in hist:
        d = row.get("date")
        c = row.get("adjustedClose")
        if c is None:
            c = row.get("close")
        if d is None or c is None:
            continue
        dates.append(d)
        closes.append(c)

    if not dates:
        raise DataError("empty", f"brapi.dev devolveu historico vazio para **{symbol}**.", symbol)

    # date do brapi e unix (segundos, UTC). Normaliza pra data BR sem tz, igual
    # ao resto do app, pra que o concat de um par case as linhas certas.
    idx = pd.to_datetime(dates, unit="s", utc=True).tz_convert(TZ_BR).tz_localize(None).normalize()
    close = pd.Series(closes, index=idx, dtype="float64")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.dropna()


def _download_with_retry(symbol: str, period: str, attempts: int = 3) -> pd.Series:
    """Backoff exponencial apenas para erros retentaveis. 'Nao existe' falha na hora."""
    delay = 0.8
    last: DataError | None = None
    for i in range(attempts):
        try:
            return _download_once(symbol, period)
        except DataError as err:
            if not err.retryable:
                raise
            last = err
            if i < attempts - 1:
                _time.sleep(delay)
                delay *= 2.2
    raise last  # type: ignore[misc]


@st.cache_data(ttl=3600, max_entries=64, show_spinner=False)
def _fetch_cached(symbol: str, _bucket: int) -> pd.Series:
    """Camada cacheada: 1 entrada por simbolo, historico maximo.

    ttl=3600 e so a rede de seguranca; quem manda na invalidacao e _bucket.
    Excecoes nao sao cacheadas pelo Streamlit -- falha nao fica grudada.
    """
    return _download_with_retry(symbol, MAX_PERIOD)


@st.cache_data(ttl=86400, max_entries=256, show_spinner=False)
def _resolve_cached(candidates: tuple, _bucket_day: int) -> str:
    """Descobre qual candidato realmente existe.

    Cache de 24h: a resposta de 'PETR4 existe?' nao muda ao longo do dia. Para
    ticker B3 (99% do uso) candidates tem 1 item e isso nem e chamado.
    """
    last: DataError | None = None
    for cand in candidates:
        try:
            _download_with_retry(cand, "5d", attempts=2)
            return cand
        except DataError as err:
            if err.kind != "not_found":
                raise           # brapi caiu / rate limit / auth: nao vire "nao existe"
            last = err
    tried = " / ".join(candidates)
    raise DataError("not_found",
                    f"Nenhum simbolo encontrado no brapi.dev. Tentei: {tried}.",
                    candidates[0]) from last


def fetch_series(user_input: str, period: str) -> SeriesResult:
    """Ponto de entrada da UI: recebe o que o usuario digitou, devolve a serie.

    Resolve o ticker (petr4 -> PETR4.SA), baixa o historico maximo uma unica vez
    e recorta o periodo pedido em memoria.
    """
    sym: Symbol = normalize_ticker(user_input)
    if not sym.candidates:
        raise DataError("not_found", "Ticker vazio.", user_input)

    warnings = [sym.note] if sym.note else []

    if len(sym.candidates) == 1:
        resolved = sym.candidates[0]
    else:
        resolved = _resolve_cached(sym.candidates, int(_time.time()) // 86400)

    full = _fetch_cached(resolved, cache_bucket())

    days = PERIOD_DAYS.get(period, PERIOD_DAYS["2y"])
    cutoff = full.index.max() - timedelta(days=days)
    series = full[full.index >= cutoff].copy()
    if series.empty:
        raise DataError("empty", f"Sem pregoes de **{resolved}** no periodo pedido.", resolved)
    series.name = sym.display

    # Serie anormalmente curta costuma indicar renomeacao de ticker na B3
    # (BIDI11->NUBR33, LAME4->AMER3, VVAR3->VIIA3->BHIA3): o historico antigo
    # fica orfao no simbolo velho e a serie comeca na data da renomeacao.
    esperado = days * 252 / 365
    if len(series) < 0.6 * esperado:
        warnings.append(
            f"{sym.display}: apenas {len(series)} pregoes no periodo "
            f"(esperado ~{esperado:.0f}). Possivel renomeacao de ticker ou "
            f"listagem recente."
        )

    note = "" if user_input.strip().upper().replace(" ", "") == resolved else \
        f"{user_input.strip()} → {resolved}"
    return SeriesResult(series=series, symbol=resolved, display=sym.display,
                        resolution_note=note, warnings=warnings)


# ---------------------------------------------------------------------------
# Alinhamento auditado
# ---------------------------------------------------------------------------
@dataclass
class Alignment:
    frame: pd.DataFrame
    n_used: int
    n_dropped: int
    dropped_dates: list
    pct_dropped: float


def align_pair(long_s: pd.Series, short_s: pd.Series) -> Alignment:
    """Faz o inner join do par CONTANDO o que foi descartado.

    pd.concat(...).dropna() joga fora, em silencio, todo pregao em que um dos
    ativos nao negociou (feriado de outra praca, leilao, suspensao). Em par
    B3 x B3 isso e ~0%.
    """
    joined = pd.concat([long_s.rename("long"), short_s.rename("short")],
                       axis=1, join="outer").sort_index()
    union = len(joined)
    clean = joined.dropna()
    dropped = joined.index.difference(clean.index)
    return Alignment(
        frame=clean,
        n_used=len(clean),
        n_dropped=union - len(clean),
        dropped_dates=[d.date() for d in dropped[-10:]],
        pct_dropped=(union - len(clean)) / union * 100 if union else 0.0,
    )
