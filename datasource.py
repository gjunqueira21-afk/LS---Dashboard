"""Camada de dados do LS Dashboard: resolucao de ticker + download resiliente.

Por que Ticker.history e nao yf.download: `yf.download` engole TODO erro num
DataFrame vazio -- ticker inexistente, rate limit e Yahoo fora do ar sao
indistinguiveis. `Ticker.history` propaga excecao tipada, o que permite dar ao
usuario uma mensagem util em vez de "verifique os tickers".
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

from tickers import Symbol, normalize_ticker

TZ_BR = ZoneInfo("America/Sao_Paulo")

# Baixa sempre a janela maxima e recorta em memoria: 1 entrada de cache por
# ticker em vez de 5 (uma por opcao do seletor). Trocar "2 anos" -> "5 anos"
# fica instantaneo e sem rede.
MAX_PERIOD = "5y"
PERIOD_DAYS = {"6mo": 183, "1y": 365, "2y": 730, "3y": 1095, "5y": 1826}


class DataError(Exception):
    """Falha de obtencao de dados ja classificada e com mensagem para o usuario."""

    def __init__(self, kind: str, message: str, symbol: str = "", retryable: bool = False):
        super().__init__(message)
        self.kind = kind            # not_found | rate_limit | source_down | network | empty
        self.message = message
        self.symbol = symbol
        self.retryable = retryable


@dataclass
class SeriesResult:
    """Serie de fechamentos + tudo que a UI precisa saber sobre a origem dela."""
    series: pd.Series
    symbol: str                     # simbolo Yahoo efetivamente usado
    display: str                    # rotulo curto para a UI
    resolution_note: str = ""       # "petr4 -> PETR4.SA" (vazio se trivial)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# TTL adaptativo
# ---------------------------------------------------------------------------
def cache_bucket(now: datetime | None = None) -> int:
    """Balde de tempo que invalida o cache na granularidade certa.

    Dado diario so muda de verdade durante o pregao (a barra de hoje e viva).
    Fora do pregao a serie esta congelada e recarregar e desperdicio -- era
    assim que o ttl=30 antigo queimava ~120 downloads/hora por ticker para
    receber bytes identicos, e era assim que se ganhava um 429 do Yahoo.

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
# Download
# ---------------------------------------------------------------------------
def _classify(exc: Exception, symbol: str) -> DataError:
    """Traduz excecao do yfinance em erro classificado com mensagem util."""
    name = type(exc).__name__
    msg = str(exc)

    if name == "YFRateLimitError" or "429" in msg or "Too Many Requests" in msg:
        return DataError(
            "rate_limit",
            "O Yahoo Finance bloqueou temporariamente as requisicoes (rate limit). "
            "Aguarde cerca de 1 minuto e tente de novo.",
            symbol, retryable=True,
        )
    # 404 cru: o Yahoo responde 404 + "symbol may be delisted" p/ ticker inexistente.
    # Sem esta regra o erro vira "fonte fora do ar" e o fallback de sufixo nao anda.
    if name in ("YFPricesMissingError", "YFTickerMissingError", "YFTzMissingError") \
            or "404" in msg or "Not Found" in msg or "may be delisted" in msg:
        return DataError(
            "not_found",
            f"O simbolo **{symbol}** nao existe no Yahoo Finance (ou foi deslistado).",
            symbol,
        )
    if name == "YFDataException" or "CURRENTLY DOWN" in msg.upper():
        return DataError(
            "source_down", "O Yahoo Finance esta fora do ar. Tente em alguns minutos.",
            symbol, retryable=True,
        )
    if any(k in name for k in ("Timeout", "Connection", "SSL", "Proxy")) or "timed out" in msg.lower():
        return DataError(
            "network", "Falha de rede ao falar com o Yahoo Finance. Verifique a conexao.",
            symbol, retryable=True,
        )
    return DataError("source_down", f"Erro inesperado no download de {symbol}: {name}: {msg[:160]}",
                     symbol, retryable=True)


def _download_once(symbol: str, period: str) -> pd.Series:
    """Uma tentativa contra o Yahoo, com excecao tipada."""
    tk = yf.Ticker(symbol)
    try:
        # yfinance >= 1.5 depreciou raise_errors em favor do config global.
        if hasattr(yf, "config"):
            yf.config.debug.hide_exceptions = False
            hist = tk.history(period=period, interval="1d", auto_adjust=True)
        else:
            hist = tk.history(period=period, interval="1d", auto_adjust=True, raise_errors=True)
    except Exception as exc:
        raise _classify(exc, symbol) from exc

    if hist is None or hist.empty or "Close" not in hist:
        raise DataError("not_found",
                        f"O Yahoo nao devolveu nenhum pregao para **{symbol}**.", symbol)

    close = hist["Close"]
    if isinstance(close, pd.DataFrame):          # MultiIndex em algumas versoes
        close = close.iloc[:, 0]

    # CRITICO: Ticker.history devolve indice tz-aware NA BOLSA DE ORIGEM
    # (America/Sao_Paulo p/ .SA, America/New_York p/ US). Sem normalizar,
    # concat de um par cross-market casa ZERO linhas.
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.dropna().astype(float)


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

    Cache de 24h: a resposta de 'PETR4.SA existe?' nao muda ao longo do dia,
    entao o fallback custa UMA chamada extra por dia, nao uma por rerun. Para
    ticker B3 (99% do uso) candidates tem 1 item e isso nem e chamado.
    """
    last: DataError | None = None
    for cand in candidates:
        try:
            _download_with_retry(cand, "5d", attempts=2)
            return cand
        except DataError as err:
            if err.kind != "not_found":
                raise           # Yahoo caiu / rate limit: nao vire "nao existe"
            last = err
    tried = " / ".join(candidates)
    raise DataError("not_found",
                    f"Nenhum simbolo encontrado no Yahoo Finance. Tentei: {tried}.",
                    candidates[0]) from last


def fetch_series(user_input: str, period: str) -> SeriesResult:
    """Ponto de entrada da UI: recebe o que o usuario digitou, devolve a serie.

    Resolve o ticker (petr4 -> PETR4.SA), tenta o fallback quando ha
    ambiguidade, baixa o historico maximo uma unica vez e recorta o periodo
    pedido em memoria.
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
    B3 x B3 isso e ~0%; em PETR4.SA x AAPL passa de 5% da amostra -- e e
    justamente o caso em que a cointegracao fica enviesada.
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
