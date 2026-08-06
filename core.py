"""Motor quantitativo do LS Dashboard — fonte unica de verdade.

=============================================================================
ESPECIFICACAO CANONICA  (decidida em 2026-08-06)
=============================================================================
    spread = ln(L) - ln(S)          -> beta = 1 IMPOSTO (posicao notional-neutra)
    z      = (spread - media) / desvio
    ADF    = adfuller(spread)       -> VALIDO: nenhum parametro foi estimado
    coint  = coint(ln L, ln S)      -> Engle-Granger, valores criticos CORRETOS
    corr   = correlacao de LOG-RETORNOS (nunca de niveis de preco)
    sizing = R$ igual em cada perna (beta = 1)

O beta OLS continua sendo calculado e exibido, mas e INFORMATIVO: nao entra no
sinal, no teste estatistico nem no dimensionamento. Ver `hedge_beta`.

Por que isso importa — o que estava errado antes:
  * O z rodava sobre ln(L/S) (beta=1), o card de beta mandava montar posicao
    beta-neutra, e o ADF validava L - beta*S em NIVEIS de preco. Tres objetos
    diferentes apresentados como um. O selo "estacionario" validava uma serie
    que ninguem operava.
  * `adfuller` sobre residuo de beta estimado por OLS usa a distribuicao errada
    (o erro classico do Engle-Granger em dois passos): falso positivo medido de
    ~15% em vez de 5%. Com beta IMPOSTO (=1), `adfuller` e legitimo — nada foi
    estimado da amostra.
  * Correlacao entre series I(1) e espuria: mede tendencia comum, nao
    co-movimento. Em pares nao cointegrados com fator de mercado comum o
    gatilho 0,75 sobre NIVEIS abria em ~30% dos dias; sobre RETORNOS, 0%.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

# Setup validado no backtest do research_ls_b3. Alterar isto e alterar a
# estrategia, nao a apresentacao.
SETUP_VALIDADO = {
    "mean_win": 63, "vol_win": 63, "corr_win": 252,
    "z_monitor": 1.10, "dz_monitor": 0.05, "corr_min": 0.60,
    "z_alert": 1.50, "dz_alert": 0.15,
    "z_exit": 0.40, "max_hold": 63,
}

# Custos B3 tipicos, round-trip, nas DUAS pernas, como fracao do notional.
# emolumentos+liquidacao ~0,0325%/lado, corretagem ~0,02%, slippage ~0,10%/lado.
CUSTO_RT_PADRAO = 0.0061      # 0,61%
ALUGUEL_AA_PADRAO = 0.05      # 5% a.a. na perna short (taxa BTC)


# =============================================================================
# Construcao do spread canonico
# =============================================================================
def build_spread(frame: pd.DataFrame) -> pd.Series:
    """spread canonico = ln(L) - ln(S). Beta = 1 imposto.

    `frame` vem de datasource.align_pair() com colunas "long" e "short",
    ja alinhado e sem NaN.
    """
    lp = np.log(frame)
    return (lp["long"] - lp["short"]).rename("spread")


def zscore(series: pd.Series, mean_win: int, vol_win: int) -> pd.Series:
    """Z-score de janela rolante.

    Guarda contra desvio ~0 (papel sem negocio repetindo o ultimo preco faria
    o z explodir para infinito).
    """
    m = series.rolling(mean_win).mean()
    s = series.rolling(vol_win).std()
    s = s.where(s > 1e-9)
    return (series - m) / s


def rolling_corr(frame: pd.DataFrame, window: int, basis: str = "returns") -> pd.Series:
    """Correlacao rolante. `basis="returns"` (padrao) usa LOG-RETORNOS.

    Correlacao entre series I(1) ("niveis") e espuria — mede tendencia comum.
    O modo "levels" existe so para comparacao com o comportamento antigo.
    """
    df = np.log(frame).diff().dropna() if basis == "returns" else frame
    return df["long"].rolling(window).corr(df["short"])


def hedge_beta(frame: pd.DataFrame, window: int = 252) -> tuple[float, pd.Series]:
    """Beta OLS em LOG-precos — INFORMATIVO, nao usado no sinal.

    Devolve (beta_full_sample, beta_rolling). O rolling existe para mostrar a
    banda de estabilidade: em par genuinamente cointegrado a amplitude
    (max-min)/mediana chega a 50% em 12 meses, e o numero no card muda sozinho
    conforme o usuario troca o seletor de historico. Ver essa banda e o ponto.
    """
    lp = np.log(frame)
    beta_full = float(np.polyfit(lp["short"].values, lp["long"].values, 1)[0])
    cov = lp["long"].rolling(window).cov(lp["short"])
    var = lp["short"].rolling(window).var()
    return beta_full, (cov / var).dropna()


# =============================================================================
# Testes estatisticos
# =============================================================================
def coint_pvalue(frame: pd.DataFrame) -> float | None:
    """Engle-Granger sobre LOG-precos — valores criticos corretos.

    Devolve None em falha, nunca 1.0: um sentinela de 1.0 renderiza na tela
    identico a um p-valor real de 0,9 e o usuario nao tem como distinguir
    "teste falhou" de "teste reprovou".
    """
    try:
        lp = np.log(frame)
        return float(coint(lp["long"], lp["short"])[1])
    except Exception:
        return None


def adf_pvalue(spread: pd.Series) -> float | None:
    """ADF sobre o PROPRIO spread canonico.

    Legitimo porque beta = 1 foi IMPOSTO, nao estimado. (Se o beta viesse de
    OLS na mesma amostra, `adfuller` usaria a distribuicao errada e o teste
    teria falso positivo ~3x o nominal — o erro classico do Engle-Granger em
    dois passos. Ai o correto seria `coint`.)
    """
    try:
        s = spread.dropna()
        if len(s) < 30:
            return None
        return float(adfuller(s, regression="c", autolag="AIC")[1])
    except Exception:
        return None


def half_life(spread: pd.Series) -> tuple[float, float, float] | None:
    """Meia-vida de reversao (Ornstein-Uhlenbeck), com IC 95%.

    Regride ds_t = a + b*s_{t-1} + e; meia-vida = -ln(2)/b.

    Reporte o INTERVALO, nao o ponto: o estimador e enviesado para baixo em
    meias-vidas longas (viés de amostra pequena do tipo Dickey-Fuller). Se o
    limite superior estoura o historico disponivel, o par nao reverte de forma
    utilizavel.

    Devolve (hl, lo, hi) em pregoes, ou None se nao ha amostra suficiente.
    b >= 0 (sem reversao) devolve inf.
    """
    s = spread.dropna()
    if len(s) < 40:
        return None
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().reindex(ds.index)
    ok = lag.notna() & ds.notna()
    ds, lag = ds[ok], lag[ok]
    if len(ds) < 30 or float(np.std(lag.values)) < 1e-12:
        return None

    b, a = np.polyfit(lag.values, ds.values, 1)
    if b >= 0:
        return (np.inf, np.inf, np.inf)

    resid = ds.values - (a + b * lag.values)
    se_b = float(np.std(resid, ddof=2) / (np.std(lag.values) * np.sqrt(len(lag))))
    ln2 = np.log(2.0)
    hl = float(-ln2 / b)
    b_lo, b_hi = b - 1.96 * se_b, b + 1.96 * se_b     # mais/menos negativo
    lo = float(-ln2 / b_lo) if b_lo < 0 else np.inf   # b mais negativo -> hl menor
    hi = float(-ln2 / b_hi) if b_hi < 0 else np.inf   # b menos negativo -> hl maior
    return (hl, lo, hi)


def breakeven_z(sigma: float, cost_rt: float, rent_aa: float, hold_days: float) -> float:
    """Custo convertido para UNIDADES DE Z — o 'z de break-even'.

    E o unico numero que responde "esse sinal paga a conta?". Com custo total
    ~0,95% (round-trip nas duas pernas + aluguel no holding) e sigma do
    log-ratio de 1,5%, z_be = 0,63 — ou seja, sair em |z| <= 0,40 seria sair
    no zero a zero.
    """
    if not sigma or sigma <= 0 or not np.isfinite(sigma):
        return float("inf")
    hold = max(float(hold_days), 0.0)
    custo = cost_rt + rent_aa * hold / 252.0
    return float(custo / sigma)


def z_frequency(z: pd.Series, threshold: float) -> float:
    """Fracao dos pregoes em que |z| >= threshold, no proprio par.

    O z de janela rolante NAO e normal padrao: std(z) > 1, e — o detalhe que
    inverte a intuicao — std(z) e MAIOR quanto pior o par (passeio aleatorio
    ~1,37 vs par com meia-vida de 5 dias ~1,11). Um limiar fixo em sigma
    portanto seleciona preferencialmente pares nao estacionarios. Mostrar a
    frequencia empirica desarma o numero magico.
    """
    zz = z.dropna().abs()
    return float((zz >= threshold).mean()) if len(zz) else float("nan")


def dz_background(z: pd.Series, window: int = 252) -> float:
    """|Δz| mediano de fundo — o ruido do proprio estimador.

    Δz mistura o movimento de hoje com o efeito da janela DESCARTANDO a
    observacao de N dias atras (roll-off). Com preco parado, o roll-off sozinho
    gera |Δz| mediano ~0,027 e cruza o gatilho de 0,05 em ~24% dos dias. Um
    limiar abaixo deste numero e ruido, nao momentum.
    """
    d = z.diff().abs().dropna()
    if len(d) < 20:
        return float("nan")
    return float(d.tail(window).median())


# =============================================================================
# Agregado do par
# =============================================================================
@dataclass
class PairStats:
    frame: pd.DataFrame
    spread: pd.Series
    z: pd.Series
    corr: pd.Series
    beta_now: float
    beta_roll: pd.Series
    coint_p: float | None
    adf_p: float | None
    hl: tuple[float, float, float] | None
    z_now: float
    z_prev: float
    dz_now: float
    zero_cross: bool
    corr_now: float | None
    dz_noise: float
    gap_dias: int
    last_date: pd.Timestamp
    n_obs: int
    mean_win: int
    vol_win: int
    corr_win: int
    corr_basis: str
    problems: list = field(default_factory=list)

    @property
    def sigma(self) -> float:
        """Desvio corrente do spread — escala para converter custo em z."""
        s = self.spread.rolling(self.vol_win).std().dropna()
        return float(s.iloc[-1]) if len(s) else float("nan")

    @property
    def ratio_now(self) -> float:
        return float(self.spread.iloc[-1])

    @property
    def ratio_mean(self) -> float:
        m = self.spread.rolling(self.mean_win).mean().dropna()
        return float(m.iloc[-1]) if len(m) else float("nan")


def compute_pair(frame: pd.DataFrame, mean_win: int, vol_win: int,
                 corr_win: int, corr_basis: str = "returns") -> PairStats:
    """Tudo que nao depende dos gatilhos. Caro — cacheie por (par, janelas).

    Separado de `classify` de proposito: mexer num slider de gatilho nao pode
    refazer coint, adfuller e polyfit.
    """
    problems: list[str] = []
    spread = build_spread(frame)
    z = zscore(spread, mean_win, vol_win)
    corr = rolling_corr(frame, corr_win, corr_basis)
    beta_now, beta_roll = hedge_beta(frame)

    z_clean = z.dropna()
    if z_clean.empty:
        raise ValueError(
            f"Janelas ({mean_win}/{vol_win} pregoes) maiores que o historico "
            f"alinhado ({len(frame)} pregoes). Reduza as janelas ou aumente o periodo."
        )

    z_now = float(z_clean.iloc[-1])
    z_prev = float(z_clean.iloc[-2]) if len(z_clean) > 1 else z_now
    dz_now = z_now - z_prev

    # Cruzamento do zero so conta como SAIDA se veio de dentro da banda.
    # Um gap de -1,6 para +1,6 (evento, leilao) trocava de sinal e o painel
    # antigo imprimia "CONVERGENCIA -> encerrar" no maior alerta da amostra.
    zero_cross = (np.sign(z_now) != np.sign(z_prev)) and (abs(z_prev) <= 1.10)

    gap = 0
    if len(z_clean) > 1:
        gap = int((z_clean.index[-1] - z_clean.index[-2]).days)
        if gap > 5:
            problems.append(
                f"Δz cobre {gap} dias corridos — ha pregao faltante entre as "
                f"duas ultimas observacoes validas."
            )

    corr_clean = corr.dropna()
    if corr_clean.empty:
        corr_now = None
        problems.append(
            f"Janela de correlacao ({corr_win} pregoes) maior que o historico "
            f"disponivel ({len(frame)} pregoes) — correlacao indisponivel."
        )
    else:
        corr_now = float(corr_clean.iloc[-1])

    # Papel sem negocio: yfinance repete o ultimo preco. Gera z artificialmente
    # estavel e vol subestimada.
    for col in ("long", "short"):
        travado = (frame[col].diff() == 0).rolling(5).sum().max()
        if travado is not None and travado >= 5:
            problems.append(f"{col.upper()}: 5+ pregoes consecutivos sem variacao de preco.")

    return PairStats(
        frame=frame, spread=spread, z=z, corr=corr,
        beta_now=beta_now, beta_roll=beta_roll,
        coint_p=coint_pvalue(frame), adf_p=adf_pvalue(spread),
        hl=half_life(spread),
        z_now=z_now, z_prev=z_prev, dz_now=dz_now, zero_cross=zero_cross,
        corr_now=corr_now, dz_noise=dz_background(z), gap_dias=gap,
        last_date=frame.index[-1], n_obs=len(frame),
        mean_win=mean_win, vol_win=vol_win, corr_win=corr_win,
        corr_basis=corr_basis, problems=problems,
    )


# =============================================================================
# Classificacao do sinal
# =============================================================================
@dataclass
class Gate:
    nome: str
    valor: str
    gatilho: str
    ok: bool


@dataclass
class Signal:
    level: str          # "alert" | "watch" | "exit" | "flat"
    side: str           # "long_ratio" | "short_ratio" | ""
    label: str
    desc: str
    gates: list
    esticando: bool | None

    @property
    def falhas(self) -> list:
        return [g.nome for g in self.gates if not g.ok]


def classify(st_: PairStats, cfg: dict, tem_posicao: bool = False) -> Signal:
    """Classifica o sinal E devolve a tríade de check por gatilho.

    A mudanca de comportamento em relacao ao painel antigo e so uma: o rotulo
    NEUTRO agora vem acompanhado de qual condicao falhou. Antes, um par em
    z=2,4 com Δz=0,02 num dia parado exibia "⚪ NEUTRO · Fora dos gatilhos" e o
    usuario via um |z| gritante sem saber que foi o Δz que barrou.

    Nota sobre Δz: o gatilho usa |Δz|, entao "esticando" (Δz na mesma direcao
    do z) e "revertendo" (direcao contraria) — dois trades economicamente
    opostos — entram no mesmo rotulo. Isso preserva o comportamento validado no
    backtest; a direcao e exposta em `Signal.esticando` para voce ver, e a
    decisao de separar os dois fica registrada como pendente.
    """
    z, dz = st_.z_now, st_.dz_now
    corr = st_.corr_now
    az, adz = abs(z), abs(dz)
    esticando = bool(np.sign(dz) == np.sign(z)) if dz != 0 else None

    side = "short_ratio" if z > 0 else "long_ratio"
    corr_ok = (corr is not None) and (corr >= cfg["corr_min"])
    corr_txt = "n/d" if corr is None else f"{corr:.2f}"

    def gates(zt: float, dzt: float) -> list:
        return [
            Gate("|z|", f"{az:.2f}", f"≥ {zt:.2f}", az >= zt),
            Gate("Δz", f"{adz:.3f}", f"≥ {dzt:.3f}", adz >= dzt),
            Gate("corr", corr_txt, f"≥ {cfg['corr_min']:.2f}", corr_ok),
        ]

    g_alert = gates(cfg["z_alert"], cfg["dz_alert"])
    g_watch = gates(cfg["z_monitor"], cfg["dz_monitor"])

    # Zona de saida tem prioridade — mas so e "encerrar" se houver posicao.
    if az <= cfg["z_exit"] or st_.zero_cross:
        motivo = "cruzou o zero" if st_.zero_cross and az > cfg["z_exit"] \
            else f"|z| = {az:.2f} ≤ {cfg['z_exit']:.2f}"
        if tem_posicao:
            return Signal("exit", "", "CONVERGÊNCIA · encerrar / realizar",
                          f"Zona de saída ({motivo})", g_watch, esticando)
        return Signal("flat", "", "SEM EDGE · dentro da banda de saída",
                      f"{motivo} — nada a fazer sem posição aberta", g_watch, esticando)

    if all(g.ok for g in g_alert):
        return Signal("alert", side, "ALERTA MÁXIMO",
                      "todos os gatilhos de entrada satisfeitos", g_alert, esticando)

    if all(g.ok for g in g_watch):
        return Signal("watch", side, "MONITORAR",
                      "zona operável — acompanhar de perto", g_watch, esticando)

    return Signal("flat", "", "NEUTRO · aguardar",
                  "fora dos gatilhos de entrada", g_watch, esticando)


def side_text(side: str, long_lbl: str, short_lbl: str) -> str:
    """Vocabulario unico para a direcao. O painel antigo tinha tres versoes
    diferentes da mesma frase espalhadas pelo codigo."""
    if side == "short_ratio":
        return f"vender o ratio · vender {long_lbl} / comprar {short_lbl}"
    if side == "long_ratio":
        return f"comprar o ratio · comprar {long_lbl} / vender {short_lbl}"
    return ""


# =============================================================================
# Dimensionamento
# =============================================================================
def sizing(frame: pd.DataFrame, capital: float) -> dict:
    """Sizing notional-neutro (beta = 1), em lotes de 100 da B3.

    Expoe o residuo de arredondamento de lote: em papel caro ele vira exposicao
    direcional relevante e normalmente ninguem percebe.
    """
    p_l = float(frame["long"].iloc[-1])
    p_s = float(frame["short"].iloc[-1])
    q_l = int((capital / p_l) // 100) * 100
    q_s = int((capital / p_s) // 100) * 100
    fin_l, fin_s = q_l * p_l, q_s * p_s
    return {
        "p_long": p_l, "p_short": p_s, "q_long": q_l, "q_short": q_s,
        "fin_long": fin_l, "fin_short": fin_s,
        "residuo": fin_l - fin_s,
        "residuo_pct": (fin_l - fin_s) / capital if capital else 0.0,
    }


# =============================================================================
# Analise via Claude
# =============================================================================
def analyze_with_claude(api_key: str, long_lbl: str, short_lbl: str,
                        st_: PairStats, sig: Signal, cfg: dict,
                        z_be: float, conciso: bool = False) -> str:
    """Analise qualitativa do par via Claude API.

    max_tokens generoso de proposito: no Claude Sonnet 5 o adaptive thinking
    esta LIGADO por padrao quando o campo `thinking` e omitido (no 4.6 era
    desligado), e max_tokens limita thinking + texto JUNTOS. Com o limite
    antigo de 1200 a resposta truncava no meio.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    hl_txt = "n/d"
    if st_.hl:
        hl, lo, hi = st_.hl
        hl_txt = "sem reversão" if not np.isfinite(hl) else \
            f"{hl:.0f} pregões (IC 95%: {lo:.0f}–{'∞' if not np.isfinite(hi) else f'{hi:.0f}'})"

    coint_txt = "n/d (teste falhou)" if st_.coint_p is None else \
        f"{st_.coint_p:.4f} ({'cointegrado' if st_.coint_p < 0.05 else 'NAO cointegrado'})"
    adf_txt = "n/d (teste falhou)" if st_.adf_p is None else \
        f"{st_.adf_p:.4f} ({'estacionario' if st_.adf_p < 0.05 else 'NAO estacionario'})"
    corr_txt = "n/d" if st_.corr_now is None else f"{st_.corr_now:.3f}"
    mom_txt = {True: "esticando", False: "revertendo", None: "parado"}[sig.esticando]

    prompt = f"""Você é um analista quantitativo especialista em long/short (pairs trading) na B3.
Analise o par {long_lbl} (LONG) × {short_lbl} (SHORT) e responda em português.

ESPECIFICAÇÃO (importante — não confunda com outras convenções):
- O spread é ln({long_lbl}) − ln({short_lbl}), ou seja beta = 1 IMPOSTO (posição notional-neutra,
  mesmo valor financeiro em cada perna). O beta OLS abaixo é informativo e NÃO é usado.
- O ADF roda sobre esse mesmo spread (válido, pois nada foi estimado da amostra).
- A correlação é de LOG-RETORNOS, não de níveis de preço.

SITUAÇÃO ATUAL (último pregão: {st_.last_date:%d/%m/%Y}, {st_.n_obs} observações):
- Spread: {st_.ratio_now:.4f} | média {st_.mean_win}p: {st_.ratio_mean:.4f} | σ: {st_.sigma:.4f}
- Z-Score: {st_.z_now:+.3f} | Δz (1 pregão): {st_.dz_now:+.3f} ({mom_txt})
- Δz de ruído de fundo (mediana): {st_.dz_noise:.3f}
- Meia-vida de reversão: {hl_txt}
- Cointegração (Engle-Granger, p): {coint_txt}
- ADF do spread (p): {adf_txt}
- Correlação de retornos ({st_.corr_win}p): {corr_txt}
- Beta OLS em log (informativo): {st_.beta_now:.3f}
- z de break-even (custo+aluguel em unidades de z): {z_be:.2f}

GATILHOS CONFIGURADOS:
- Monitorar: |z| ≥ {cfg['z_monitor']:.2f} · Δz ≥ {cfg['dz_monitor']:.2f} · corr ≥ {cfg['corr_min']:.2f}
- Alerta máximo: |z| ≥ {cfg['z_alert']:.2f} · Δz ≥ {cfg['dz_alert']:.2f}
- Saída: |z| ≤ {cfg['z_exit']:.2f} ou cruzamento do zero · holding máx {cfg['max_hold']}p
- Sinal atual do sistema: {sig.label}{(' — ' + ', '.join(sig.falhas) + ' não passou') if sig.falhas else ''}

Estruture a resposta em:
1. Leitura do z e do Δz — o par está esticando ou revertendo?
2. Qualidade estatística — a meia-vida é compatível com o holding máximo? Os testes concordam?
3. O edge sobrevive ao custo? Compare a amplitude esperada (z atual até a saída) com o z de break-even.
4. Riscos concretos (liquidez, aluguel/BTC, evento societário, quebra da relação).
5. Resumo executivo em 2-3 linhas."""

    if conciso:
        prompt += "\n\nResponda de forma concisa — será lido no celular. Máximo 4 blocos curtos."

    with client.messages.stream(
        model="claude-sonnet-5",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        final = stream.get_final_message()

    return "\n".join(b.text for b in final.content if b.type == "text").strip()
