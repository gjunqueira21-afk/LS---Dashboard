import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint, adfuller
import anthropic
import requests
import time
from datetime import datetime

st.set_page_config(page_title="LS Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ─── Design System — tokens de cor (usados no CSS e nos gráficos Plotly) ────────
THEME = {
    "bg":         "#0f1117",  # fundo do app
    "surface":    "#1a1d29",  # cards
    "surface_2":  "#232734",  # elevado
    "border":     "#2a2e3d",
    "text":       "#e6e8ef",  # primário
    "text_muted": "#9ca3b4",  # secundário
    "accent":     "#7c6cf0",  # violeta/índigo — acento único
    "pos":        "#34d399",  # verde — reservado ao sinal
    "neg":        "#f87171",  # vermelho — reservado ao sinal
    "warn":       "#fbbf24",
    "grid":       "#232734",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: {THEME['bg']};
    color: {THEME['text']};
}}
.stApp {{ background: {THEME['bg']}; }}

h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; color: {THEME['text']}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; }}
h2, h3 {{ font-weight: 600; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {THEME['text_muted']}; }}

/* Header de produto */
.ls-header {{
    display: flex; align-items: baseline; gap: 0.9rem;
    padding: 0.4rem 0 1.1rem 0; margin-bottom: 0.4rem;
    border-bottom: 1px solid {THEME['border']};
}}
.ls-header .ls-logo {{
    font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; color: {THEME['text']};
}}
.ls-header .ls-logo .dot {{ color: {THEME['accent']}; }}
.ls-header .ls-tag {{ font-size: 0.9rem; color: {THEME['text_muted']}; font-weight: 400; }}

/* Cards de métrica premium */
div[data-testid="stMetric"] {{
    background: {THEME['surface']};
    border: 1px solid {THEME['border']};
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.25);
}}
div[data-testid="stMetric"] label {{ color: {THEME['text_muted']} !important; }}
div[data-testid="stMetricValue"] {{
    font-variant-numeric: tabular-nums; font-weight: 600; color: {THEME['text']};
}}

/* Containers com borda */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {THEME['surface']};
    border-radius: 14px;
}}

/* Signal box (herói) */
.signal-box {{
    padding: 1.3rem 1.5rem; border-radius: 16px; font-weight: 700;
    font-size: 1.5rem; margin: 0.2rem 0 0.4rem 0;
    border: 1px solid {THEME['border']}; text-align: left;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}}
.signal-box .sig-sub {{
    display: block; font-size: 0.9rem; font-weight: 400; margin-top: 0.35rem;
    opacity: 0.85;
}}
.sig-long    {{ background: linear-gradient(135deg, #16241d, {THEME['surface']}); color: {THEME['pos']}; border-color: rgba(52,211,153,0.35); }}
.sig-short   {{ background: linear-gradient(135deg, #26161a, {THEME['surface']}); color: {THEME['neg']}; border-color: rgba(248,113,113,0.35); }}
.sig-neutral {{ background: linear-gradient(135deg, {THEME['surface_2']}, {THEME['surface']}); color: {THEME['text']}; border-color: {THEME['border']}; }}

/* Estado vazio */
.empty-state {{
    background: {THEME['surface']};
    border: 1px solid {THEME['border']};
    border-radius: 16px;
    padding: 2.4rem 1.5rem; text-align: center; margin: 1rem 0;
}}
.empty-state .es-icon {{ font-size: 2.4rem; }}
.empty-state .es-title {{ font-size: 1.15rem; font-weight: 600; color: {THEME['text']}; margin: 0.5rem 0 0.3rem 0; }}
.empty-state .es-text {{ font-size: 0.92rem; color: {THEME['text_muted']}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.3rem; border-bottom: 1px solid {THEME['border']}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 10px 10px 0 0; color: {THEME['text_muted']};
    padding: 0.5rem 1rem; font-weight: 500;
}}
.stTabs [aria-selected="true"] {{ color: {THEME['text']}; border-bottom: 2px solid {THEME['accent']}; }}

/* Botões */
.stButton button {{ border-radius: 10px; font-weight: 600; border: 1px solid {THEME['border']}; }}
.stButton button[kind="primary"] {{ background: {THEME['accent']}; border-color: {THEME['accent']}; }}
.stButton button[kind="primary"]:hover {{ filter: brightness(1.08); }}
.stButton button:hover {{ border-color: {THEME['accent']}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {THEME['surface']}; border-right: 1px solid {THEME['border']}; }}
section[data-testid="stSidebar"] .stExpander {{ border-radius: 12px; }}

/* Dataframe */
div[data-testid="stDataFrame"] {{ border: 1px solid {THEME['border']}; border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)


# ─── Helpers de UI (camada de apresentação) ────────────────────────────────────
def render_empty_state(icon, titulo, texto):
    st.markdown(
        f"""<div class="empty-state">
        <div class="es-icon">{icon}</div>
        <div class="es-title">{titulo}</div>
        <div class="es-text">{texto}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_signal(sig, cls, sig_desc):
    box_cls = {"signal-long": "sig-long", "signal-short": "sig-short", "signal-neutral": "sig-neutral"}.get(cls, "sig-neutral")
    st.markdown(
        f"""<div class="signal-box {box_cls}">⚡ {sig}
        <span class="sig-sub">{sig_desc}</span></div>""",
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """<div class="ls-header">
        <span class="ls-logo">LS<span class="dot"> Dashboard</span></span>
        <span class="ls-tag">Pairs trading · Z-Score · Fundamentos B3</span>
        </div>""",
        unsafe_allow_html=True,
    )


default_api_key   = st.secrets.get("ANTHROPIC_API_KEY", "")
default_brapi_key = st.secrets.get("BRAPI_TOKEN", "")

with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    mode = st.radio("Painel", ["📈 Long / Short", "🏦 Fundamentos"], label_visibility="collapsed")
    st.divider()

    if mode == "📈 Long / Short":
        st.subheader("Ativos")
        col1, col2 = st.columns(2)
        with col1:
            long_ticker  = st.text_input("LONG",  value="PETR4.SA").upper().strip()
        with col2:
            short_ticker = st.text_input("SHORT", value="VALE3.SA").upper().strip()

        st.subheader("Período")
        # 252 pregões de correlação exigem histórico longo → default 2 anos
        period_map   = {"6 meses": "6mo", "1 ano": "1y", "2 anos": "2y", "3 anos": "3y", "5 anos": "5y"}
        period_label = st.selectbox("Histórico", list(period_map.keys()), index=2)
        period       = period_map[period_label]
        interval     = "1d"  # estratégia definida em pregões (diário)

        run_btn      = st.button("▶ Analisar", use_container_width=True, type="primary")

        with st.expander("⚙️ Setup avançado", expanded=False):
            st.subheader("Ratio")
            use_log = st.checkbox("Usar log ratio  ln(LONG/SHORT)", value=True)

            st.subheader("Janelas (pregões)")
            mean_win = st.number_input("Média do ratio (z-score)",    5, 504, 63)
            vol_win  = st.number_input("Bandas / volatilidade (z-score)", 5, 504, 63)
            corr_win = st.number_input("Correlação",                 20, 504, 252)
            st.caption("O Z-Score usa a média e a volatilidade acima.")

            st.subheader("🟡 Monitorar")
            z_monitor  = st.slider("|Z-Score| ≥",  0.5, 3.0, 1.10, 0.05)
            dz_monitor = st.slider("Δz (1 pregão) ≥", 0.0, 0.5, 0.05, 0.01)
            corr_min   = st.slider("Correlação ≥", 0.0, 1.0, 0.75, 0.05)

            st.subheader("🔴 Alerta Máximo")
            z_alert  = st.slider("|Z-Score| ≥ ", 0.5, 3.0, 1.50, 0.05)
            dz_alert = st.slider("Δz (1 pregão) ≥ ", 0.0, 0.5, 0.15, 0.01)

            st.subheader("Saída / Convergência")
            z_exit   = st.slider("Convergência |Z| ≤", 0.0, 1.0, 0.40, 0.05)
            max_hold = st.number_input("Máx. holding (pregões)", 5, 252, 63)

            auto_refresh = st.toggle("Auto-refresh (30s)", value=False)

        with st.expander("🔑 Conexões", expanded=False):
            api_key = st.text_input("Anthropic API Key", value=default_api_key, type="password", placeholder="sk-ant-...")

    else:
        st.subheader("Empresas")
        PRESETS_FUND = {
            "Varejo BR":       "MGLU3\nAMER3\nVIVA3\nALLD3\nRENNER3",
            "Bancos BR":       "ITUB4\nBBDC4\nBBAS3\nSANB11\nBRAY3",
            "Petro & Energia": "PETR4\nVALE3\nPRIO3\nRRPH3\nGBIO3",
            "Real Estate":     "CYRE3\nMRVE3\nEZTC3\nDIRR3\nTRIS3",
            "Custom":          "",
        }
        fund_preset = st.selectbox("Grupo pré-definido", list(PRESETS_FUND.keys()))
        default_tickers = PRESETS_FUND[fund_preset]
        fund_tickers_raw = st.text_area(
            "Tickers (um por linha, sem .SA)",
            value=default_tickers,
            height=160,
            help="Ex: PETR4, VALE3, ITUB4"
        )
        load_btn = st.button("🔍 Carregar Fundamentos", use_container_width=True, type="primary")

        with st.expander("🔑 Conexões", expanded=False):
            brapi_key = st.text_input("brapi.dev Token", value=default_brapi_key, type="password", placeholder="token (opcional)")
            st.caption("Dados: brapi.dev (B3 — dados confiáveis)")

@st.cache_data(ttl=30)
def fetch(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float, name=ticker)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return close.dropna()

def compute_ratio(long, short, use_log=True):
    df = pd.concat([long, short], axis=1).dropna()
    r = df.iloc[:, 0] / df.iloc[:, 1]
    if use_log:
        r = np.log(r)
    return r

def zscore(series, mean_win, vol_win):
    m = series.rolling(mean_win).mean()
    s = series.rolling(vol_win).std()
    return (series - m) / s

def hedge_ratio(long, short):
    df = pd.concat([long, short], axis=1).dropna()
    return float(np.polyfit(df.iloc[:, 1].values, df.iloc[:, 0].values, 1)[0])

def cointegration_test(long, short):
    df = pd.concat([long, short], axis=1).dropna()
    _, p, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    return p

def adf_test(series):
    return adfuller(series.dropna())[1]

def rolling_corr(long, short, window):
    df = pd.concat([long, short], axis=1).dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])

def classify_signal(z, dz, corr, cfg, zero_cross=False):
    """Classifica em 3 níveis: Alerta Máximo, Monitorar, Convergência ou Neutro.
    Retorna (rótulo, classe_css, descrição)."""
    az, adz = abs(z), abs(dz)
    side = "SHORT ratio → vender LONG / comprar SHORT" if z > 0 else "LONG ratio → comprar LONG / vender SHORT"
    cls  = "signal-short" if z > 0 else "signal-long"

    # Convergência / saída tem prioridade
    if az <= cfg["z_exit"] or zero_cross:
        motivo = "cruzou o zero" if zero_cross and az > cfg["z_exit"] else f"|z| ≤ {cfg['z_exit']:.2f}"
        return "⚪ CONVERGÊNCIA → encerrar / realizar", "signal-neutral", f"Zona de saída ({motivo})"

    if az >= cfg["z_alert"] and adz >= cfg["dz_alert"] and corr >= cfg["corr_min"]:
        return f"🔴 ALERTA MÁXIMO · {side}", cls, \
               f"|z| ≥ {cfg['z_alert']:.2f} · Δz ≥ {cfg['dz_alert']:.2f} · corr ≥ {cfg['corr_min']:.2f}"

    if az >= cfg["z_monitor"] and adz >= cfg["dz_monitor"] and corr >= cfg["corr_min"]:
        return f"🟡 MONITORAR · {side}", cls, \
               f"|z| ≥ {cfg['z_monitor']:.2f} · Δz ≥ {cfg['dz_monitor']:.2f} · corr ≥ {cfg['corr_min']:.2f}"

    return "⚪ NEUTRO · aguardar", "signal-neutral", "Fora dos gatilhos de entrada"

def analyze_with_claude(api_key, long_t, short_t, ratio, z, coint_p, adf_p, hedge, corr_last,
                        mean_win=63, dz=0.0, cfg=None, use_log=True, sinal="—"):
    client = anthropic.Anthropic(api_key=api_key)
    cfg = cfg or {}
    ratio_lbl = "log-ratio ln(L/S)" if use_log else "ratio L/S"
    prompt = f"""Você é um analista quantitativo especialista em estratégias Long/Short (pairs trading) na B3.
Analise o par {long_t} (LONG) x {short_t} (SHORT) com os dados abaixo e responda em português.

Setup (parâmetros da estratégia):
- {ratio_lbl}, janelas em pregões: média={mean_win}, z-score e volatilidade conforme configurado.
- Monitorar: |z| ≥ {cfg.get('z_monitor', 1.10):.2f}, Δz ≥ {cfg.get('dz_monitor', 0.05):.2f}, corr ≥ {cfg.get('corr_min', 0.75):.2f}
- Alerta Máximo: |z| ≥ {cfg.get('z_alert', 1.50):.2f}, Δz ≥ {cfg.get('dz_alert', 0.15):.2f}
- Saída/convergência: |z| ≤ {cfg.get('z_exit', 0.40):.2f} ou cruzamento do zero; holding máx {cfg.get('max_hold', 63)} pregões.

Situação atual:
- {ratio_lbl} atual: {ratio.iloc[-1]:.4f} | Média ({mean_win}p): {ratio.rolling(mean_win).mean().iloc[-1]:.4f}
- Z-Score atual: {z.iloc[-1]:.3f} | Δz (1 pregão): {dz:+.3f} | Máx: {z.max():.3f} | Mín: {z.min():.3f}
- Cointegração p-valor: {coint_p:.4f} → {"✅ cointegrado" if coint_p < 0.05 else "❌ não cointegrado"}
- ADF spread p-valor: {adf_p:.4f} → {"✅ estacionário" if adf_p < 0.05 else "❌ não estacionário"}
- Hedge ratio (β): {hedge:.4f} | Correlação ({cfg.get('corr_win', 252)}p): {corr_last:.3f}
- Sinal atual do sistema: {sinal}

Estruture sua resposta em:
1. Leitura do Z-Score e do Δz (o par está esticando ou revertendo?)
2. Qualidade estatística (cointegração, correlação, estacionaridade)
3. Sinal operacional — coerente com Monitorar / Alerta Máximo / Convergência acima?
4. Riscos e checklist (liquidez, spread, aluguel, custo)
5. Resumo executivo (2-3 linhas)"""
    with client.messages.stream(model="claude-sonnet-4-6", max_tokens=1200,
                                 messages=[{"role": "user", "content": prompt}]) as stream:
        return stream.get_final_text()

@st.cache_data(ttl=300)
def fetch_brapi(ticker: str, token: str) -> dict:
    t = ticker.upper().replace(".SA", "").strip()
    url = f"https://brapi.dev/api/quote/{t}"
    params = {"modules": "defaultKeyStatistics,financialData,summaryProfile"}
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        params["token"] = token
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        data = r.json()
        if "results" in data and data["results"]:
            return data["results"][0]
        return {"_error": data.get("message") or data.get("error") or f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"_error": str(e)}

def safe_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}%"

def safe_num(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"

def fmt_market_cap(val):
    if val is None:
        return "—"
    b = val / 1e9
    if b >= 1:
        return f"R${b:.1f}B"
    m = val / 1e6
    return f"R${m:.0f}M"

def parse_fundamentals(ticker: str, token: str) -> dict:
    info = fetch_brapi(ticker, token)
    if not info or "_error" in info:
        return {"_error": (info or {}).get("_error", "sem resposta"), "Ticker": ticker}

    ks = info.get("defaultKeyStatistics") or {}
    fd = info.get("financialData") or {}
    sp = info.get("summaryProfile") or {}

    div_yield_raw = ks.get("dividendYield") or info.get("dividendYield")
    if div_yield_raw is not None and div_yield_raw > 1:
        div_yield = div_yield_raw
    elif div_yield_raw is not None:
        div_yield = div_yield_raw * 100
    else:
        div_yield = None

    dte_raw = fd.get("debtToEquity") or ks.get("debtToEquity")

    def to_pct(v):
        if v is None: return None
        return v * 100

    return {
        "Ticker":         ticker.upper().replace(".SA", ""),
        "Nome":           sp.get("longName") or ks.get("longName") or info.get("longName", ticker),
        "Setor":          sp.get("sector") or info.get("sector", "—"),
        "P/L":            ks.get("trailingPE") or info.get("trailingPE"),
        "P/L Fwd":        ks.get("forwardPE")  or info.get("forwardPE"),
        "P/VP":           ks.get("priceToBook") or info.get("priceToBook"),
        "EV/EBITDA":      ks.get("enterpriseToEbitda") or info.get("enterpriseToEbitda"),
        "ROE (%)":        to_pct(fd.get("returnOnEquity") or ks.get("returnOnEquity")),
        "ROA (%)":        to_pct(fd.get("returnOnAssets") or ks.get("returnOnAssets")),
        "Marg. Liq. (%)": to_pct(fd.get("profitMargins") or ks.get("profitMargins")),
        "Marg. EBITDA (%)": to_pct(fd.get("ebitdaMargins") or ks.get("ebitdaMargins")),
        "Div. Yield (%)": div_yield,
        "Dív./PL":        dte_raw,
        "Liq. Corrente":  fd.get("currentRatio"),
        "Cresc. Receita (%)": to_pct(fd.get("revenueGrowth")),
        "Cresc. Lucro (%)":   to_pct(fd.get("earningsGrowth") or ks.get("earningsGrowth")),
        "Beta":           ks.get("beta") or info.get("beta"),
        "Market Cap":     ks.get("marketCap") or info.get("marketCap"),
    }

render_header()

if mode == "📈 Long / Short":

    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if run_btn:
        st.session_state["ls_ran"] = True
        st.session_state["ls_long"] = long_ticker
        st.session_state["ls_short"] = short_ticker

    already_ran = (
        st.session_state.get("ls_ran")
        and st.session_state.get("ls_long") == long_ticker
        and st.session_state.get("ls_short") == short_ticker
    )

    if not (run_btn or auto_refresh or already_ran):
        render_empty_state("📈", "Pronto para analisar",
                           "Configure os ativos na barra lateral e clique em ▶ Analisar.")
        st.stop()

    if not long_ticker or not short_ticker:
        render_empty_state("⚠️", "Tickers ausentes", "Preencha os dois tickers antes de analisar.")
        st.stop()

    with st.spinner(f"Baixando dados de {long_ticker} e {short_ticker}…"):
        long_data  = fetch(long_ticker,  period, interval)
        short_data = fetch(short_ticker, period, interval)

    if long_data.empty or short_data.empty:
        render_empty_state("🚫", "Sem dados", "Não foi possível baixar dados. Verifique os tickers.")
        st.stop()

    cfg = {
        "z_monitor": z_monitor, "dz_monitor": dz_monitor, "corr_min": corr_min,
        "z_alert": z_alert, "dz_alert": dz_alert,
        "z_exit": z_exit, "max_hold": max_hold, "corr_win": corr_win,
    }

    ratio_series = compute_ratio(long_data, short_data, use_log)
    z_series     = zscore(ratio_series, mean_win, vol_win)
    corr_series  = rolling_corr(long_data, short_data, corr_win)
    hedge        = hedge_ratio(long_data, short_data)

    try:    coint_p = cointegration_test(long_data, short_data)
    except: coint_p = 1.0
    try:
        spread = long_data - hedge * short_data
        adf_p  = adf_test(spread)
    except: adf_p = 1.0

    z_clean = z_series.dropna()
    if z_clean.empty:
        render_empty_state("📏", "Histórico insuficiente",
                           "Janelas maiores que o histórico disponível. Reduza as janelas ou aumente o período.")
        st.stop()
    z_now    = float(z_clean.iloc[-1])
    z_prev   = float(z_clean.iloc[-2]) if len(z_clean) > 1 else z_now
    dz_now   = z_now - z_prev
    zero_cross = (z_now == 0) or (z_prev != 0 and np.sign(z_now) != np.sign(z_prev))
    corr_now = float(corr_series.dropna().iloc[-1]) if not corr_series.dropna().empty else 0.0

    ratio_lbl = "Log-Ratio" if use_log else "Ratio"
    st.subheader(f"{long_ticker}  ×  {short_ticker}")

    # SINAL — card herói full-width
    sig, cls, sig_desc = classify_signal(z_now, dz_now, corr_now, cfg, zero_cross)
    render_signal(sig, cls, sig_desc)

    # Checklist do Alerta Máximo
    if sig.startswith("🔴"):
        with st.container(border=True):
            st.markdown("**✅ Checklist Alerta Máximo — validar antes de executar**")
            chk1, chk2 = st.columns(2)
            with chk1:
                st.checkbox("Liquidez OK", key="chk_liq")
                st.checkbox("Spread OK", key="chk_spread")
            with chk2:
                st.checkbox(f"Cointegração plausível (p={coint_p:.3f})", value=coint_p < 0.05, key="chk_coint")
                st.checkbox("Aluguel OK / checado", key="chk_alug")

    # 6 métricas → 2 grupos
    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.markdown("**Preço & Sinal**")
            a1, a2, a3 = st.columns(3)
            a1.metric(ratio_lbl,       f"{ratio_series.iloc[-1]:.4f}")
            a2.metric("Z-Score",       f"{z_now:.3f}")
            a3.metric("Δz (1 pregão)", f"{dz_now:+.3f}")
    with g2:
        with st.container(border=True):
            st.markdown("**Qualidade estatística**")
            b1, b2, b3 = st.columns(3)
            b1.metric(f"Correlação {corr_win}p", f"{corr_now:.3f}", delta="OK" if corr_now >= corr_min else "Baixa")
            b2.metric("Cointegração p", f"{coint_p:.4f}", delta="✅ OK" if coint_p < 0.05 else "❌ Fraco")
            b3.metric("ADF spread p",   f"{adf_p:.4f}",   delta="✅ OK" if adf_p < 0.05 else "❌ Fraco")

    st.divider()

    tab_graf, tab_stats, tab_ia, tab_manual = st.tabs(["📈 Gráficos", "📋 Estatísticas", "🤖 Análise IA", "ℹ️ Manual"])

    with tab_graf:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.30, 0.25, 0.25, 0.20],
            subplot_titles=["Preços normalizados", f"{ratio_lbl} {long_ticker}/{short_ticker} (média {mean_win}p, bandas {vol_win}p)",
                            f"Z-Score ({mean_win}p/{vol_win}p)", f"Correlação rolling {corr_win}p"],
            vertical_spacing=0.06)

        long_norm  = long_data  / long_data.iloc[0]  * 100
        short_norm = short_data / short_data.iloc[0] * 100
        fig.add_trace(go.Scatter(x=long_norm.index,  y=long_norm,  name=long_ticker,  line=dict(color=THEME["pos"], width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=short_norm.index, y=short_norm, name=short_ticker, line=dict(color=THEME["neg"], width=1.5)), row=1, col=1)

        roll_mean = ratio_series.rolling(mean_win).mean()
        roll_std  = ratio_series.rolling(vol_win).std()
        fig.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, name=ratio_lbl, line=dict(color=THEME["accent"], width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, name="Média", line=dict(color=THEME["text_muted"], width=1, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean + z_alert * roll_std,   name=f"+{z_alert:.2f}σ (alerta)",   line=dict(color=THEME["neg"], width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean + z_monitor * roll_std, name=f"+{z_monitor:.2f}σ (monitor)", line=dict(color=THEME["warn"], width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean - z_monitor * roll_std, name=f"-{z_monitor:.2f}σ (monitor)", line=dict(color=THEME["warn"], width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean - z_alert * roll_std,   name=f"-{z_alert:.2f}σ (alerta)",   line=dict(color=THEME["accent"], width=1, dash="dot")), row=2, col=1)

        colors_z = [THEME["neg"] if v >= z_alert else THEME["warn"] if v >= z_monitor else
                    THEME["accent"] if v <= -z_alert else THEME["pos"] if v <= -z_monitor else THEME["text_muted"]
                    for v in z_series.fillna(0)]
        fig.add_trace(go.Bar(x=z_series.index, y=z_series, name="Z-Score", marker_color=colors_z, opacity=0.8), row=3, col=1)
        for level, dash in [(z_alert, "solid"), (-z_alert, "solid"), (z_monitor, "dot"), (-z_monitor, "dot"), (z_exit, "dash"), (-z_exit, "dash")]:
            fig.add_hline(y=level, line_dash=dash, line_color=THEME["text_muted"], opacity=0.5, row=3, col=1)
        fig.add_hline(y=0, line_color=THEME["text"], opacity=0.3, row=3, col=1)

        fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series, name="Correlação",
            line=dict(color=THEME["warn"], width=1.5), fill="tozeroy", fillcolor="rgba(251,191,36,0.10)"), row=4, col=1)
        fig.add_hline(y=corr_min, line_dash="dot", line_color=THEME["pos"], opacity=0.6, row=4, col=1)

        fig.update_layout(height=750, paper_bgcolor=THEME["surface"], plot_bgcolor=THEME["surface"],
            font=dict(color=THEME["text"], family="Inter"), legend=dict(bgcolor=THEME["surface_2"]),
            margin=dict(l=0, r=0, t=40, b=0))
        fig.update_xaxes(gridcolor=THEME["grid"])
        fig.update_yaxes(gridcolor=THEME["grid"])
        st.plotly_chart(fig, use_container_width=True)

    with tab_stats:
        st.subheader("📋 Estatísticas do Spread")
        df_stats = pd.DataFrame({
            "Métrica": [f"{ratio_lbl} atual", f"{ratio_lbl} médio ({mean_win}p)", "Z-Score atual", "Δz (1 pregão)",
                        "Z-Score máx", "Z-Score mín", "Hedge ratio (β)", f"Correlação ({corr_win}p)",
                        "Coint. p-valor", "ADF p-valor", "Holding máx (pregões)"],
            "Valor":   [f"{ratio_series.iloc[-1]:.4f}", f"{ratio_series.rolling(mean_win).mean().iloc[-1]:.4f}",
                        f"{z_now:.3f}", f"{dz_now:+.3f}",
                        f"{z_series.max():.3f}", f"{z_series.min():.3f}", f"{hedge:.4f}", f"{corr_now:.3f}",
                        f"{coint_p:.4f}", f"{adf_p:.4f}", f"{max_hold}"],
            "Status":  ["—","—",
                        "🔴 Vender ratio" if z_now >= z_monitor else "🟢 Comprar ratio" if z_now <= -z_monitor else "⚪ Neutro",
                        "↗ esticando" if (dz_now > 0) == (z_now > 0) and abs(dz_now) >= dz_monitor else "↘ revertendo" if abs(dz_now) >= dz_monitor else "—",
                        "—","—","—",
                        "✅ OK" if corr_now >= corr_min else "❌ Baixa",
                        "✅ Cointegrado" if coint_p < 0.05 else "❌ Não cointegrado",
                        "✅ Estacionário" if adf_p < 0.05 else "❌ Não estacionário", "—"],
        })
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

    with tab_ia:
        st.subheader("🤖 Análise Claude AI")
        if not api_key:
            render_empty_state("🔑", "API Key necessária",
                               "Insira sua Anthropic API Key em Conexões (barra lateral) para ativar a análise via Claude.")
        else:
            analysis_key = f"claude_analysis_{long_ticker}_{short_ticker}"
            if st.button("🧠 Gerar análise com Claude", type="secondary"):
                with st.spinner("Claude analisando o par…"):
                    try:
                        result = analyze_with_claude(api_key, long_ticker, short_ticker,
                                                     ratio_series, z_series, coint_p, adf_p, hedge, corr_now,
                                                     mean_win=mean_win, dz=dz_now, cfg=cfg, use_log=use_log, sinal=sig)
                        st.session_state[analysis_key] = result
                    except anthropic.AuthenticationError:
                        st.error("API Key inválida.")
                    except Exception as e:
                        st.error(f"Erro: {e}")
            if analysis_key in st.session_state:
                st.markdown(st.session_state[analysis_key])

    with tab_manual:
        st.markdown("""
**📐 Log-Ratio** — `ln(preço LONG / preço SHORT)`. O log torna as variações simétricas (subir e cair têm o mesmo peso) e estabiliza a variância — padrão em pairs trading.

**📊 Z-Score** — Distância do log-ratio em relação à média móvel (63 pregões), em desvios padrão (vol 63 pregões). Coração da estratégia.

**🔀 Δz (delta-z)** — Variação do Z-Score em 1 pregão. Confirma **momentum**: um |z| alto *com* Δz na mesma direção indica que o par ainda está esticando; Δz contrário sugere reversão a caminho.

**🔗 Correlação (252 pregões)** — Sincronia de longo prazo entre os ativos. O gatilho exige **≥ 0,75** para operar.

**🧪 Cointegração p-valor** — Teste Engle-Granger. p < 0,05 confirma relação estável de longo prazo (tendência a convergir).

**📉 ADF spread p** — Estacionaridade do spread. p < 0,05 = spread oscila em torno de média fixa.

**⚖️ Hedge Ratio (β)** — Quanto do SHORT por unidade de LONG para neutralizar o risco de mercado.

---
**⚡ Níveis de Sinal (setup do backtest — meanWin 63 / volWin 63 / corrWin 252 / maxHold 63):**

- **🟡 Monitorar** — `|z| ≥ 1,10` · `Δz ≥ 0,05` · `corr ≥ 0,75`. Par entrando em zona operável; acompanhar de perto.
- **🔴 Alerta Máximo** — `|z| ≥ 1,50` · `Δz ≥ 0,15` · `corr ≥ 0,75` + checklist (liquidez, spread, cointegração, aluguel). Sinal forte de entrada.
- **⚪ Convergência / Saída** — `|z| ≤ 0,40` **ou** cruzamento do zero. Encerrar / realizar. Holding máximo: **63 pregões**.

> ⚠️ Rodar inicialmente em **produção sombra** antes de execução real. Validar custos, aluguel, spread, slippage e walk-forward.
        """)

    st.caption(f"Dados via Yahoo Finance · Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


else:
    st.subheader("🏦 Fundamentos — Comparação de Empresas")
    st.caption("Dados via brapi.dev · Fonte: B3 / CVM")

    if not load_btn:
        render_empty_state("🏦", "Comparar fundamentos",
                           "Selecione um grupo ou digite os tickers na barra lateral e clique em 🔍 Carregar Fundamentos.")
        st.stop()

    tickers = [t.strip().upper().replace(".SA", "") for t in fund_tickers_raw.splitlines() if t.strip()]
    if not tickers:
        render_empty_state("⚠️", "Nenhum ticker", "Nenhum ticker informado.")
        st.stop()

    rows = []
    errors = []
    prog = st.progress(0, text="Buscando fundamentos…")
    for i, t in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"Carregando {t}…")
        row = parse_fundamentals(t, brapi_key)
        if row and "_error" not in row:
            rows.append(row)
        else:
            err_msg = (row or {}).get("_error", "erro desconhecido")
            errors.append(f"{t} ({err_msg})")
    prog.empty()

    if errors:
        st.warning("Não foi possível carregar:\n\n- " + "\n- ".join(errors))

    if not rows:
        render_empty_state("🚫", "Sem dados", "Nenhum dado encontrado. Verifique os tickers e o token da brapi.dev.")
        st.stop()

    df = pd.DataFrame(rows)

    DISPLAY_COLS = [
        "Ticker", "Nome", "Setor",
        "P/L", "P/L Fwd", "P/VP", "EV/EBITDA",
        "ROE (%)", "ROA (%)", "Marg. Liq. (%)", "Marg. EBITDA (%)",
        "Div. Yield (%)", "Dív./PL", "Liq. Corrente",
        "Cresc. Receita (%)", "Cresc. Lucro (%)",
        "Beta", "Market Cap",
    ]

    df_display = df[DISPLAY_COLS].copy()
    pct_cols  = ["ROE (%)", "ROA (%)", "Marg. Liq. (%)", "Marg. EBITDA (%)", "Div. Yield (%)", "Cresc. Receita (%)", "Cresc. Lucro (%)"]
    num_cols  = ["P/L", "P/L Fwd", "P/VP", "EV/EBITDA", "Dív./PL", "Liq. Corrente", "Beta"]

    def fmt_cell(v, is_pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if is_pct:
            return f"{v:.2f}%"
        return f"{v:.2f}"

    for col in pct_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: fmt_cell(x, is_pct=True))
    for col in num_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: fmt_cell(x))
    df_display["Market Cap"] = df["Market Cap"].apply(fmt_market_cap)

    tab_tab, tab_comp, tab_radar, tab_emp, tab_man = st.tabs(
        ["📋 Tabela", "📊 Comparação", "🕸️ Radar", "🔍 Empresa", "ℹ️ Manual"])

    with tab_tab:
        st.subheader("📋 Tabela Comparativa")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab_comp:
        st.subheader("📊 Comparação Visual")

        CHART_METRICS = {
            "P/L": ("P/L", False),
            "P/VP": ("P/VP", False),
            "ROE (%)": ("ROE (%)", True),
            "Marg. Liq. (%)": ("Marg. Liq. (%)", True),
            "Div. Yield (%)": ("Div. Yield (%)", True),
            "EV/EBITDA": ("EV/EBITDA", False),
            "Dív./PL": ("Dív./PL", False),
            "Cresc. Receita (%)": ("Cresc. Receita (%)", True),
        }

        selected_metrics = st.multiselect(
            "Métricas para comparar",
            list(CHART_METRICS.keys()),
            default=["ROE (%)", "Marg. Liq. (%)", "Div. Yield (%)", "P/L"],
        )

        if selected_metrics:
            cols_per_row = 2
            metric_chunks = [selected_metrics[i:i+cols_per_row] for i in range(0, len(selected_metrics), cols_per_row)]
            COLORS = [THEME["accent"], THEME["warn"], THEME["text_muted"], THEME["text"], THEME["pos"], THEME["neg"]]

            for chunk in metric_chunks:
                row_cols = st.columns(len(chunk))
                for ci, metric in enumerate(chunk):
                    col_name, _ = CHART_METRICS[metric]
                    vals = df[col_name].tolist()
                    labels = df["Ticker"].tolist()
                    y = []
                    x = []
                    for label, v in zip(labels, vals):
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            y.append(float(v))
                            x.append(label)
                    if not y:
                        continue
                    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(x))]
                    fig_bar = go.Figure(go.Bar(
                        x=x, y=y,
                        marker_color=bar_colors,
                        text=[f"{v:.1f}" for v in y],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        title=metric,
                        height=320,
                        paper_bgcolor=THEME["surface"],
                        plot_bgcolor=THEME["surface"],
                        font=dict(color=THEME["text"], size=11, family="Inter"),
                        margin=dict(l=0, r=0, t=40, b=0),
                        showlegend=False,
                    )
                    fig_bar.update_xaxes(gridcolor=THEME["grid"])
                    fig_bar.update_yaxes(gridcolor=THEME["grid"])
                    with row_cols[ci]:
                        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_radar:
        RADAR_METRICS = ["ROE (%)", "ROA (%)", "Marg. Liq. (%)", "Marg. EBITDA (%)", "Div. Yield (%)"]
        radar_df = df[["Ticker"] + RADAR_METRICS].copy()

        # Converte para numérico e substitui NaN pela mediana da coluna (evita empresas sumindo no centro)
        for col in RADAR_METRICS:
            radar_df[col] = pd.to_numeric(radar_df[col], errors="coerce")
            median_val = radar_df[col].median()
            radar_df[col] = radar_df[col].fillna(median_val if not np.isnan(median_val) else 0)

        # Normaliza 0-100 via percentile rank (robusto a outliers — evita formas distorcidas)
        for col in RADAR_METRICS:
            if radar_df[col].nunique() > 1:
                radar_df[col] = radar_df[col].rank(pct=True) * 100
            else:
                radar_df[col] = 50.0

        st.subheader("🕸️ Radar — Qualidade (ranking relativo)")
        st.caption("Posição relativa de cada empresa dentro do grupo (0 = pior, 100 = melhor). Dados ausentes preenchidos pela mediana.")
        fig_radar = go.Figure()
        RADAR_COLORS = [THEME["accent"], THEME["warn"], THEME["text_muted"], THEME["text"], THEME["pos"], THEME["neg"]]
        for idx, row_ in radar_df.iterrows():
            vals = [round(float(row_[m]), 1) for m in RADAR_METRICS]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=RADAR_METRICS + [RADAR_METRICS[0]],
                fill="toself",
                name=row_["Ticker"],
                line=dict(color=RADAR_COLORS[idx % len(RADAR_COLORS)], width=2),
                opacity=0.55,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=THEME["surface_2"],
                radialaxis=dict(visible=True, range=[0, 100], color=THEME["text"], tickfont=dict(size=10)),
                angularaxis=dict(color=THEME["text"], tickfont=dict(size=11)),
            ),
            paper_bgcolor=THEME["surface"],
            font=dict(color=THEME["text"], family="Inter"),
            legend=dict(bgcolor=THEME["surface_2"], font=dict(size=12)),
            height=500,
            margin=dict(l=60, r=60, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_emp:
        st.subheader("🔍 Visão Individual")
        selected_ticker = st.selectbox("Selecionar empresa", df["Ticker"].tolist())
        row_ind = df[df["Ticker"] == selected_ticker].iloc[0]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            with st.container(border=True):
                st.markdown("**Identidade & Mercado**")
                st.markdown(f"**{row_ind['Nome']}**")
                st.caption(f"Setor: {row_ind['Setor']}")
                st.metric("Market Cap", fmt_market_cap(row_ind["Market Cap"]))
                st.metric("Beta", safe_num(row_ind["Beta"]))
        with col_b:
            with st.container(border=True):
                st.markdown("**Valuation**")
                st.metric("P/L",      safe_num(row_ind["P/L"]))
                st.metric("P/L Fwd",  safe_num(row_ind["P/L Fwd"]))
                st.metric("P/VP",     safe_num(row_ind["P/VP"]))
                st.metric("EV/EBITDA",safe_num(row_ind["EV/EBITDA"]))
        with col_c:
            with st.container(border=True):
                st.markdown("**Rentabilidade & Saúde**")
                st.metric("ROE",         safe_pct(row_ind["ROE (%)"]))
                st.metric("Marg. Líq.",  safe_pct(row_ind["Marg. Liq. (%)"]))
                st.metric("Div. Yield",  safe_pct(row_ind["Div. Yield (%)"]))
                st.metric("Dívida/PL",   safe_num(row_ind["Dív./PL"]))
                st.metric("Liq. Corrente", safe_num(row_ind["Liq. Corrente"]))
                st.metric("ROA",         safe_pct(row_ind["ROA (%)"]))
                st.metric("Cresc. Receita",  safe_pct(row_ind["Cresc. Receita (%)"]))
                st.metric("Cresc. Lucro",    safe_pct(row_ind["Cresc. Lucro (%)"]))

    with tab_man:
        st.markdown("""
**P/L (Preço/Lucro)** — Quantas vezes o lucro anual está embutido no preço. Menor é mais barato.

**P/L Fwd** — Mesmo que P/L, mas usando a estimativa de lucro futuro.

**P/VP (Preço/Valor Patrimonial)** — Quanto o mercado paga pelo patrimônio contábil. P/VP < 1 pode indicar desconto.

**EV/EBITDA** — Valor da empresa (incluindo dívida) dividido pelo EBITDA. Menor = mais barato.

**ROE (%)** — Retorno sobre o patrimônio líquido. Acima de 15% é considerado bom.

**ROA (%)** — Retorno sobre ativos totais. Mais conservador que o ROE.

**Margem Líquida (%)** — % da receita que vira lucro.

**Margem EBITDA (%)** — % da receita que vira lucro operacional.

**Div. Yield (%)** — Dividendos pagos no ano ÷ preço atual. Acima de 6% no Brasil é generoso.

**Dívida/PL** — Alavancagem financeira. Acima de 2x indica alta alavancagem.

**Liquidez Corrente** — Ativo circulante ÷ passivo circulante. Acima de 1.5 é saudável.

**Cresc. Receita / Lucro (%)** — Crescimento período sobre período.

**Beta** — Sensibilidade ao mercado. Beta < 1 = mais defensivo.
        """)

    st.caption(f"Dados via brapi.dev · Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
