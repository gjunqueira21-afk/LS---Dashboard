import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint, adfuller
import anthropic
import time
from datetime import datetime

st.set_page_config(page_title="LS Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Design system — Catppuccin Mocha
# LONG  = sempre verde  #a6e3a1   |   SHORT = sempre vermelho #f38ba8
# Acento = roxo #cba6f7           |   Texto = #cdd6f4 sobre #1e1e2e / #313244
# ---------------------------------------------------------------------------
C_LONG    = "#a6e3a1"
C_SHORT   = "#f38ba8"
C_ACCENT  = "#cba6f7"
C_MONITOR = "#fab387"
C_TEAL    = "#94e2d5"
C_YELLOW  = "#f9e2af"
C_TEXT    = "#cdd6f4"
C_MUTED   = "#a6adc8"
C_FAINT   = "#7f849c"
C_BG      = "#1e1e2e"
C_SURF    = "#313244"
C_CARD    = "#232334"
C_GRID    = "#313244"
C_NEUTRAL = "#45475a"

st.markdown("""
<style>
    /* ---------- base ---------- */
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
    h1, h2, h3 { letter-spacing: -0.02em; }

    /* ---------- legacy signal text (compat) ---------- */
    .signal-long    { color: #a6e3a1; font-weight: bold; font-size: 1.4rem; }
    .signal-short   { color: #f38ba8; font-weight: bold; font-size: 1.4rem; }
    .signal-neutral { color: #cdd6f4; font-weight: bold; font-size: 1.4rem; }
    .fund-header    { color: #cba6f7; font-size: 1.1rem; font-weight: bold; }

    /* ---------- hero ---------- */
    .ls-hero {
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 16px;
        background: linear-gradient(135deg, #26263a 0%, #1e1e2e 65%);
        border: 1px solid #313244; border-radius: 16px;
        padding: 20px 26px; margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
    }
    .ls-hero h1 { font-size: 1.55rem; font-weight: 800; margin: 0; color: #cdd6f4; }
    .ls-hero p  { margin: 4px 0 0; color: #a6adc8; font-size: 0.85rem; }
    .ls-hero-pair { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .chip {
        display: inline-block; padding: 6px 16px; border-radius: 999px;
        font-weight: 700; font-size: 0.92rem; white-space: nowrap;
    }
    .chip-long  { background: rgba(166,227,161,0.12); color: #a6e3a1; border: 1px solid rgba(166,227,161,0.45); }
    .chip-short { background: rgba(243,139,168,0.12); color: #f38ba8; border: 1px solid rgba(243,139,168,0.45); }
    .chip-fund  { background: rgba(203,166,247,0.12); color: #cba6f7; border: 1px solid rgba(203,166,247,0.45); }
    .pair-x { color: #585b70; font-weight: 800; font-size: 1.05rem; }

    /* ---------- signal banner ---------- */
    .signal-banner {
        display: flex; align-items: baseline; flex-wrap: wrap; gap: 6px 14px;
        border-radius: 14px; border: 1px solid #45475a;
        padding: 15px 22px; margin: 4px 0 14px;
        background: rgba(49,50,68,0.35);
        box-shadow: 0 4px 14px rgba(0,0,0,0.22);
    }
    .signal-banner .sb-label { font-size: 1.28rem; font-weight: 800; color: #cdd6f4; }
    .signal-banner .sb-desc  { font-size: 0.82rem; color: #a6adc8; }
    .signal-banner.signal-long  { background: rgba(166,227,161,0.09); border-color: rgba(166,227,161,0.50); }
    .signal-banner.signal-long .sb-label  { color: #a6e3a1; }
    .signal-banner.signal-short { background: rgba(243,139,168,0.09); border-color: rgba(243,139,168,0.50); }
    .signal-banner.signal-short .sb-label { color: #f38ba8; }

    /* ---------- metric cards ---------- */
    .metric-card {
        background: #232334; border: 1px solid #313244; border-left: 3px solid #585b70;
        border-radius: 12px; padding: 13px 15px; height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }
    .metric-label {
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.07em; color: #a6adc8; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 1.42rem; font-weight: 800; color: #cdd6f4;
        margin-top: 2px; font-variant-numeric: tabular-nums;
    }
    .metric-sub { font-size: 0.75rem; color: #7f849c; margin-top: 2px; }
    .metric-card.tone-green  { border-left-color: #a6e3a1; }
    .metric-card.tone-green  .metric-value { color: #a6e3a1; }
    .metric-card.tone-red    { border-left-color: #f38ba8; }
    .metric-card.tone-red    .metric-value { color: #f38ba8; }
    .metric-card.tone-purple { border-left-color: #cba6f7; }
    .metric-card.tone-purple .metric-value { color: #cba6f7; }
    .metric-card.tone-amber  { border-left-color: #f9e2af; }
    .metric-card.tone-amber  .metric-value { color: #f9e2af; }

    /* ---------- empty state ---------- */
    .empty-state {
        border: 1px dashed #45475a; border-radius: 14px;
        padding: 44px 24px; text-align: center;
        color: #a6adc8; background: rgba(49,50,68,0.25); font-size: 0.95rem;
    }
    .empty-state b { color: #cba6f7; }

    /* ---------- streamlit widgets ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #313244; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px 9px 0 0; padding: 8px 18px;
        color: #a6adc8; font-weight: 600; background: transparent;
    }
    .stTabs [aria-selected="true"] { color: #cba6f7 !important; background: rgba(203,166,247,0.08); }

    div[data-testid="stMetric"] {
        background: #232334; border: 1px solid #313244;
        border-radius: 12px; padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }
    div[data-testid="stMetric"] label { color: #a6adc8 !important; }

    div[data-testid="stExpander"] {
        border: 1px solid #313244; border-radius: 10px;
        background: rgba(49,50,68,0.25); overflow: hidden;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #313244; border-radius: 12px; overflow: hidden; }

    section[data-testid="stSidebar"] { background: #181825; border-right: 1px solid #313244; }
    section[data-testid="stSidebar"] hr { margin: 0.6rem 0; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #cba6f7 0%, #89b4fa 100%);
        color: #11111b; font-weight: 700; border: none; border-radius: 10px;
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, sub="", tone="neutral"):
    """Card HTML para métricas-chave (camada visual apenas)."""
    return (
        f'<div class="metric-card tone-{tone}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>'
    )

def style_fig(fig, height, top_margin=60, show_legend=True, legend_y=1.02):
    """Layout Plotly consistente com o tema (camada visual apenas)."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(49,50,68,0.25)",
        font=dict(color=C_TEXT, size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#181825", bordercolor=C_SURF, font=dict(color=C_TEXT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=legend_y, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        showlegend=show_legend,
        margin=dict(l=0, r=0, t=top_margin, b=0),
    )
    fig.update_xaxes(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_SURF)
    fig.update_yaxes(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor=C_SURF)
    return fig

PLOTLY_CONFIG = {"displayModeBar": False}

def get_secret(key, default=""):
    """Lê um secret com segurança — não quebra o app se não houver secrets.toml."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

default_api_key   = get_secret("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.title("⚙️ Configurações")
    mode = "📈 Long / Short"
    st.divider()

    if mode == "📈 Long / Short":
        st.subheader("Ativos")
        col1, col2 = st.columns(2)
        with col1:
            long_ticker  = st.text_input("LONG",  value="PETR4.SA").upper().strip()
        with col2:
            short_ticker = st.text_input("SHORT", value="VALE3.SA").upper().strip()

        # 252 pregões de correlação exigem histórico longo → default 2 anos
        period_map   = {"6 meses": "6mo", "1 ano": "1y", "2 anos": "2y", "3 anos": "3y", "5 anos": "5y"}
        period_label = st.selectbox("Histórico", list(period_map.keys()), index=2)
        period       = period_map[period_label]
        interval     = "1d"  # estratégia definida em pregões (diário)

        run_btn      = st.button("▶ Analisar", use_container_width=True, type="primary")
        auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
        st.divider()

        with st.expander("📐 Ratio & Janelas", expanded=False):
            use_log  = st.checkbox("Usar log ratio  ln(LONG/SHORT)", value=True)
            mean_win = st.number_input("Média do ratio (z-score)",    5, 504, 63)
            vol_win  = st.number_input("Bandas / volatilidade (z-score)", 5, 504, 63)
            corr_win = st.number_input("Correlação",                 20, 504, 252)
            st.caption("O Z-Score usa a média e a volatilidade acima. Janelas em pregões.")

        with st.expander("⚡ Gatilhos de Sinal", expanded=False):
            st.markdown("**🟡 Monitorar**")
            z_monitor  = st.slider("|Z-Score| ≥",  0.5, 3.0, 1.10, 0.05)
            dz_monitor = st.slider("Δz (1 pregão) ≥", 0.0, 0.5, 0.05, 0.01)
            corr_min   = st.slider("Correlação ≥", 0.0, 1.0, 0.75, 0.05)
            st.markdown("**🔴 Alerta Máximo**")
            z_alert  = st.slider("|Z-Score| ≥ ", 0.5, 3.0, 1.50, 0.05)
            dz_alert = st.slider("Δz (1 pregão) ≥ ", 0.0, 0.5, 0.15, 0.01)

        with st.expander("🏁 Saída / Convergência", expanded=False):
            z_exit   = st.slider("Convergência |Z| ≤", 0.0, 1.0, 0.40, 0.05)
            max_hold = st.number_input("Máx. holding (pregões)", 5, 252, 63)

        with st.expander("🔑 API", expanded=False):
            api_key = st.text_input("Anthropic API Key", value=default_api_key, type="password", placeholder="sk-ant-...")

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
        return "CONVERGÊNCIA → encerrar / realizar", "signal-neutral", f"Zona de saída ({motivo})"

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

if mode == "📈 Long / Short":
    # ---------- hero ----------
    st.markdown(f"""
    <div class="ls-hero">
        <div>
            <h1>📊 Long / Short Dashboard</h1>
            <p>Ratio em tempo real · Z-Score · Cointegração · Análise via Claude AI</p>
        </div>
        <div class="ls-hero-pair">
            <span class="chip chip-long">▲ LONG · {long_ticker or "—"}</span>
            <span class="pair-x">×</span>
            <span class="chip chip-short">▼ SHORT · {short_ticker or "—"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown(
            '<div class="empty-state">⚙️ Configure os ativos na barra lateral e clique em '
            '<b>▶ Analisar</b> para carregar o par.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    if not long_ticker or not short_ticker:
        st.error("Preencha os dois tickers antes de analisar.")
        st.stop()

    with st.spinner(f"Baixando dados de {long_ticker} e {short_ticker}…"):
        long_data  = fetch(long_ticker,  period, interval)
        short_data = fetch(short_ticker, period, interval)

    if long_data.empty or short_data.empty:
        st.error("Não foi possível baixar dados. Verifique os tickers.")
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
        st.error(f"Janelas maiores que o histórico disponível. Reduza as janelas ou aumente o período.")
        st.stop()
    z_now    = float(z_clean.iloc[-1])
    z_prev   = float(z_clean.iloc[-2]) if len(z_clean) > 1 else z_now
    dz_now   = z_now - z_prev
    zero_cross = (z_now == 0) or (z_prev != 0 and np.sign(z_now) != np.sign(z_prev))
    corr_now = float(corr_series.dropna().iloc[-1]) if not corr_series.dropna().empty else 0.0

    ratio_lbl = "Log-Ratio" if use_log else "Ratio"

    # ---------- sinal em destaque ----------
    sig, cls, sig_desc = classify_signal(z_now, dz_now, corr_now, cfg, zero_cross)
    st.markdown(
        f'<div class="signal-banner {cls}">'
        f'<span class="sb-label">⚡ {sig}</span>'
        f'<span class="sb-desc">{sig_desc}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Checklist do Alerta Máximo
    if sig.startswith("🔴"):
        st.markdown("**Checklist Alerta Máximo — validar antes de executar:**")
        chk1, chk2 = st.columns(2)
        with chk1:
            st.checkbox("Liquidez OK", key="chk_liq")
            st.checkbox("Spread OK", key="chk_spread")
        with chk2:
            st.checkbox(f"Cointegração plausível (p={coint_p:.3f})", value=coint_p < 0.05, key="chk_coint")
            st.checkbox("Aluguel OK / checado", key="chk_alug")

    # ---------- métricas-chave em cards ----------
    z_tone    = "red" if z_now >= z_monitor else "green" if z_now <= -z_monitor else "neutral"
    corr_ok   = corr_now >= corr_min
    coint_ok  = coint_p < 0.05
    adf_ok    = adf_p < 0.05

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(metric_card(ratio_lbl, f"{ratio_series.iloc[-1]:.4f}",
                            f"média {mean_win}p: {ratio_series.rolling(mean_win).mean().iloc[-1]:.4f}",
                            tone="purple"), unsafe_allow_html=True)
    c2.markdown(metric_card("Z-Score", f"{z_now:.3f}",
                            "vender ratio" if z_now >= z_monitor else "comprar ratio" if z_now <= -z_monitor else "zona neutra",
                            tone=z_tone), unsafe_allow_html=True)
    c3.markdown(metric_card("Δz · 1 pregão", f"{dz_now:+.3f}",
                            "momentum do z-score", tone="neutral"), unsafe_allow_html=True)
    c4.markdown(metric_card(f"Correlação {corr_win}p", f"{corr_now:.3f}",
                            f"{'✅ OK' if corr_ok else '❌ Baixa'} · gatilho ≥ {corr_min:.2f}",
                            tone="green" if corr_ok else "red"), unsafe_allow_html=True)
    c5.markdown(metric_card("Cointegração p", f"{coint_p:.4f}",
                            "✅ cointegrado" if coint_ok else "❌ fraco (p ≥ 0.05)",
                            tone="green" if coint_ok else "red"), unsafe_allow_html=True)
    c6.markdown(metric_card("ADF spread p", f"{adf_p:.4f}",
                            "✅ estacionário" if adf_ok else "❌ fraco (p ≥ 0.05)",
                            tone="green" if adf_ok else "red"), unsafe_allow_html=True)

    st.markdown("")

    # ---------- conteúdo em abas ----------
    tab_charts, tab_stats, tab_ai, tab_manual = st.tabs(
        ["📊 Gráficos", "📋 Estatísticas", "🤖 Análise IA", "ℹ️ Manual"]
    )

    with tab_charts:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.30, 0.25, 0.25, 0.20],
            subplot_titles=["Preços normalizados (base 100)",
                            f"{ratio_lbl} {long_ticker}/{short_ticker} · média {mean_win}p · bandas {vol_win}p",
                            f"Z-Score ({mean_win}p/{vol_win}p)", f"Correlação rolling {corr_win}p"],
            vertical_spacing=0.06)

        long_norm  = long_data  / long_data.iloc[0]  * 100
        short_norm = short_data / short_data.iloc[0] * 100
        fig.add_trace(go.Scatter(x=long_norm.index,  y=long_norm,  name=f"▲ {long_ticker}",  line=dict(color=C_LONG,  width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=short_norm.index, y=short_norm, name=f"▼ {short_ticker}", line=dict(color=C_SHORT, width=2)), row=1, col=1)

        roll_mean = ratio_series.rolling(mean_win).mean()
        roll_std  = ratio_series.rolling(vol_win).std()
        fig.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, name=ratio_lbl, line=dict(color=C_ACCENT, width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, name="Média", line=dict(color="#585b70", width=1, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean + z_alert * roll_std,   name=f"+{z_alert:.2f}σ (alerta)",   line=dict(color=C_SHORT,   width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean + z_monitor * roll_std, name=f"±{z_monitor:.2f}σ (monitor)", line=dict(color=C_MONITOR, width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean - z_monitor * roll_std, showlegend=False, name=f"-{z_monitor:.2f}σ (monitor)", line=dict(color=C_MONITOR, width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean - z_alert * roll_std,   name=f"-{z_alert:.2f}σ (alerta)",   line=dict(color=C_LONG,    width=1, dash="dot")), row=2, col=1)

        # verde = zona LONG ratio (z negativo) · vermelho = zona SHORT ratio (z positivo)
        colors_z = [C_SHORT if v >= z_alert else C_MONITOR if v >= z_monitor else
                    C_LONG if v <= -z_alert else C_TEAL if v <= -z_monitor else C_NEUTRAL
                    for v in z_series.fillna(0)]
        fig.add_trace(go.Bar(x=z_series.index, y=z_series, name="Z-Score", marker_color=colors_z, opacity=0.9), row=3, col=1)
        for level, dash in [(z_alert, "solid"), (-z_alert, "solid"), (z_monitor, "dot"), (-z_monitor, "dot"), (z_exit, "dash"), (-z_exit, "dash")]:
            fig.add_hline(y=level, line_dash=dash, line_color="#585b70", opacity=0.55, row=3, col=1)
        fig.add_hline(y=0, line_color=C_TEXT, opacity=0.3, row=3, col=1)

        fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series, name="Correlação",
            line=dict(color=C_YELLOW, width=2), fill="tozeroy", fillcolor="rgba(249,226,175,0.08)"), row=4, col=1)
        fig.add_hline(y=corr_min, line_dash="dot", line_color=C_LONG, opacity=0.6, row=4, col=1)

        style_fig(fig, height=800, top_margin=95, legend_y=1.05)
        fig.update_annotations(font=dict(size=13, color=C_MUTED))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with tab_stats:
        st.markdown(f'<span class="fund-header">📋 Estatísticas do Spread</span>', unsafe_allow_html=True)
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

    with tab_ai:
        st.markdown('<span class="fund-header">🤖 Análise Claude AI</span>', unsafe_allow_html=True)
        if not api_key:
            st.warning("Insira sua Anthropic API Key na barra lateral (expander **🔑 API**) para ativar a análise via Claude.")
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
