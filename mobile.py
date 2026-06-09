import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint, adfuller
import anthropic
from datetime import datetime

# ─── Config mobile-first ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="LS Mobile",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Touch-friendly + compacto pra celular */
    .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 640px; }
    .stButton button { height: 3rem; font-size: 1.05rem; border-radius: 12px; }
    .signal-box {
        padding: 0.9rem 1rem; border-radius: 14px; font-weight: 700;
        font-size: 1.15rem; text-align: center; margin: 0.4rem 0 0.8rem 0;
    }
    .sig-long    { background:#1e3a2a; color:#a6e3a1; }
    .sig-short   { background:#3a1e26; color:#f38ba8; }
    .sig-neutral { background:#2a2a3a; color:#cdd6f4; }
    div[data-testid="stMetric"] { background:#1e1e2e; padding:0.6rem; border-radius:12px; }
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

default_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")


# ─── Lógica (cópias puras, independente do app desktop) ───────────────────────
@st.cache_data(ttl=60)
def fetch(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float, name=ticker)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return close.dropna()


def compute_ratio(long, short):
    df = pd.concat([long, short], axis=1).dropna()
    return df.iloc[:, 0] / df.iloc[:, 1]


def zscore(series, window):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
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


def signal_label(z, entry, exit_):
    if z > entry:       return "SHORT ratio", "sig-short", "Comprar SHORT · vender LONG"
    if z < -entry:      return "LONG ratio", "sig-long", "Comprar LONG · vender SHORT"
    if abs(z) < exit_:  return "FECHAR", "sig-neutral", "Convergência — encerrar posição"
    return "NEUTRO", "sig-neutral", "Aguardar"


def analyze_with_claude(api_key, long_t, short_t, ratio, z, coint_p, adf_p, hedge, corr_last, window):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Você é um analista quantitativo especialista em estratégias Long/Short.
Analise o par {long_t} (LONG) x {short_t} (SHORT) e responda em português, de forma concisa (ideal para leitura no celular).

- Ratio atual: {ratio.iloc[-1]:.4f} | Média ({window}d): {ratio.rolling(window).mean().iloc[-1]:.4f}
- Z-Score atual: {z.iloc[-1]:.3f} | Máx: {z.max():.3f} | Mín: {z.min():.3f}
- Cointegração p-valor: {coint_p:.4f} → {"cointegrado" if coint_p < 0.05 else "não cointegrado"}
- ADF p-valor: {adf_p:.4f} → {"estacionário" if adf_p < 0.05 else "não estacionário"}
- Hedge ratio (β): {hedge:.4f}
- Correlação rolling: {corr_last:.3f}

Responda em 4 blocos curtos:
1. Sinal (entrar/sair/aguardar) — 1 linha
2. Qualidade estatística — 1-2 linhas
3. Riscos — 1-2 linhas
4. Resumo — 1 linha"""
    with client.messages.stream(model="claude-sonnet-4-6", max_tokens=900,
                                 messages=[{"role": "user", "content": prompt}]) as stream:
        return stream.get_final_text()


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("📱 Long / Short")

with st.expander("⚙️ Configurações", expanded=not st.session_state.get("m_ran")):
    c1, c2 = st.columns(2)
    with c1:
        long_ticker = st.text_input("LONG", value="PETR4.SA").upper().strip()
    with c2:
        short_ticker = st.text_input("SHORT", value="VALE3.SA").upper().strip()

    period_map = {"3 meses": "3mo", "6 meses": "6mo", "1 ano": "1y", "2 anos": "2y"}
    period_label = st.selectbox("Histórico", list(period_map.keys()), index=1)
    period = period_map[period_label]

    z_window = st.slider("Janela Z-Score (dias)", 10, 90, 30)
    cc1, cc2 = st.columns(2)
    with cc1:
        z_entry = st.slider("Entrada |Z|>", 0.5, 3.0, 2.0, 0.1)
    with cc2:
        z_exit = st.slider("Saída |Z|<", 0.0, 1.5, 0.5, 0.1)

    api_key = st.text_input("Anthropic API Key", value=default_api_key,
                            type="password", placeholder="sk-ant-… (opcional)")

run = st.button("▶ Analisar", use_container_width=True, type="primary")

if run:
    st.session_state["m_ran"] = True
    st.session_state["m_long"] = long_ticker
    st.session_state["m_short"] = short_ticker

already_ran = (
    st.session_state.get("m_ran")
    and st.session_state.get("m_long") == long_ticker
    and st.session_state.get("m_short") == short_ticker
)

if not (run or already_ran):
    st.info("Preencha os ativos acima e toque em **▶ Analisar**.")
    st.stop()

if not long_ticker or not short_ticker:
    st.error("Preencha os dois tickers.")
    st.stop()

with st.spinner(f"Baixando {long_ticker} e {short_ticker}…"):
    long_data = fetch(long_ticker, period, "1d")
    short_data = fetch(short_ticker, period, "1d")

if long_data.empty or short_data.empty:
    st.error("Não foi possível baixar dados. Verifique os tickers.")
    st.stop()

ratio_series = compute_ratio(long_data, short_data)
z_series = zscore(ratio_series, z_window)
corr_series = rolling_corr(long_data, short_data, z_window)
hedge = hedge_ratio(long_data, short_data)

try:    coint_p = cointegration_test(long_data, short_data)
except: coint_p = 1.0
try:
    adf_p = adf_test(long_data - hedge * short_data)
except: adf_p = 1.0

if z_series.dropna().empty:
    st.error(f"Janela ({z_window}d) maior que o histórico. Reduza a janela ou aumente o período.")
    st.stop()

z_now = float(z_series.dropna().iloc[-1])
corr_now = float(corr_series.dropna().iloc[-1]) if not corr_series.dropna().empty else 0.0

# ─── Sinal (destaque no topo, mobile) ─────────────────────────────────────────
label, cls, hint = signal_label(z_now, z_entry, z_exit)
st.markdown(f'<div class="signal-box {cls}">⚡ {label}<br><span style="font-size:0.85rem;font-weight:400">{hint}</span></div>',
            unsafe_allow_html=True)

# Métricas em 2 colunas (cabe no celular)
m1, m2 = st.columns(2)
m1.metric("Ratio", f"{ratio_series.iloc[-1]:.4f}")
m2.metric("Z-Score", f"{z_now:.3f}")
m3, m4 = st.columns(2)
m3.metric("Correlação", f"{corr_now:.3f}", delta="OK" if corr_now > 0.7 else "Baixa")
m4.metric("Hedge β", f"{hedge:.3f}")
m5, m6 = st.columns(2)
m5.metric("Coint. p", f"{coint_p:.3f}", delta="OK" if coint_p < 0.05 else "Fraco")
m6.metric("ADF p", f"{adf_p:.3f}", delta="OK" if adf_p < 0.05 else "Fraco")

st.divider()

# ─── Gráficos empilhados (mobile) ─────────────────────────────────────────────
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.36, 0.34, 0.30],
                    subplot_titles=["Preços (base 100)", "Ratio", f"Z-Score ({z_window}d)"],
                    vertical_spacing=0.08)

long_norm = long_data / long_data.iloc[0] * 100
short_norm = short_data / short_data.iloc[0] * 100
fig.add_trace(go.Scatter(x=long_norm.index, y=long_norm, name=long_ticker,
                         line=dict(color="#a6e3a1", width=1.6)), row=1, col=1)
fig.add_trace(go.Scatter(x=short_norm.index, y=short_norm, name=short_ticker,
                         line=dict(color="#f38ba8", width=1.6)), row=1, col=1)

roll_mean = ratio_series.rolling(z_window).mean()
roll_std = ratio_series.rolling(z_window).std()
fig.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, name="Ratio",
                         line=dict(color="#cba6f7", width=1.6)), row=2, col=1)
fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, name="Média",
                         line=dict(color="gray", width=1, dash="dash")), row=2, col=1)
fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean + z_entry * roll_std, name="+σ",
                         line=dict(color="#fab387", width=1, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean - z_entry * roll_std, name="-σ",
                         line=dict(color="#89dceb", width=1, dash="dot")), row=2, col=1)

colors_z = ["#f38ba8" if v > z_entry else "#a6e3a1" if v < -z_entry else "#cdd6f4"
            for v in z_series.fillna(0)]
fig.add_trace(go.Bar(x=z_series.index, y=z_series, name="Z", marker_color=colors_z,
                     opacity=0.85), row=3, col=1)
for level in [z_entry, -z_entry]:
    fig.add_hline(y=level, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
fig.add_hline(y=0, line_color="white", opacity=0.3, row=3, col=1)

fig.update_layout(height=620, paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                  font=dict(color="#cdd6f4", size=11),
                  legend=dict(orientation="h", y=-0.08, bgcolor="rgba(0,0,0,0)"),
                  margin=dict(l=0, r=0, t=30, b=0))
fig.update_xaxes(gridcolor="#313244")
fig.update_yaxes(gridcolor="#313244")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─── Análise Claude ───────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 Análise Claude")
if not api_key:
    st.caption("Adicione sua Anthropic API Key nas Configurações para ativar.")
else:
    akey = f"m_analysis_{long_ticker}_{short_ticker}"
    if st.button("🧠 Gerar análise", use_container_width=True):
        with st.spinner("Claude analisando…"):
            try:
                st.session_state[akey] = analyze_with_claude(
                    api_key, long_ticker, short_ticker, ratio_series, z_series,
                    coint_p, adf_p, hedge, corr_now, z_window)
            except anthropic.AuthenticationError:
                st.error("API Key inválida.")
            except Exception as e:
                st.error(f"Erro: {e}")
    if akey in st.session_state:
        st.markdown(st.session_state[akey])

st.caption(f"Yahoo Finance · {datetime.now().strftime('%d/%m/%Y %H:%M')}")
