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

st.set_page_config(page_title="Long/Short Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .signal-long  { color: #a6e3a1; font-weight: bold; font-size: 1.4rem; }
    .signal-short { color: #f38ba8; font-weight: bold; font-size: 1.4rem; }
    .signal-neutral { color: #cdd6f4; font-weight: bold; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Configurações")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    st.divider()
    st.subheader("Ativos")
    PRESETS = {
        "BTC / ETH": ("BTC-USD", "ETH-USD"),
        "PETR4 / VALE3": ("PETR4.SA", "VALE3.SA"),
        "ITUB4 / BBDC4": ("ITUB4.SA", "BBDC4.SA"),
        "MGLU3 / VIIA3": ("MGLU3.SA", "VIIA3.SA"),
        "SPY / QQQ": ("SPY", "QQQ"),
        "GLD / SLV": ("GLD", "SLV"),
        "PETR4 / BRKM5": ("PETR4.SA", "BRKM5.SA"),
        "Custom": ("", ""),
    }
    preset = st.selectbox("Par pré-definido", list(PRESETS.keys()))
    default_long, default_short = PRESETS[preset]
    col1, col2 = st.columns(2)
    with col1:
        long_ticker = st.text_input("LONG (comprado)", value=default_long).upper().strip()
    with col2:
        short_ticker = st.text_input("SHORT (vendido)", value=default_short).upper().strip()
    st.divider()
    st.subheader("Período")
    period_map = {"1 mês": "1mo", "3 meses": "3mo", "6 meses": "6mo", "1 ano": "1y", "2 anos": "2y", "5 anos": "5y"}
    period_label = st.selectbox("Histórico", list(period_map.keys()), index=2)
    period = period_map[period_label]
    interval_map = {"Diário": "1d", "Semanal": "1wk", "Horário": "1h"}
    interval_label = st.selectbox("Intervalo", list(interval_map.keys()))
    interval = interval_map[interval_label]
    st.divider()
    z_window = st.slider("Janela Z-Score (dias)", 10, 120, 30)
    z_entry  = st.slider("Entrada (|Z| >)", 0.5, 3.0, 2.0, 0.1)
    z_exit   = st.slider("Saída  (|Z| <)", 0.0, 1.5, 0.5, 0.1)
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    run_btn = st.button("▶ Analisar", use_container_width=True, type="primary")

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
    if z > entry:   return "SHORT ratio → comprar SHORT / vender LONG"
    if z < -entry:  return "LONG ratio → comprar LONG / vender SHORT"
    if abs(z) < exit_: return "FECHAR posição (convergência)"
    return "NEUTRO / aguardar"

def signal_class(z, entry):
    if z > entry:  return "signal-short"
    if z < -entry: return "signal-long"
    return "signal-neutral"

def analyze_with_claude(api_key, long_t, short_t, ratio, z, coint_p, adf_p, hedge, corr_last):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Você é um analista quantitativo especialista em estratégias Long/Short.
Analise o par {long_t} (LONG) x {short_t} (SHORT) com os dados abaixo e responda em português.

- Ratio atual: {ratio.iloc[-1]:.4f} | Média: {ratio.rolling(30).mean().iloc[-1]:.4f}
- Z-Score atual: {z.iloc[-1]:.3f} | Máx: {z.max():.3f} | Mín: {z.min():.3f}
- Cointegração p-valor: {coint_p:.4f} → {"✅ cointegrado" if coint_p < 0.05 else "❌ não cointegrado"}
- ADF p-valor: {adf_p:.4f} → {"✅ estacionário" if adf_p < 0.05 else "❌ não estacionário"}
- Hedge ratio (β): {hedge:.4f}
- Correlação rolling atual: {corr_last:.3f}

Estruture sua resposta em:
1. Interpretação do Z-Score
2. Qualidade estatística do par
3. Sinal operacional (entrada/saída/aguardar)
4. Riscos e alertas
5. Resumo executivo (2-3 linhas)"""
    with client.messages.stream(model="claude-sonnet-4-6", max_tokens=1200,
                                 messages=[{"role": "user", "content": prompt}]) as stream:
        return stream.get_final_text()

# ── Main ──
st.title("📊 Long / Short Dashboard")
st.caption("Ratio em tempo real · Z-Score · Cointegração · Análise via Claude AI")

if auto_refresh:
    time.sleep(30)
    st.rerun()

if not (run_btn or auto_refresh):
    st.info("Configure os ativos na barra lateral e clique em **▶ Analisar**.")
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

ratio_series = compute_ratio(long_data, short_data)
z_series     = zscore(ratio_series, z_window)
corr_series  = rolling_corr(long_data, short_data, z_window)
hedge        = hedge_ratio(long_data, short_data)

try:    coint_p = cointegration_test(long_data, short_data)
except: coint_p = 1.0
try:
    spread = long_data - hedge * short_data
    adf_p  = adf_test(spread)
except: adf_p = 1.0

z_now    = float(z_series.dropna().iloc[-1])
corr_now = float(corr_series.dropna().iloc[-1]) if not corr_series.dropna().empty else 0.0

st.subheader(f"{long_ticker}  ×  {short_ticker}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ratio",          f"{ratio_series.iloc[-1]:.4f}")
c2.metric("Z-Score",        f"{z_now:.3f}")
c3.metric("Correlação",     f"{corr_now:.3f}")
c4.metric("Cointegração p", f"{coint_p:.4f}", delta="✅ OK" if coint_p < 0.05 else "❌ Fraco")
c5.metric("ADF spread p",   f"{adf_p:.4f}",   delta="✅ OK" if adf_p < 0.05 else "❌ Fraco")

sig = signal_label(z_now, z_entry, z_exit)
cls = signal_class(z_now, z_entry)
st.markdown(f'<div class="{cls}">⚡ Sinal: {sig}</div>', unsafe_allow_html=True)
st.divider()

fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.30, 0.25, 0.25, 0.20],
    subplot_titles=["Preços normalizados", f"Ratio {long_ticker}/{short_ticker}",
                    f"Z-Score (janela {z_window}d)", "Correlação rolling"],
    vertical_spacing=0.06)

long_norm  = long_data  / long_data.iloc[0]  * 100
short_norm = short_data / short_data.iloc[0] * 100
fig.add_trace(go.Scatter(x=long_norm.index,  y=long_norm,  name=long_ticker,  line=dict(color="#a6e3a1", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=short_norm.index, y=short_norm, name=short_ticker, line=dict(color="#f38ba8", width=1.5)), row=1, col=1)

roll_mean  = ratio_series.rolling(z_window).mean()
roll_upper = roll_mean + z_entry * ratio_series.rolling(z_window).std()
roll_lower = roll_mean - z_entry * ratio_series.rolling(z_window).std()
fig.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, name="Ratio",    line=dict(color="#cba6f7", width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=roll_mean.index,    y=roll_mean,    name="Média",    line=dict(color="gray",    width=1, dash="dash")), row=2, col=1)
fig.add_trace(go.Scatter(x=roll_upper.index,   y=roll_upper,   name=f"+{z_entry}σ", line=dict(color="#fab387", width=1, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=roll_lower.index,   y=roll_lower,   name=f"-{z_entry}σ", line=dict(color="#89dceb", width=1, dash="dot")), row=2, col=1)

colors_z = ["#f38ba8" if v > z_entry else "#a6e3a1" if v < -z_entry else "#cdd6f4" for v in z_series.fillna(0)]
fig.add_trace(go.Bar(x=z_series.index, y=z_series, name="Z-Score", marker_color=colors_z, opacity=0.8), row=3, col=1)
for level in [z_entry, -z_entry, z_exit, -z_exit]:
    fig.add_hline(y=level, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
fig.add_hline(y=0, line_color="white", opacity=0.3, row=3, col=1)

fig.add_trace(go.Scatter(x=corr_series.index, y=corr_series, name="Correlação",
    line=dict(color="#f9e2af", width=1.5), fill="tozeroy", fillcolor="rgba(249,226,175,0.1)"), row=4, col=1)

fig.update_layout(height=750, paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
    font=dict(color="#cdd6f4"), legend=dict(bgcolor="#313244"),
    margin=dict(l=0, r=0, t=40, b=0))
fig.update_xaxes(gridcolor="#313244")
fig.update_yaxes(gridcolor="#313244")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Estatísticas do Spread")
df_stats = pd.DataFrame({
    "Métrica": ["Ratio atual","Ratio médio","Ratio máx","Ratio mín","Z-Score atual","Z-Score máx","Z-Score mín","Hedge ratio (β)","Correlação atual","Coint. p-valor","ADF p-valor"],
    "Valor":   [f"{ratio_series.iloc[-1]:.4f}", f"{ratio_series.rolling(z_window).mean().iloc[-1]:.4f}",
                f"{ratio_series.max():.4f}", f"{ratio_series.min():.4f}", f"{z_now:.3f}",
                f"{z_series.max():.3f}", f"{z_series.min():.3f}", f"{hedge:.4f}", f"{corr_now:.3f}",
                f"{coint_p:.4f}", f"{adf_p:.4f}"],
    "Status":  ["—","—","—","—",
                "🔴 Vender ratio" if z_now > z_entry else "🟢 Comprar ratio" if z_now < -z_entry else "⚪ Neutro",
                "—","—","—",
                "✅ Alta" if corr_now > 0.7 else "⚠️ Média" if corr_now > 0.4 else "❌ Baixa",
                "✅ Cointegrado" if coint_p < 0.05 else "❌ Não cointegrado",
                "✅ Estacionário" if adf_p < 0.05 else "❌ Não estacionário"],
})
st.dataframe(df_stats, use_container_width=True, hide_index=True)

st.divider()
st.subheader("🤖 Análise Claude AI")
if not api_key:
    st.warning("Insira sua Anthropic API Key na barra lateral para ativar a análise via Claude.")
else:
    if st.button("🧠 Gerar análise com Claude", type="secondary"):
        with st.spinner("Claude analisando o par…"):
            try:
                st.markdown(analyze_with_claude(api_key, long_ticker, short_ticker,
                                                ratio_series, z_series, coint_p, adf_p, hedge, corr_now))
            except anthropic.AuthenticationError:
                st.error("API Key inválida.")
            except Exception as e:
                st.error(f"Erro: {e}")

st.divider()
st.caption(f"Dados via Yahoo Finance · Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")