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

st.markdown("""
<style>
    .signal-long    { color: #a6e3a1; font-weight: bold; font-size: 1.4rem; }
    .signal-short   { color: #f38ba8; font-weight: bold; font-size: 1.4rem; }
    .signal-neutral { color: #cdd6f4; font-weight: bold; font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

default_api_key   = st.secrets.get("ANTHROPIC_API_KEY", "")
default_brapi_key = st.secrets.get("BRAPI_TOKEN", "")

with st.sidebar:
    st.title("⚙️ Configurações")
    mode = st.radio("Painel", ["📈 Long / Short", "🏦 Fundamentos"], label_visibility="collapsed")
    st.divider()

    if mode == "📈 Long / Short":
        api_key = st.text_input("Anthropic API Key", value=default_api_key, type="password", placeholder="sk-ant-...")
        st.divider()
        st.subheader("Ativos")
        PRESETS_LS = {
            "PETR4 / VALE3":  ("PETR4.SA", "VALE3.SA"),
            "ITUB4 / BBDC4":  ("ITUB4.SA", "BBDC4.SA"),
            "MGLU3 / VIIA3":  ("MGLU3.SA", "VIIA3.SA"),
            "BTC / ETH":      ("BTC-USD",  "ETH-USD"),
            "SPY / QQQ":      ("SPY",      "QQQ"),
            "GLD / SLV":      ("GLD",      "SLV"),
            "PETR4 / BRKM5":  ("PETR4.SA", "BRKM5.SA"),
            "Custom":         ("",         ""),
        }
        preset = st.selectbox("Par pré-definido", list(PRESETS_LS.keys()))
        default_long, default_short = PRESETS_LS[preset]
        col1, col2 = st.columns(2)
        with col1:
            long_ticker  = st.text_input("LONG",  value=default_long).upper().strip()
        with col2:
            short_ticker = st.text_input("SHORT", value=default_short).upper().strip()
        st.divider()
        st.subheader("Período")
        period_map   = {"1 mês": "1mo", "3 meses": "3mo", "6 meses": "6mo", "1 ano": "1y", "2 anos": "2y", "5 anos": "5y"}
        period_label = st.selectbox("Histórico", list(period_map.keys()), index=2)
        period       = period_map[period_label]
        interval_map   = {"Diário": "1d", "Semanal": "1wk", "Horário": "1h"}
        interval_label = st.selectbox("Intervalo", list(interval_map.keys()))
        interval       = interval_map[interval_label]
        st.divider()
        z_window     = st.slider("Janela Z-Score (dias)", 10, 120, 30)
        z_entry      = st.slider("Entrada (|Z| >)", 0.5, 3.0, 2.0, 0.1)
        z_exit       = st.slider("Saída  (|Z| <)", 0.0, 1.5, 0.5, 0.1)
        auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
        run_btn      = st.button("▶ Analisar", use_container_width=True, type="primary")

    else:
        brapi_key = st.text_input("brapi.dev Token", value=default_brapi_key, type="password", placeholder="token")
        st.divider()
        st.subheader("Empresas")
        PRESETS_FUND = {
            "Varejo BR":       "MGLU3\nAMER3\nLREN3\nGUAR3\nVIVA3",
            "Bancos BR":       "ITUB4\nBBDC4\nBBAS3\nSANB11\nBPAC11",
            "Petro & Energia": "PETR4\nPRIO3\nRECV3\nUGPA3\nVBBR3",
            "Real Estate":     "CYRE3\nMRVE3\nEZTC3\nDIRR3\nTRIS3",
            "Custom":          "",
        }
        fund_preset = st.selectbox("Grupo pré-definido", list(PRESETS_FUND.keys()))
        default_tickers = PRESETS_FUND[fund_preset]
        fund_tickers_raw = st.text_area("Tickers (um por linha, sem .SA)", value=default_tickers, height=160)
        load_btn = st.button("🔍 Carregar Fundamentos", use_container_width=True, type="primary")
        st.divider()
        st.caption("Dados: brapi.dev (B3)")

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
    if z > entry:       return "SHORT ratio → comprar SHORT / vender LONG"
    if z < -entry:      return "LONG ratio → comprar LONG / vender SHORT"
    if abs(z) < exit_:  return "FECHAR posição (convergência)"
    return "NEUTRO / aguardar"

def signal_class(z, entry):
    if z > entry:   return "signal-short"
    if z < -entry:  return "signal-long"
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
3. Sinal operacional
4. Riscos e alertas
5. Resumo executivo"""
    with client.messages.stream(model="claude-sonnet-4-6", max_tokens=1200,
                                 messages=[{"role": "user", "content": prompt}]) as stream:
        return stream.get_final_text()

@st.cache_data(ttl=300)
def fetch_brapi(ticker: str, token: str) -> dict:
    t = ticker.upper().replace(".SA", "").strip()
    url = f"https://brapi.dev/api/quote/{t}"
    params = {"modules": "defaultKeyStatistics,financialData,summaryProfile,price"}
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        params["token"] = token
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        data = r.json()
        if "results" in data and data["results"]:
            return data["results"][0]
        return {"_error": data.get("message") or data.get("error") or f"HTTP {r.status_code}: {str(r.text)[:200]}"}
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
    pr = info.get("price") or {}

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
        "Nome":           pr.get("longName") or info.get("longName", ticker),
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
        "Market Cap":     pr.get("marketCap") or info.get("marketCap"),
    }

if mode == "📈 Long / Short":
    st.title("📊 Long / Short Dashboard")
    st.caption("Ratio · Z-Score · Cointegração · Claude AI")

    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if not (run_btn or auto_refresh):
        st.info("Configure os ativos e clique em **▶ Analisar**.")
        st.stop()

    if not long_ticker or not short_ticker:
        st.error("Preencha os dois tickers.")
        st.stop()

    with st.spinner(f"Baixando dados…"):
        long_data  = fetch(long_ticker,  period, interval)
        short_data = fetch(short_ticker, period, interval)

    if long_data.empty or short_data.empty:
        st.error("Não foi possível baixar dados.")
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
    fig.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, name="Ratio", line=dict(color="#cba6f7", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=roll_mean.index,    y=roll_mean,    name="Média", line=dict(color="gray", width=1, dash="dash")), row=2, col=1)
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
        st.warning("Insira sua Anthropic API Key.")
    else:
        if st.button("🧠 Gerar análise com Claude", type="secondary"):
            with st.spinner("Claude analisando…"):
                try:
                    st.markdown(analyze_with_claude(api_key, long_ticker, short_ticker,
                                                    ratio_series, z_series, coint_p, adf_p, hedge, corr_now))
                except anthropic.AuthenticationError:
                    st.error("API Key inválida.")
                except Exception as e:
                    st.error(f"Erro: {e}")

    st.divider()
    with st.expander("ℹ️ Manual de Métricas — Long/Short"):
        st.markdown("""
**📐 Ratio** — Preço do LONG ÷ preço do SHORT.

**📊 Z-Score** — Distância do Ratio em relação à média histórica (desvios padrão). Z > +2 → SHORT ratio. Z < -2 → LONG ratio.

**🔗 Correlação** — Sincronia dos ativos (−1 a +1). Acima de 0.7 é ideal.

**🧪 Cointegração p-valor** — Engle-Granger. p < 0.05 = relação de longo prazo.

**📉 ADF spread p** — Estacionaridade. p < 0.05 = spread reverte à média.

**⚖️ Hedge Ratio (β)** — Quanto vender no SHORT por unidade de LONG comprado.

**⚡ Sinal** — LONG ratio, SHORT ratio, FECHAR ou NEUTRO.
        """)
    st.caption(f"Dados Yahoo Finance · {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

else:
    st.title("🏦 Fundamentos — Comparação de Empresas")
    st.caption("Dados via brapi.dev · Fonte: B3 / CVM")

    if not load_btn:
        st.info("Selecione um grupo e clique em **🔍 Carregar Fundamentos**.")
        st.stop()

    tickers = [t.strip().upper().replace(".SA", "") for t in fund_tickers_raw.splitlines() if t.strip()]
    if not tickers:
        st.error("Nenhum ticker informado.")
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
        st.error("Nenhum dado encontrado. Verifique o token brapi.dev (PRO usa Bearer).")
        st.stop()

    df = pd.DataFrame(rows)

    st.subheader("📋 Tabela Comparativa")
    DISPLAY_COLS = ["Ticker","Nome","Setor","P/L","P/L Fwd","P/VP","EV/EBITDA",
        "ROE (%)","ROA (%)","Marg. Liq. (%)","Marg. EBITDA (%)",
        "Div. Yield (%)","Dív./PL","Liq. Corrente",
        "Cresc. Receita (%)","Cresc. Lucro (%)","Beta","Market Cap"]
    df_display = df[DISPLAY_COLS].copy()
    pct_cols = ["ROE (%)","ROA (%)","Marg. Liq. (%)","Marg. EBITDA (%)","Div. Yield (%)","Cresc. Receita (%)","Cresc. Lucro (%)"]
    num_cols = ["P/L","P/L Fwd","P/VP","EV/EBITDA","Dív./PL","Liq. Corrente","Beta"]

    def fmt_cell(v, is_pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.2f}%" if is_pct else f"{v:.2f}"

    for c in pct_cols:
        df_display[c] = df_display[c].apply(lambda x: fmt_cell(x, True))
    for c in num_cols:
        df_display[c] = df_display[c].apply(lambda x: fmt_cell(x))
    df_display["Market Cap"] = df["Market Cap"].apply(fmt_market_cap)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📊 Comparação Visual")
    CHART_METRICS = ["P/L","P/VP","ROE (%)","Marg. Liq. (%)","Div. Yield (%)","EV/EBITDA","Dív./PL","Cresc. Receita (%)"]
    selected_metrics = st.multiselect("Métricas", CHART_METRICS,
        default=["ROE (%)","Marg. Liq. (%)","Div. Yield (%)","P/L"])

    if selected_metrics:
        COLORS = ["#cba6f7","#a6e3a1","#f38ba8","#fab387","#89dceb","#f9e2af"]
        chunks = [selected_metrics[i:i+2] for i in range(0, len(selected_metrics), 2)]
        for chunk in chunks:
            row_cols = st.columns(len(chunk))
            for ci, metric in enumerate(chunk):
                vals = df[metric].tolist()
                labels = df["Ticker"].tolist()
                y, x = [], []
                for lab, v in zip(labels, vals):
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        y.append(float(v)); x.append(lab)
                if not y: continue
                fig_bar = go.Figure(go.Bar(x=x, y=y,
                    marker_color=[COLORS[i % len(COLORS)] for i in range(len(x))],
                    text=[f"{v:.1f}" for v in y], textposition="outside"))
                fig_bar.update_layout(title=metric, height=320,
                    paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                    font=dict(color="#cdd6f4", size=11),
                    margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
                fig_bar.update_xaxes(gridcolor="#313244")
                fig_bar.update_yaxes(gridcolor="#313244")
                with row_cols[ci]:
                    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("🕸️ Radar — Qualidade (normalizado)")
    RADAR_METRICS = ["ROE (%)","ROA (%)","Marg. Liq. (%)","Marg. EBITDA (%)","Div. Yield (%)"]
    radar_df = df[["Ticker"] + RADAR_METRICS].copy()
    for c in RADAR_METRICS:
        mn, mx = radar_df[c].min(), radar_df[c].max()
        radar_df[c] = (radar_df[c] - mn) / (mx - mn) * 100 if mx != mn else 50.0

    fig_radar = go.Figure()
    RADAR_COLORS = ["#cba6f7","#a6e3a1","#f38ba8","#fab387","#89dceb","#f9e2af"]
    for idx, r in radar_df.iterrows():
        vals = [r[m] for m in RADAR_METRICS]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=RADAR_METRICS + [RADAR_METRICS[0]],
            fill="toself", name=r["Ticker"],
            line=dict(color=RADAR_COLORS[idx % len(RADAR_COLORS)]), opacity=0.6))
    fig_radar.update_layout(
        polar=dict(bgcolor="#313244",
            radialaxis=dict(visible=True, range=[0,100], color="#cdd6f4"),
            angularaxis=dict(color="#cdd6f4")),
        paper_bgcolor="#1e1e2e", font=dict(color="#cdd6f4"),
        height=450, margin=dict(l=40, r=40, t=20, b=20))
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.subheader("🔍 Visão Individual")
    sel = st.selectbox("Selecionar empresa", df["Ticker"].tolist())
    r = df[df["Ticker"] == sel].iloc[0]
    a, b, c = st.columns(3)
    with a:
        st.markdown(f"**{r['Nome']}**")
        st.caption(f"Setor: {r['Setor']}")
        st.metric("Market Cap", fmt_market_cap(r["Market Cap"]))
        st.metric("Beta", safe_num(r["Beta"]))
    with b:
        st.metric("P/L", safe_num(r["P/L"]))
        st.metric("P/L Fwd", safe_num(r["P/L Fwd"]))
        st.metric("P/VP", safe_num(r["P/VP"]))
        st.metric("EV/EBITDA", safe_num(r["EV/EBITDA"]))
    with c:
        st.metric("ROE", safe_pct(r["ROE (%)"]))
        st.metric("ROA", safe_pct(r["ROA (%)"]))
        st.metric("Marg. Líq.", safe_pct(r["Marg. Liq. (%)"]))
        st.metric("Div. Yield", safe_pct(r["Div. Yield (%)"]))
    d, e = st.columns(2)
    with d:
        st.metric("Dívida/PL", safe_num(r["Dív./PL"]))
        st.metric("Liq. Corrente", safe_num(r["Liq. Corrente"]))
    with e:
        st.metric("Cresc. Receita", safe_pct(r["Cresc. Receita (%)"]))
        st.metric("Cresc. Lucro", safe_pct(r["Cresc. Lucro (%)"]))

    st.divider()
    with st.expander("ℹ️ Manual de Métricas — Fundamentos"):
        st.markdown("""
**P/L** — Anos de lucro embutidos no preço. Menor = mais barato.
**P/L Fwd** — P/L com lucro futuro estimado.
**P/VP** — Preço ÷ patrimônio. <1 = desconto.
**EV/EBITDA** — Valor da empresa ÷ EBITDA. Menor = mais barato.
**ROE (%)** — Retorno sobre patrimônio. >15% é bom.
**ROA (%)** — Retorno sobre ativos.
**Margem Líquida (%)** — % da receita que vira lucro.
**Margem EBITDA (%)** — % da receita que vira lucro operacional.
**Div. Yield (%)** — Dividendos ÷ preço.
**Dívida/PL** — Alavancagem.
**Liq. Corrente** — Saúde financeira de curto prazo.
**Cresc. Receita / Lucro (%)** — Crescimento período sobre período.
**Beta** — Sensibilidade ao mercado.
        """)
    st.caption(f"Dados brapi.dev · {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
