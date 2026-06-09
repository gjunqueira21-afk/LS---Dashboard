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
    .fund-header    { color: #cba6f7; font-size: 1.1rem; font-weight: bold; }
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
        col1, col2 = st.columns(2)
        with col1:
            long_ticker  = st.text_input("LONG",  value="PETR4.SA").upper().strip()
        with col2:
            short_ticker = st.text_input("SHORT", value="VALE3.SA").upper().strip()
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
        brapi_key = st.text_input("brapi.dev Token", value=default_brapi_key, type="password", placeholder="token (opcional)")
        st.divider()
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
        st.divider()
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

def analyze_with_claude(api_key, long_t, short_t, ratio, z, coint_p, adf_p, hedge, corr_last, window=30):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Você é um analista quantitativo especialista em estratégias Long/Short.
Analise o par {long_t} (LONG) x {short_t} (SHORT) com os dados abaixo e responda em português.

- Ratio atual: {ratio.iloc[-1]:.4f} | Média ({window}d): {ratio.rolling(window).mean().iloc[-1]:.4f}
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

if mode == "📈 Long / Short":
    st.title("📊 Long / Short Dashboard")
    st.caption("Ratio em tempo real · Z-Score · Cointegração · Análise via Claude AI")

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

    if z_series.dropna().empty:
        st.error(f"Janela Z-Score ({z_window}d) maior que o histórico disponível. Reduza a janela ou aumente o período.")
        st.stop()
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
    fig.add_trace(go.Scatter(x=roll_mean.index,    y=roll_mean,    name="Média",    line=dict(color="gray",   width=1, dash="dash")), row=2, col=1)
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
        analysis_key = f"claude_analysis_{long_ticker}_{short_ticker}"
        if st.button("🧠 Gerar análise com Claude", type="secondary"):
            with st.spinner("Claude analisando o par…"):
                try:
                    result = analyze_with_claude(api_key, long_ticker, short_ticker,
                                                 ratio_series, z_series, coint_p, adf_p, hedge, corr_now, z_window)
                    st.session_state[analysis_key] = result
                except anthropic.AuthenticationError:
                    st.error("API Key inválida.")
                except Exception as e:
                    st.error(f"Erro: {e}")
        if analysis_key in st.session_state:
            st.markdown(st.session_state[analysis_key])

    st.divider()
    with st.expander("ℹ️ Manual de Métricas — Long/Short"):
        st.markdown("""
**📐 Ratio** — Preço do LONG ÷ preço do SHORT. Sobe quando o LONG se valoriza mais que o SHORT.

**📊 Z-Score** — Distância do Ratio em relação à sua média histórica (em desvios padrão). Coração da estratégia:
- Z > +2 → par esticado → sinal SHORT no ratio
- Z < -2 → par comprimido → sinal LONG no ratio
- Z ≈ 0 → par na média → neutro

**🔗 Correlação** — Grau de sincronia entre os dois ativos (−1 a +1). Acima de 0.7 é ideal para Long/Short.

**🧪 Cointegração p-valor** — Teste Engle-Granger. p < 0.05 confirma que os ativos têm relação de longo prazo estável e tendem a convergir.

**📉 ADF spread p** — Teste de estacionaridade do spread. p < 0.05 significa que o spread oscila em torno de uma média fixa (necessário para a estratégia funcionar).

**⚖️ Hedge Ratio (β)** — Quanto do SHORT vender por unidade de LONG comprado para neutralizar o risco de mercado. Ex: β = 1.3 → vender R$1.300 no SHORT para cada R$1.000 no LONG.

**⚡ Sinal** — Conclusão prática: LONG ratio (comprar LONG + vender SHORT), SHORT ratio (inverso), FECHAR (Z voltou ao centro) ou NEUTRO.
        """)

    st.caption(f"Dados via Yahoo Finance · Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


else:
    st.title("🏦 Fundamentos — Comparação de Empresas")
    st.caption("Dados via brapi.dev · Fonte: B3 / CVM")

    if not load_btn:
        st.info("Selecione um grupo ou digite os tickers na barra lateral e clique em **🔍 Carregar Fundamentos**.")
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
        st.error("Nenhum dado encontrado. Verifique os tickers e o token da brapi.dev.")
        st.stop()

    df = pd.DataFrame(rows)

    st.subheader("📋 Tabela Comparativa")

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

    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.divider()

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
        COLORS = ["#cba6f7", "#a6e3a1", "#f38ba8", "#fab387", "#89dceb", "#f9e2af"]

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
                    paper_bgcolor="#1e1e2e",
                    plot_bgcolor="#1e1e2e",
                    font=dict(color="#cdd6f4", size=11),
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False,
                )
                fig_bar.update_xaxes(gridcolor="#313244")
                fig_bar.update_yaxes(gridcolor="#313244")
                with row_cols[ci]:
                    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

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
    RADAR_COLORS = ["#cba6f7", "#a6e3a1", "#f38ba8", "#fab387", "#89dceb", "#f9e2af"]
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
            bgcolor="#313244",
            radialaxis=dict(visible=True, range=[0, 100], color="#cdd6f4", tickfont=dict(size=10)),
            angularaxis=dict(color="#cdd6f4", tickfont=dict(size=11)),
        ),
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#313244", font=dict(size=12)),
        height=500,
        margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    st.subheader("🔍 Visão Individual")
    selected_ticker = st.selectbox("Selecionar empresa", df["Ticker"].tolist())
    row_ind = df[df["Ticker"] == selected_ticker].iloc[0]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"**{row_ind['Nome']}**")
        st.caption(f"Setor: {row_ind['Setor']}")
        st.metric("Market Cap", fmt_market_cap(row_ind["Market Cap"]))
        st.metric("Beta", safe_num(row_ind["Beta"]))
    with col_b:
        st.metric("P/L",      safe_num(row_ind["P/L"]))
        st.metric("P/L Fwd",  safe_num(row_ind["P/L Fwd"]))
        st.metric("P/VP",     safe_num(row_ind["P/VP"]))
        st.metric("EV/EBITDA",safe_num(row_ind["EV/EBITDA"]))
    with col_c:
        st.metric("ROE",         safe_pct(row_ind["ROE (%)"]))
        st.metric("ROA",         safe_pct(row_ind["ROA (%)"]))
        st.metric("Marg. Líq.",  safe_pct(row_ind["Marg. Liq. (%)"]))
        st.metric("Div. Yield",  safe_pct(row_ind["Div. Yield (%)"]))

    col_d, col_e = st.columns(2)
    with col_d:
        st.metric("Dívida/PL",       safe_num(row_ind["Dív./PL"]))
        st.metric("Liq. Corrente",   safe_num(row_ind["Liq. Corrente"]))
    with col_e:
        st.metric("Cresc. Receita",  safe_pct(row_ind["Cresc. Receita (%)"]))
        st.metric("Cresc. Lucro",    safe_pct(row_ind["Cresc. Lucro (%)"]))

    st.divider()
    with st.expander("ℹ️ Manual de Métricas — Fundamentos"):
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
