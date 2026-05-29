import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint, adfuller
import anthropic
import time
from datetime import datetime

st.set_page_config(page_title="Long/Short Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .signal-long    { color: #a6e3a1; font-weight: bold; font-size: 1.4rem; }
    .signal-short   { color: #f38ba8; font-weight: bold; font-size: 1.4rem; }
    .signal-neutral { color: #cdd6f4; font-weight: bold; font-size: 1.4rem; }
    .metric-good    { color: #a6e3a1; }
    .metric-warn    { color: #f9e2af; }
    .metric-bad     { color: #f38ba8; }
</style>
""", unsafe_allow_html=True)

default_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configurações")
    api_key = st.text_input("Anthropic API Key", value=default_api_key, type="password", placeholder="sk-ant-...")
    st.divider()

    # Long/Short config
    st.subheader("📈 Long / Short")
    PRESETS = {
        "BTC / ETH":        ("BTC-USD",   "ETH-USD"),
        "PETR4 / VALE3":    ("PETR4.SA",  "VALE3.SA"),
        "ITUB4 / BBDC4":    ("ITUB4.SA",  "BBDC4.SA"),
        "MGLU3 / VIIA3":    ("MGLU3.SA",  "VIIA3.SA"),
        "SPY / QQQ":        ("SPY",       "QQQ"),
        "GLD / SLV":        ("GLD",       "SLV"),
        "PETR4 / BRKM5":    ("PETR4.SA",  "BRKM5.SA"),
        "Custom":           ("",          ""),
    }
    preset = st.selectbox("Par pré-definido", list(PRESETS.keys()))
    default_long, default_short = PRESETS[preset]
    col1, col2 = st.columns(2)
    with col1:
        long_ticker = st.text_input("LONG (comprado)", value=default_long).upper().strip()
    with col2:
        short_ticker = st.text_input("SHORT (vendido)", value=default_short).upper().strip()

    period_map = {"1 mês": "1mo", "3 meses": "3mo", "6 meses": "6mo", "1 ano": "1y", "2 anos": "2y", "5 anos": "5y"}
    period_label = st.selectbox("Histórico", list(period_map.keys()), index=2)
    period = period_map[period_label]
    interval_map = {"Diário": "1d", "Semanal": "1wk", "Horário": "1h"}
    interval_label = st.selectbox("Intervalo", list(interval_map.keys()))
    interval = interval_map[interval_label]
    z_window = st.slider("Janela Z-Score (dias)", 10, 120, 30)
    z_entry  = st.slider("Entrada (|Z| >)", 0.5, 3.0, 2.0, 0.1)
    z_exit   = st.slider("Saída  (|Z| <)", 0.0, 1.5, 0.5, 0.1)
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    run_btn = st.button("▶ Analisar", use_container_width=True, type="primary")

    st.divider()

    # Fundamentos config
    st.subheader("🏦 Fundamentos")
    FUND_PRESETS = {
        "Bancos BR":      "ITUB4.SA, BBDC4.SA, SANB11.SA, BBAS3.SA",
        "Petro/Energia":  "PETR4.SA, PRIO3.SA, CSAN3.SA, UGPA3.SA",
        "Varejo BR":      "MGLU3.SA, AMER3.SA, LREN3.SA, SBFG3.SA",
        "Big Techs US":   "AAPL, MSFT, GOOGL, META, AMZN",
        "ETFs":           "SPY, QQQ, IWM, EEM",
        "Custom":         "",
    }
    fund_preset = st.selectbox("Grupo pré-definido", list(FUND_PRESETS.keys()), key="fund_preset")
    default_tickers_str = FUND_PRESETS[fund_preset]
    fund_tickers_input = st.text_area(
        "Tickers (separados por vírgula)",
        value=default_tickers_str,
        height=80,
        key="fund_tickers",
        placeholder="PETR4.SA, VALE3.SA, ITUB4.SA",
    )
    fund_btn = st.button("🔍 Carregar Fundamentos", use_container_width=True, type="primary")


# ── Helpers: Long/Short ───────────────────────────────────────────────────────
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
    if z > entry:        return "SHORT ratio → comprar SHORT / vender LONG"
    if z < -entry:       return "LONG ratio → comprar LONG / vender SHORT"
    if abs(z) < exit_:   return "FECHAR posição (convergência)"
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


# ── Helpers: Fundamentos ──────────────────────────────────────────────────────
FUND_FIELDS = {
    "Nome":               "shortName",
    "Setor":              "sector",
    "P/L":                "trailingPE",
    "P/L Fwd":            "forwardPE",
    "P/VP":               "priceToBook",
    "EV/EBITDA":          "enterpriseToEbitda",
    "ROE (%)":            "returnOnEquity",
    "ROA (%)":            "returnOnAssets",
    "Marg. Líq. (%)":     "profitMargins",
    "Marg. EBITDA (%)":   "ebitdaMargins",
    "Div. Yield (%)":     "dividendYield",
    "Dív./PL":            "debtToEquity",
    "Liq. Corrente":      "currentRatio",
    "Cresc. Receita (%)": "revenueGrowth",
    "Cresc. Lucro (%)":   "earningsGrowth",
    "Beta":               "beta",
    "Market Cap":         "marketCap",
}

PERCENT_FIELDS = {"ROE (%)", "ROA (%)", "Marg. Líq. (%)", "Marg. EBITDA (%)", "Div. Yield (%)", "Cresc. Receita (%)", "Cresc. Lucro (%)"}
NUMERIC_CHART_FIELDS = ["P/L", "P/L Fwd", "P/VP", "EV/EBITDA", "ROE (%)", "ROA (%)", "Marg. Líq. (%)", "Dív./PL", "Beta"]

@st.cache_data(ttl=300)
def fetch_fundamentals(tickers: tuple) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}
        row = {"Ticker": t}
        for label, key in FUND_FIELDS.items():
            val = info.get(key)
            if val is not None and label in PERCENT_FIELDS:
                val = round(float(val) * 100, 2)
            elif val is not None and label == "Market Cap":
                val = int(val)
            elif val is not None:
                try:
                    val = round(float(val), 2)
                except (TypeError, ValueError):
                    pass
            row[label] = val
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")

def fmt_market_cap(v):
    if pd.isna(v) or v is None:
        return "—"
    v = float(v)
    if v >= 1e12:  return f"${v/1e12:.2f}T"
    if v >= 1e9:   return f"${v/1e9:.2f}B"
    if v >= 1e6:   return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def color_value(val, field):
    if pd.isna(val) or val is None:
        return "—"
    v = float(val)
    if field == "P/L":
        color = "metric-good" if 5 < v < 20 else "metric-warn" if v <= 5 or v <= 30 else "metric-bad"
    elif field == "ROE (%)":
        color = "metric-good" if v >= 15 else "metric-warn" if v >= 8 else "metric-bad"
    elif field == "Marg. Líq. (%)":
        color = "metric-good" if v >= 10 else "metric-warn" if v >= 5 else "metric-bad"
    elif field == "Dív./PL":
        color = "metric-good" if v < 50 else "metric-warn" if v < 150 else "metric-bad"
    elif field == "Div. Yield (%)":
        color = "metric-good" if v >= 4 else "metric-warn" if v >= 2 else "metric-bad"
    else:
        return f"{v:.2f}"
    return f'<span class="{color}">{v:.2f}</span>'

PLOTLY_COLORS = ["#cba6f7", "#a6e3a1", "#f38ba8", "#89dceb", "#fab387", "#f9e2af", "#b4befe", "#94e2d5"]


# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("📊 Long / Short Dashboard")
st.caption("Ratio em tempo real · Z-Score · Cointegração · Fundamentos · Análise via Claude AI")

tab_ls, tab_fund = st.tabs(["📈 Long / Short", "🏦 Fundamentos"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Long / Short
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ls:
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fundamentos
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fund:
    st.subheader("🏦 Análise de Fundamentos")

    tickers_raw = [t.strip().upper() for t in fund_tickers_input.replace(";", ",").split(",") if t.strip()]

    if not fund_btn and not tickers_raw:
        st.info("Selecione um grupo ou insira tickers na barra lateral e clique em **🔍 Carregar Fundamentos**.")
        st.stop()

    if not tickers_raw:
        st.warning("Nenhum ticker informado.")
        st.stop()

    with st.spinner(f"Buscando fundamentos de {len(tickers_raw)} ativo(s)…"):
        df_fund = fetch_fundamentals(tuple(tickers_raw))

    if df_fund.empty:
        st.error("Não foi possível carregar dados. Verifique os tickers.")
        st.stop()

    # Sub-tabs: Comparar | Individual
    sub_compare, sub_individual = st.tabs(["⚖️ Comparar Empresas", "🔍 Empresa Individual"])

    # ── Sub-tab: Comparar ────────────────────────────────────────────────────
    with sub_compare:
        st.markdown("#### Tabela Comparativa")

        # Display table with formatting
        display_df = df_fund.copy()
        for col in display_df.columns:
            if col == "Market Cap":
                display_df[col] = display_df[col].apply(fmt_market_cap)
            elif col in PERCENT_FIELDS:
                display_df[col] = display_df[col].apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
            elif col not in ("Nome", "Setor"):
                display_df[col] = display_df[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")

        st.dataframe(display_df, use_container_width=True)

        st.divider()
        st.markdown("#### Gráficos Comparativos")

        # Pick which metric to chart
        available_chart_fields = [f for f in NUMERIC_CHART_FIELDS if f in df_fund.columns]
        selected_metrics = st.multiselect(
            "Métricas para comparar",
            available_chart_fields,
            default=available_chart_fields[:4],
            key="compare_metrics",
        )

        if selected_metrics:
            n_cols = min(2, len(selected_metrics))
            n_rows = -(-len(selected_metrics) // n_cols)  # ceiling div
            fig_cmp = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_metrics,
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )
            for idx, metric in enumerate(selected_metrics):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                series = df_fund[metric].dropna()
                if series.empty:
                    continue
                bar_colors = [PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i in range(len(series))]
                fig_cmp.add_trace(
                    go.Bar(
                        x=series.index.tolist(),
                        y=series.values.tolist(),
                        marker_color=bar_colors,
                        showlegend=False,
                        text=[f"{v:.2f}" for v in series.values],
                        textposition="outside",
                    ),
                    row=row, col=col,
                )

            fig_cmp.update_layout(
                height=300 * n_rows,
                paper_bgcolor="#1e1e2e",
                plot_bgcolor="#1e1e2e",
                font=dict(color="#cdd6f4"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig_cmp.update_xaxes(gridcolor="#313244")
            fig_cmp.update_yaxes(gridcolor="#313244")
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Radar chart for multi-metric comparison
        st.divider()
        st.markdown("#### Radar — Perfil Comparativo")
        radar_metrics = [f for f in ["P/L", "ROE (%)", "Marg. Líq. (%)", "EV/EBITDA", "Div. Yield (%)", "Beta"]
                         if f in df_fund.columns]

        if radar_metrics and len(df_fund) >= 2:
            # Normalize each metric 0-1 across companies (min-max)
            radar_df = df_fund[radar_metrics].copy().astype(float)
            for col in radar_df.columns:
                mn, mx = radar_df[col].min(), radar_df[col].max()
                if mx != mn:
                    radar_df[col] = (radar_df[col] - mn) / (mx - mn)
                else:
                    radar_df[col] = 0.5

            fig_radar = go.Figure()
            for i, ticker in enumerate(radar_df.index):
                vals = radar_df.loc[ticker].tolist()
                vals += [vals[0]]  # close polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=ticker,
                    line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)]),
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#313244",
                    radialaxis=dict(visible=True, gridcolor="#45475a", color="#cdd6f4"),
                    angularaxis=dict(gridcolor="#45475a", color="#cdd6f4"),
                ),
                paper_bgcolor="#1e1e2e",
                font=dict(color="#cdd6f4"),
                legend=dict(bgcolor="#313244"),
                height=450,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Radar normalizado (0 = pior, 1 = melhor no grupo) — apenas para comparação relativa entre os ativos selecionados.")
        else:
            st.info("Selecione ao menos 2 empresas para o gráfico de radar.")

    # ── Sub-tab: Individual ──────────────────────────────────────────────────
    with sub_individual:
        valid_tickers = [t for t in tickers_raw if t in df_fund.index]
        if not valid_tickers:
            st.warning("Nenhum ticker com dados disponíveis.")
            st.stop()

        selected_ticker = st.selectbox("Selecione a empresa", valid_tickers, key="ind_ticker")
        row_data = df_fund.loc[selected_ticker]

        company_name = row_data.get("Nome") or selected_ticker
        sector       = row_data.get("Setor") or "—"

        st.markdown(f"### {company_name}")
        st.caption(f"**Ticker:** {selected_ticker} &nbsp;|&nbsp; **Setor:** {sector}")
        st.divider()

        # Valuation metrics
        st.markdown("**📐 Valuation**")
        v1, v2, v3, v4 = st.columns(4)
        def _fmt(val, pct=False, mult=False):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            if pct:
                return f"{float(val):.2f}%"
            if mult:
                return f"{float(val):.2f}x"
            return f"{float(val):.2f}"

        v1.metric("P/L",        _fmt(row_data.get("P/L"),        mult=True))
        v2.metric("P/L Fwd",    _fmt(row_data.get("P/L Fwd"),    mult=True))
        v3.metric("P/VP",       _fmt(row_data.get("P/VP"),        mult=True))
        v4.metric("EV/EBITDA",  _fmt(row_data.get("EV/EBITDA"),   mult=True))

        # Profitability
        st.markdown("**💰 Rentabilidade**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("ROE",            _fmt(row_data.get("ROE (%)"),        pct=True))
        p2.metric("ROA",            _fmt(row_data.get("ROA (%)"),        pct=True))
        p3.metric("Margem Líquida", _fmt(row_data.get("Marg. Líq. (%)"),  pct=True))
        p4.metric("Margem EBITDA",  _fmt(row_data.get("Marg. EBITDA (%)"), pct=True))

        # Financial health
        st.markdown("**🏗️ Saúde Financeira**")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Dívida/PL",      _fmt(row_data.get("Dív./PL")))
        h2.metric("Liq. Corrente",  _fmt(row_data.get("Liq. Corrente")))
        h3.metric("Div. Yield",     _fmt(row_data.get("Div. Yield (%)"), pct=True))
        h4.metric("Beta",           _fmt(row_data.get("Beta")))

        # Growth & size
        st.markdown("**📈 Crescimento & Tamanho**")
        g1, g2, g3 = st.columns(3)
        g1.metric("Cresc. Receita", _fmt(row_data.get("Cresc. Receita (%)"), pct=True))
        g2.metric("Cresc. Lucro",   _fmt(row_data.get("Cresc. Lucro (%)"),   pct=True))
        g3.metric("Market Cap",     fmt_market_cap(row_data.get("Market Cap")))

        st.divider()

        # Horizontal bar chart for all numeric metrics
        st.markdown("**📊 Visão Geral dos Indicadores**")
        chart_fields = [f for f in NUMERIC_CHART_FIELDS if f in df_fund.columns]
        ind_vals = {f: row_data.get(f) for f in chart_fields}
        ind_vals = {k: v for k, v in ind_vals.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

        if ind_vals:
            fig_ind = go.Figure(go.Bar(
                x=list(ind_vals.values()),
                y=list(ind_vals.keys()),
                orientation="h",
                marker_color="#cba6f7",
                text=[f"{v:.2f}" for v in ind_vals.values()],
                textposition="outside",
            ))
            fig_ind.update_layout(
                height=max(300, len(ind_vals) * 45),
                paper_bgcolor="#1e1e2e",
                plot_bgcolor="#1e1e2e",
                font=dict(color="#cdd6f4"),
                margin=dict(l=0, r=60, t=20, b=0),
                xaxis=dict(gridcolor="#313244"),
                yaxis=dict(gridcolor="#313244"),
            )
            st.plotly_chart(fig_ind, use_container_width=True)

        # Historical price chart
        st.divider()
        st.markdown("**📉 Histórico de Preços (1 ano)**")
        with st.spinner("Carregando histórico de preços…"):
            hist_data = fetch(selected_ticker, "1y", "1d")
        if not hist_data.empty:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=hist_data.index, y=hist_data.values,
                name=selected_ticker,
                line=dict(color="#cba6f7", width=2),
                fill="tozeroy",
                fillcolor="rgba(203,166,247,0.1)",
            ))
            fig_price.update_layout(
                height=300,
                paper_bgcolor="#1e1e2e",
                plot_bgcolor="#1e1e2e",
                font=dict(color="#cdd6f4"),
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(gridcolor="#313244"),
                yaxis=dict(gridcolor="#313244"),
                showlegend=False,
            )
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("Histórico de preços não disponível.")

    st.divider()
    st.caption(f"Dados via Yahoo Finance · Fundamentos podem ter defasagem · Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
