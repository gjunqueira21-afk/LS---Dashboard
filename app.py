"""LS Dashboard — versão desktop.

Toda a lógica quantitativa vive em core.py; os dados em datasource.py; o
visual em theme.py. Este arquivo é só a camada de apresentação — foi assim
que app.py e mobile.py pararam de divergir.
"""

import html
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
import theme
from core import SETUP_VALIDADO
from datasource import TZ_BR, DataError, align_pair, fetch_series
from theme import (C_ACCENT, C_GRAPHIC, C_LONG, C_MUTED, C_SHORT, C_STAT_OK,
                   PLOTLY_CONFIG, metric_card, rgba, style_fig, z_bar_colors,
                   z_tone)

st.set_page_config(page_title="LS Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(theme.CSS, unsafe_allow_html=True)

PERIOD_MAP = {"6 meses": "6mo", "1 ano": "1y", "2 anos": "2y",
              "3 anos": "3y", "5 anos": "5y"}


def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def load_pair(long_in: str, short_in: str, period: str, mean_win: int,
              vol_win: int, corr_win: int, corr_basis: str):
    """Baixa, alinha e calcula. Cacheado por (par, período, janelas).

    Os gatilhos NÃO entram na chave: mexer num slider de z não pode refazer
    coint, adfuller e polyfit.
    """
    lr = fetch_series(long_in, period)
    sr = fetch_series(short_in, period)
    al = align_pair(lr.series, sr.series)
    stats = core.compute_pair(al.frame, mean_win, vol_win, corr_win, corr_basis)
    return lr, sr, al, stats


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.title("Configurações")

    st.subheader("Par")
    # A troca precisa acontecer ANTES dos widgets existirem: o Streamlit
    # proíbe reescrever a chave de um widget já instanciado no mesmo run.
    if st.session_state.pop("_do_swap", False):
        a = st.session_state.get("in_long", "PETR4")
        b = st.session_state.get("in_short", "VALE3")
        st.session_state["in_long"], st.session_state["in_short"] = b, a

    c1, c2 = st.columns(2)
    with c1:
        long_in = st.text_input("LONG", value="PETR4", key="in_long",
                                help="Digite só o ticker: petr4, vale3, bova11, "
                                     "aapl, ibov. O sufixo .SA é automático.")
    with c2:
        short_in = st.text_input("SHORT", value="VALE3", key="in_short")

    if st.button("⇄ inverter par", width="stretch"):
        st.session_state["_do_swap"] = True
        st.rerun()

    period_label = st.selectbox("Histórico", list(PERIOD_MAP.keys()), index=2)
    period = PERIOD_MAP[period_label]

    run_btn = st.button("▶ Analisar", width="stretch", type="primary")
    st.divider()

    # --- janelas -------------------------------------------------------------
    with st.expander(f"Janelas · {SETUP_VALIDADO['mean_win']}/"
                     f"{SETUP_VALIDADO['vol_win']}/{SETUP_VALIDADO['corr_win']}",
                     expanded=False):
        mean_win = st.number_input("Média do spread (pregões)", 5, 504,
                                   SETUP_VALIDADO["mean_win"])
        vol_win = st.number_input("Volatilidade (pregões)", 5, 504,
                                  SETUP_VALIDADO["vol_win"])
        corr_win = st.number_input("Correlação (pregões)", 20, 504,
                                   SETUP_VALIDADO["corr_win"])
        corr_basis = st.radio(
            "Base da correlação",
            ["returns", "levels"], index=0, horizontal=True,
            format_func=lambda x: "log-retornos (correto)" if x == "returns"
            else "níveis (legado)",
            help="Correlação entre séries de preço em nível é espúria — mede "
                 "tendência comum, não co-movimento.",
        )
        if mean_win != vol_win:
            st.warning(
                "Janelas descasadas: o resultado deixa de ser um z-score "
                "(numerador e denominador medem horizontes diferentes e a "
                "variância não é 1). Os gatilhos foram calibrados em "
                f"{SETUP_VALIDADO['mean_win']}/{SETUP_VALIDADO['vol_win']}."
            )

    # --- gatilhos ------------------------------------------------------------
    with st.expander("Gatilhos de sinal", expanded=False):
        st.caption("Monitorar")
        z_monitor = st.slider("|z| >=", 0.5, 3.0, SETUP_VALIDADO["z_monitor"], 0.05)
        dz_monitor = st.slider("dz (1 pregão) >=", 0.0, 0.5,
                               SETUP_VALIDADO["dz_monitor"], 0.01)
        corr_min = st.slider("Correlação >=", 0.0, 1.0,
                             SETUP_VALIDADO["corr_min"], 0.05)
        st.caption("Alerta máximo")
        z_alert = st.slider("|z| >= ", 0.5, 3.0, SETUP_VALIDADO["z_alert"], 0.05)
        dz_alert = st.slider("dz (1 pregão) >= ", 0.0, 0.5,
                             SETUP_VALIDADO["dz_alert"], 0.01)

    with st.expander("Saída / convergência", expanded=False):
        z_exit = st.slider("Convergência |z| <=", 0.0, 1.0,
                           SETUP_VALIDADO["z_exit"], 0.05)
        max_hold = st.number_input("Máx. holding (pregões)", 5, 252,
                                   SETUP_VALIDADO["max_hold"])
        tem_posicao = st.checkbox(
            "Tenho posição aberta neste par", value=False,
            help="Sem posição, 'CONVERGÊNCIA -> encerrar' não faz sentido — o "
                 "painel mostra 'SEM EDGE' em vez disso.")

    with st.expander("Custos", expanded=False):
        cost_rt = st.number_input(
            "Custo round-trip nas 2 pernas (%)", 0.0, 5.0,
            core.CUSTO_RT_PADRAO * 100, 0.01,
            help="Emolumentos + liquidação + corretagem + slippage.") / 100
        rent_aa = st.number_input(
            "Aluguel da perna short (% a.a.)", 0.0, 100.0,
            core.ALUGUEL_AA_PADRAO * 100, 0.5,
            help="Taxa BTC do papel. Use a taxa real, não a média — em papel "
                 "apertado ela vai a dois dígitos e mata o par sozinha.") / 100
        capital = st.number_input("Capital por perna (R$)", 1_000, 50_000_000,
                                  100_000, 10_000)

    with st.expander("API", expanded=False):
        api_key = st.text_input("Anthropic API Key",
                                value=get_secret("ANTHROPIC_API_KEY", ""),
                                type="password", placeholder="sk-ant-...")

    st.divider()
    auto_refresh = st.toggle("Auto-refresh no pregão (60s)", value=False)

cfg = {"z_monitor": z_monitor, "dz_monitor": dz_monitor, "corr_min": corr_min,
       "z_alert": z_alert, "dz_alert": dz_alert, "z_exit": z_exit,
       "max_hold": max_hold, "corr_win": corr_win}

# Aviso de drift: você está no setup validado ou em algo que mexeu semana passada?
drift = [k for k, v in SETUP_VALIDADO.items()
         if k in cfg and abs(cfg[k] - v) > 1e-9]
if mean_win != SETUP_VALIDADO["mean_win"]:
    drift.append(f"mean_win={mean_win}")
if vol_win != SETUP_VALIDADO["vol_win"]:
    drift.append(f"vol_win={vol_win}")

# =============================================================================
# Commit: só o que afeta o cálculo pesado exige clique em Analisar
# =============================================================================
data_key = (long_in.strip(), short_in.strip(), period, mean_win, vol_win,
            corr_win, corr_basis)
if run_btn:
    st.session_state["committed"] = data_key
committed = st.session_state.get("committed")

if committed is None:
    st.markdown(
        '<div class="empty-state">Digite o par na barra lateral — só o ticker, '
        'sem <b>.SA</b> — e clique em <b>Analisar</b>.</div>',
        unsafe_allow_html=True)
    st.stop()

stale = committed != data_key

try:
    with st.spinner("Baixando e calculando..."):
        lr, sr, al, ps = load_pair(*committed)
except DataError as err:
    st.error(err.message)
    if err.retryable:
        st.button("Tentar de novo")
    st.stop()
except ValueError as err:
    st.error(str(err))
    st.stop()

sig = core.classify(ps, cfg, tem_posicao=tem_posicao)
L, S = html.escape(lr.display), html.escape(sr.display)
hold_ref = ps.hl[0] if ps.hl and np.isfinite(ps.hl[0]) else max_hold
z_be = core.breakeven_z(ps.sigma, cost_rt, rent_aa, hold_ref)

# =============================================================================
# Cabeçalho + sinal (um bloco só)
# =============================================================================
hoje = pd.Timestamp(datetime.now().date())
atraso = int(np.busday_count(ps.last_date.date(), hoje.date()))
if atraso <= 0:
    badge = f'<span class="ls-meta">último pregão · {ps.last_date:%d/%m/%Y}</span>'
elif atraso == 1:
    badge = (f'<span class="ls-meta">fechamento de <b>{ps.last_date:%d/%m/%Y}</b>'
             f' · D-1</span>')
else:
    badge = (f'<span class="ls-meta ls-stale">dado de {ps.last_date:%d/%m/%Y} '
             f'· {atraso} pregões atrás</span>')

dir_cls = "dir-long" if sig.side == "long_ratio" else ""
side_txt = core.side_text(sig.side, L, S)
side_html = f'<div class="sb-side">{side_txt}</div>' if side_txt else ""
gates_html = "".join(
    f'<span class="gate {"ok" if g.ok else "fail"}">'
    f'<span class="g-mark">{"OK" if g.ok else "X"}</span> '
    f'{g.nome} <b>{g.valor}</b> <span class="g-thr">{g.gatilho}</span></span>'
    for g in sig.gates)

st.markdown(f"""
<div class="ls-head">
  <div class="ls-pair">
    <span class="chip chip-long">LONG · {L}</span>
    <span class="pair-x">x</span>
    <span class="chip chip-short">SHORT · {S}</span>
    {badge}
  </div>
  <div class="lv-{sig.level} {dir_cls}">
    <div class="sb-label">{sig.label}</div>
    {side_html}
    <div class="sb-desc">{sig.desc}</div>
  </div>
  <div class="gates">{gates_html}</div>
</div>
""", unsafe_allow_html=True)

if stale:
    st.warning("A configuração mudou desde o último cálculo. "
               "Clique em **Analisar** para atualizar — os números abaixo "
               "ainda são do par anterior.")

for note in [n for n in (lr.resolution_note, sr.resolution_note) if n]:
    st.caption(note)
for w in lr.warnings + sr.warnings + ps.problems:
    st.warning(w)
if al.pct_dropped >= 2:
    st.warning(
        f"Calendários diferentes: {al.n_dropped} pregões descartados "
        f"({al.pct_dropped:.1f}% da amostra) por falta de negócio em um dos "
        f"ativos. Estatísticas calculadas sobre {al.n_used} pregões.")
if drift:
    st.info("Parâmetros fora do setup validado do backtest: "
            + ", ".join(str(d) for d in drift))

# checklist por par — chaves com o ticker, senão vaza entre pares
if sig.level == "alert":
    st.markdown("**Checklist do Alerta Máximo — validar antes de executar:**")
    k = f"{lr.symbol}_{sr.symbol}"
    q1, q2, q3, q4 = st.columns(4)
    q1.checkbox("Liquidez OK", key=f"chk_liq_{k}")
    q2.checkbox("Spread de tela OK", key=f"chk_spr_{k}")
    q3.checkbox("Aluguel checado", key=f"chk_alu_{k}")
    q4.checkbox("Sem evento societário", key=f"chk_evt_{k}")


# =============================================================================
# Métricas — 2 linhas de 3 (6 colunas truncavam os rótulos)
# =============================================================================
def pv(p, ok_txt, bad_txt):
    if p is None:
        return "n/d", "teste não pôde ser calculado", "na"
    return f"{p:.4f}", (ok_txt if p < 0.05 else bad_txt), ("ok" if p < 0.05 else "weak")


r1 = st.columns(3)
r1[0].markdown(metric_card(
    "Z-Score", f"{ps.z_now:+.3f}",
    f"sigma do spread: {ps.sigma:.4f}", z_tone(ps.z_now, cfg)),
    unsafe_allow_html=True)

mom = {True: "esticando", False: "revertendo", None: "parado"}[sig.esticando]
r1[1].markdown(metric_card(
    "Delta z · 1 pregão", f"{ps.dz_now:+.3f}",
    f"{mom} · ruído de fundo {ps.dz_noise:.3f}", "accent"), unsafe_allow_html=True)

if ps.hl is None:
    hl_v, hl_s, hl_t = "n/d", "amostra insuficiente", "na"
elif not np.isfinite(ps.hl[0]):
    hl_v, hl_s, hl_t = "infinita", "spread não reverte", "weak"
else:
    _hl, _lo, _hi = ps.hl
    _hi_txt = "inf" if not np.isfinite(_hi) else f"{_hi:.0f}"
    hl_v, hl_s = f"{_hl:.0f}p", f"IC 95%: {_lo:.0f}-{_hi_txt}p"
    hl_t = "ok" if _hl <= max_hold else "weak"
r1[2].markdown(metric_card("Meia-vida de reversão", hl_v, hl_s, hl_t),
               unsafe_allow_html=True)

r2 = st.columns(3)
if ps.corr_now is None:
    cv, cs, ct = "n/d", f"janela {corr_win}p > histórico", "na"
else:
    base = "retornos" if corr_basis == "returns" else "níveis"
    cv, cs = f"{ps.corr_now:.3f}", f"{base} · gatilho >= {corr_min:.2f}"
    ct = "ok" if ps.corr_now >= corr_min else "weak"
r2[0].markdown(metric_card(f"Correlação {corr_win}p", cv, cs, ct),
               unsafe_allow_html=True)

v, s_, t = pv(ps.coint_p, "cointegrado", "não cointegrado")
r2[1].markdown(metric_card("Cointegração · p", v, s_, t), unsafe_allow_html=True)

v, s_, t = pv(ps.adf_p, "spread estacionário", "spread não estacionário")
r2[2].markdown(metric_card("ADF do spread · p", v, s_, t), unsafe_allow_html=True)

# edge líquido — o número que responde "esse sinal paga a conta?"
edge = abs(ps.z_now) - z_exit - z_be
st.markdown("")
e1, e2 = st.columns([1, 2])
e1.markdown(metric_card(
    "Edge líquido", f"{edge:+.2f} z", f"break-even em {z_be:.2f} z",
    "long" if edge > 0.30 else "short" if edge < 0 else "weak"),
    unsafe_allow_html=True)
with e2:
    custo_total = cost_rt + rent_aa * hold_ref / 252
    if z_be > z_exit:
        st.error(
            f"**A regra de saída não paga o custo.** O break-even é "
            f"{z_be:.2f} z e você sai em |z| <= {z_exit:.2f} — sair nesse "
            f"ponto é sair no zero a zero ou pior. Custo total considerado: "
            f"{custo_total * 100:.2f}% do notional em {hold_ref:.0f} pregões.")
    elif edge < 0:
        st.warning(f"Amplitude disponível (|z| {abs(ps.z_now):.2f} até a saída "
                   f"em {z_exit:.2f}) menor que o custo de {z_be:.2f} z.")
    else:
        st.success(f"Amplitude de {abs(ps.z_now) - z_exit:.2f} z contra custo "
                   f"de {z_be:.2f} z — sobram {edge:.2f} z líquidos.")

st.markdown("")

# =============================================================================
# Abas
# =============================================================================
tab_g, tab_e, tab_s, tab_ia, tab_m = st.tabs(
    ["Gráficos", "Estatísticas", "Sizing", "Análise IA", "Manual"])

with tab_g:
    # Ordem: Z-Score primeiro. Antes ele nascia abaixo da dobra.
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.40, 0.33, 0.27],
        subplot_titles=[f"Z-Score ({mean_win}p / {vol_win}p)",
                        f"Spread ln({L}) - ln({S}) · média {mean_win}p",
                        "Preços normalizados (base 100)"],
        vertical_spacing=0.08)

    # 1) Z-Score — zonas como faixas, não 7 linhas horizontais
    fig.add_trace(go.Bar(x=ps.z.index, y=ps.z, name="Z-Score",
                         marker_color=z_bar_colors(ps.z, cfg),
                         hovertemplate="z = %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    zvals = ps.z.dropna()
    zmax = float(np.nanmax(np.abs(zvals.values))) if len(zvals) else 3.0
    top = max(zmax * 1.1, z_alert * 1.3)
    fig.add_hrect(y0=z_alert, y1=top, fillcolor=rgba(C_SHORT, 0.07),
                  layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=-top, y1=-z_alert, fillcolor=rgba(C_LONG, 0.07),
                  layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=-z_exit, y1=z_exit, fillcolor=rgba(C_ACCENT, 0.10),
                  layer="below", line_width=0, row=1, col=1)
    fig.add_hline(y=0, line_color=C_GRAPHIC, opacity=0.8, row=1, col=1)
    fig.add_hline(y=z_monitor, line_dash="dash", line_color=C_GRAPHIC,
                  opacity=0.6, row=1, col=1)
    fig.add_hline(y=-z_monitor, line_dash="dash", line_color=C_GRAPHIC,
                  opacity=0.6, row=1, col=1)
    fig.update_yaxes(range=[-top, top], row=1, col=1)

    # 2) Spread + bandas (bandas fora da legenda: são autoevidentes)
    roll_m = ps.spread.rolling(mean_win).mean()
    roll_s = ps.spread.rolling(vol_win).std()
    fig.add_trace(go.Scatter(x=ps.spread.index, y=ps.spread, name="Spread",
                             line=dict(color=C_ACCENT, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m, name="Média",
                             line=dict(color=C_GRAPHIC, width=1)), row=2, col=1)
    for mult, color in ((z_alert, C_SHORT), (-z_alert, C_LONG)):
        fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m + mult * roll_s,
                                 showlegend=False, hoverinfo="skip",
                                 line=dict(color=color, width=1, dash="dot")),
                      row=2, col=1)
    for mult in (z_monitor, -z_monitor):
        col = C_SHORT if mult > 0 else C_LONG
        fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m + mult * roll_s,
                                 showlegend=False, hoverinfo="skip",
                                 line=dict(color=rgba(col, 0.45), width=1,
                                           dash="dash")),
                      row=2, col=1)

    # 3) Preços normalizados — normalizados DEPOIS do alinhamento
    ln = ps.frame["long"] / ps.frame["long"].iloc[0] * 100
    sn = ps.frame["short"] / ps.frame["short"].iloc[0] * 100
    fig.add_trace(go.Scatter(x=ln.index, y=ln, name=f"LONG {L}",
                             line=dict(color=C_LONG, width=2)), row=3, col=1)
    # dash na perna SHORT: sob deuteranopia as duas cores convergem
    fig.add_trace(go.Scatter(x=sn.index, y=sn, name=f"SHORT {S}",
                             line=dict(color=C_SHORT, width=2, dash="dot")),
                  row=3, col=1)

    style_fig(fig, height=620, top_margin=70)
    fig.update_annotations(font=dict(size=13, color=C_MUTED))
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

    st.caption("Faixas: vermelha = zona de venda do ratio · verde = zona de "
               "compra · roxa = banda de saída. Tracejado = monitorar, "
               "pontilhado = alerta.")

with tab_e:
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Estatísticas do spread canônico**")
        st.dataframe(pd.DataFrame({
            "Métrica": ["Spread atual", f"Média ({mean_win}p)", "sigma do spread",
                        "Z-Score", "Delta z (1 pregão)", "Z máx", "Z mín",
                        "Meia-vida", "beta OLS em log (informativo)",
                        f"Correlação ({corr_win}p)", "Cointegração p",
                        "ADF do spread p", "Pregões usados"],
            "Valor": [f"{ps.ratio_now:.4f}", f"{ps.ratio_mean:.4f}",
                      f"{ps.sigma:.4f}", f"{ps.z_now:+.3f}", f"{ps.dz_now:+.3f}",
                      f"{ps.z.max():.3f}", f"{ps.z.min():.3f}",
                      hl_v, f"{ps.beta_now:.3f}",
                      "n/d" if ps.corr_now is None else f"{ps.corr_now:.3f}",
                      "n/d" if ps.coint_p is None else f"{ps.coint_p:.4f}",
                      "n/d" if ps.adf_p is None else f"{ps.adf_p:.4f}",
                      f"{ps.n_obs}"],
        }), width="stretch", hide_index=True)

        nota_beta = ("O beta é **informativo**: o sinal, o teste e o sizing "
                     "usam beta = 1 (notional-neutro).")
        if len(ps.beta_roll) > 20:
            br = ps.beta_roll.tail(252)
            nota_beta += (f" Faixa do beta rolante em 12 meses: "
                          f"{br.min():.2f}-{br.max():.2f} — ele muda sozinho "
                          f"conforme você troca o seletor de histórico.")
        st.caption(nota_beta)

    with cb:
        st.markdown("**Frequência empírica dos gatilhos neste par**")
        st.dataframe(pd.DataFrame({
            "Gatilho": [f"Monitorar |z| >= {z_monitor:.2f}",
                        f"Alerta |z| >= {z_alert:.2f}",
                        f"Saída |z| <= {z_exit:.2f}"],
            "Dispara em": [f"{core.z_frequency(ps.z, z_monitor):.1%} dos pregões",
                           f"{core.z_frequency(ps.z, z_alert):.1%} dos pregões",
                           f"{1 - core.z_frequency(ps.z, z_exit):.1%} dos pregões"],
        }), width="stretch", hide_index=True)
        st.caption(
            f"sigma do z neste par: **{ps.z.std():.2f}** — o z de janela "
            f"rolante não é normal padrão. Um limiar fixo em sigma dispara com "
            f"frequência diferente em cada par: leia a frequência, não o número.")

        st.markdown("**Correlação rolante**")
        cf = go.Figure()
        cf.add_trace(go.Scatter(x=ps.corr.index, y=ps.corr, name="Correlação",
                                line=dict(color=C_STAT_OK, width=2)))
        cf.add_hline(y=corr_min, line_dash="dot", line_color=C_GRAPHIC, opacity=0.7)
        cmin = float(ps.corr.min()) if len(ps.corr.dropna()) else -0.2
        cf.update_yaxes(range=[max(-1.0, cmin - 0.1), 1.0])
        style_fig(cf, height=220, top_margin=20, show_legend=False)
        st.plotly_chart(cf, width="stretch", config=PLOTLY_CONFIG)

with tab_s:
    sz = core.sizing(ps.frame, capital)
    st.markdown(f"**Sizing notional-neutro (beta = 1), lotes de 100 — "
                f"capital de R$ {capital:,.0f} por perna**".replace(",", "."))
    g1, g2, g3 = st.columns(3)
    g1.markdown(metric_card(
        f"LONG · {L}", f"{sz['q_long']}",
        f"x R$ {sz['p_long']:.2f} = R$ {sz['fin_long']:,.0f}".replace(",", "."),
        "long"), unsafe_allow_html=True)
    g2.markdown(metric_card(
        f"SHORT · {S}", f"{sz['q_short']}",
        f"x R$ {sz['p_short']:.2f} = R$ {sz['fin_short']:,.0f}".replace(",", "."),
        "short"), unsafe_allow_html=True)
    g3.markdown(metric_card(
        "Resíduo de lote", f"R$ {sz['residuo']:,.0f}".replace(",", "."),
        f"{sz['residuo_pct']:+.2%} do capital — exposição direcional",
        "weak" if abs(sz["residuo_pct"]) > 0.02 else "ok"), unsafe_allow_html=True)

    st.caption("O resíduo de arredondamento de lote é exposição direcional "
               "real. Em papel caro ele fica grande e normalmente ninguém "
               "percebe.")

    st.markdown("**Decomposição do custo**")
    st.dataframe(pd.DataFrame({
        "Componente": ["Round-trip nas 2 pernas",
                       f"Aluguel ({hold_ref:.0f} pregões)",
                       "Custo total", "sigma do spread", "z de break-even"],
        "Valor": [f"{cost_rt:.2%}", f"{rent_aa * hold_ref / 252:.2%}",
                  f"{cost_rt + rent_aa * hold_ref / 252:.2%}",
                  f"{ps.sigma:.4f}", f"{z_be:.2f} z"],
    }), width="stretch", hide_index=True)

with tab_ia:
    if not api_key:
        st.info("Adicione sua Anthropic API Key no expander **API** da barra "
                "lateral para ativar a análise.")
    else:
        # o hash inclui os gatilhos: a análise antiga citava parâmetros que
        # você já tinha mudado, como se fossem os atuais
        akey = f"ia_{lr.symbol}_{sr.symbol}_{hash(frozenset(cfg.items()))}_{period}"
        if st.button("Gerar análise com Claude", type="secondary"):
            with st.spinner("Claude analisando o par..."):
                try:
                    st.session_state[akey] = core.analyze_with_claude(
                        api_key, L, S, ps, sig, cfg, z_be)
                    st.session_state[akey + "_ts"] = datetime.now()
                except Exception as e:
                    st.error(f"Erro na chamada à API: {e}")
        if akey in st.session_state:
            ts = st.session_state.get(akey + "_ts")
            if ts:
                st.caption(f"Gerada em {ts:%d/%m/%Y %H:%M} com |z| >= "
                           f"{z_alert:.2f} · corr >= {corr_min:.2f}")
            st.markdown(st.session_state[akey])

with tab_m:
    st.markdown(f"""
### Especificação canônica

O painel usa **um único spread** para tudo — sinal, teste estatístico, gráfico
e dimensionamento:

    spread = ln({L}) - ln({S})

Isso é **beta = 1 imposto**: posição *notional-neutra*, mesmo valor financeiro
em cada perna. O beta OLS aparece na aba Estatísticas como número
**informativo** e não entra em lugar nenhum do cálculo.

### As métricas

**Z-Score** — distância do spread em relação à média de {mean_win} pregões, em
desvios de {vol_win} pregões. Coração da estratégia.

**Delta z** — variação do z em 1 pregão. Confirma momentum. Atenção: ele
mistura o movimento de hoje com o efeito da janela *descartando* a observação
de {mean_win} pregões atrás. O card mostra o **ruído de fundo**
({ps.dz_noise:.3f} neste par) — um gatilho abaixo desse valor é ruído, não
momentum.

**Meia-vida de reversão** — quanto tempo o spread leva para percorrer metade do
caminho de volta à média. É o número que responde *quanto tempo meu capital
fica preso*. Leia o **intervalo de confiança**, não o ponto: o estimador é
enviesado para baixo em meias-vidas longas.

**Correlação ({corr_win}p)** — sobre **log-retornos**, não sobre níveis de
preço. Correlação entre séries de preço em nível é espúria: mede tendência
comum. É por isso que o gatilho padrão aqui é
{SETUP_VALIDADO['corr_min']:.2f} e não 0,75 — sobre retornos, 0,75 entre dois
papéis do mesmo setor é raro.

**Cointegração p** — Engle-Granger sobre log-preços, com os valores críticos
corretos para o teste.

**ADF do spread p** — Dickey-Fuller sobre o próprio spread canônico. É legítimo
usar `adfuller` aqui **porque beta = 1 foi imposto, não estimado**.

**z de break-even** — custo total (round-trip nas duas pernas + aluguel no
holding) convertido para unidades de z. Responde *esse sinal paga a conta?*.
Se o break-even for maior que a banda de saída, sair em |z| <= {z_exit:.2f} é
sair no zero a zero.

---
### Níveis de sinal

- **Monitorar** — `|z| >= {z_monitor:.2f}` · `dz >= {dz_monitor:.2f}` ·
  `corr >= {corr_min:.2f}`
- **Alerta máximo** — `|z| >= {z_alert:.2f}` · `dz >= {dz_alert:.2f}` ·
  `corr >= {corr_min:.2f}` + checklist
- **Saída** — `|z| <= {z_exit:.2f}` ou cruzamento do zero vindo de dentro da
  banda. Holding máximo: {max_hold} pregões.

Quando o sinal é NEUTRO, a tríade de check no topo mostra **qual condição
falhou** — não só que o conjunto não passou.

### Limitação conhecida

O gatilho de delta z usa o **módulo**. Isso significa que "esticando" (dz na
mesma direção do z) e "revertendo" (direção contrária) — dois trades
economicamente opostos — entram no mesmo rótulo. O painel mostra a direção no
card de delta z para você ver, mas não separa os dois. Separar exigiria saber
qual das duas semânticas o backtest validou.

> Rodar em produção sombra antes de execução real. Registrar cada sinal
> emitido é a única amostra out-of-sample limpa que ainda pode existir.
    """)

st.caption(f"Dados via Yahoo Finance · último pregão {ps.last_date:%d/%m/%Y} · "
           f"página renderizada {datetime.now():%H:%M:%S}")

# Auto-refresh no FIM do script — antes ele ficava ANTES de todo o cálculo,
# então o app dormia 30s e reiniciava sem nunca renderizar nada.
if auto_refresh:
    agora = datetime.now(TZ_BR)
    if agora.weekday() < 5 and 10 <= agora.hour < 19:
        time.sleep(60)
        st.rerun()
    else:
        st.caption("Auto-refresh pausado fora do pregão.")
