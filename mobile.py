"""LS Dashboard — versão celular.

Importa o MESMO core.py, datasource.py e theme.py do desktop. As duas telas não
podem mais divergir: antes as cores do z-score significavam coisas opostas nas
duas versões (barra verde = alerta máximo no desktop, apenas monitorar no
celular) e o cfg do mobile nem passava corr_win.
"""

import html
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
import theme
from core import SETUP_VALIDADO
from datasource import DataError, align_pair, fetch_series
from theme import (C_ACCENT, C_GRAPHIC, C_LONG, C_MUTED, C_SHORT,
                   PLOTLY_CONFIG, metric_card, rgba, style_fig, z_bar_colors,
                   z_tone)

st.set_page_config(page_title="LS Mobile", page_icon="📱",
                   layout="centered", initial_sidebar_state="collapsed")

# Herda o design system e só sobrescreve o que é específico de tela pequena.
st.markdown(theme.CSS, unsafe_allow_html=True)
st.markdown("""
<style>
    .block-container { padding-top: 0.8rem; padding-bottom: 3rem; max-width: 640px; }
    .stButton button { height: 3rem; font-size: 1.05rem; border-radius: 12px; }
    .sb-label { font-size: 1.5rem; }
    .ls-head { padding: 14px 16px; }
    .ls-meta { margin-left: 0; width: 100%; }
    .gates { gap: 8px; }
    .gate { font-size: 0.8125rem; padding: 3px 8px; }
    /* alvos de toque maiores nos steppers */
    div[data-testid="stNumberInput"] button { min-width: 2.25rem; }
</style>
""", unsafe_allow_html=True)

PERIOD_MAP = {"1 ano": "1y", "2 anos": "2y", "3 anos": "3y", "5 anos": "5y"}


def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def load_pair(long_in, short_in, period, mean_win, vol_win, corr_win, corr_basis):
    lr = fetch_series(long_in, period)
    sr = fetch_series(short_in, period)
    al = align_pair(lr.series, sr.series)
    return lr, sr, al, core.compute_pair(al.frame, mean_win, vol_win,
                                         corr_win, corr_basis)


st.title("Long / Short")

# --- só o essencial acima do botão -------------------------------------------
if st.session_state.pop("_do_swap", False):
    a = st.session_state.get("m_long", "PETR4")
    b = st.session_state.get("m_short", "VALE3")
    st.session_state["m_long"], st.session_state["m_short"] = b, a

c1, c2 = st.columns(2)
with c1:
    long_in = st.text_input("LONG", value="PETR4", key="m_long",
                            help="Só o ticker — o .SA é automático.")
with c2:
    short_in = st.text_input("SHORT", value="VALE3", key="m_short")

c3, c4 = st.columns([2, 1])
with c3:
    period_label = st.selectbox("Histórico", list(PERIOD_MAP.keys()), index=1)
    period = PERIOD_MAP[period_label]
with c4:
    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if st.button("inverter", width="stretch", help="Inverter LONG e SHORT"):
        st.session_state["_do_swap"] = True
        st.rerun()

run = st.button("Analisar", width="stretch", type="primary")

# --- avançado, fora do caminho -----------------------------------------------
with st.expander("Avançado — janelas e gatilhos"):
    st.caption("Janelas (pregões)")
    j1, j2, j3 = st.columns(3)
    mean_win = j1.number_input("Média", 5, 504, SETUP_VALIDADO["mean_win"])
    vol_win = j2.number_input("Vol", 5, 504, SETUP_VALIDADO["vol_win"])
    corr_win = j3.number_input("Corr", 20, 504, SETUP_VALIDADO["corr_win"])

    st.caption("Monitorar")
    m1, m2, m3 = st.columns(3)
    z_monitor = m1.number_input("|z|>=", 0.1, 3.0, SETUP_VALIDADO["z_monitor"], 0.05)
    dz_monitor = m2.number_input("dz>=", 0.0, 0.5, SETUP_VALIDADO["dz_monitor"], 0.01)
    corr_min = m3.number_input("corr>=", 0.0, 1.0, SETUP_VALIDADO["corr_min"], 0.05)

    st.caption("Alerta máximo")
    a1, a2 = st.columns(2)
    z_alert = a1.number_input("|z|>= ", 0.1, 3.0, SETUP_VALIDADO["z_alert"], 0.05)
    dz_alert = a2.number_input("dz>= ", 0.0, 0.5, SETUP_VALIDADO["dz_alert"], 0.01)

    s1, s2 = st.columns(2)
    z_exit = s1.number_input("Saída |z|<=", 0.0, 1.0, SETUP_VALIDADO["z_exit"], 0.05)
    max_hold = s2.number_input("Máx hold", 5, 252, SETUP_VALIDADO["max_hold"])

    tem_posicao = st.checkbox("Tenho posição aberta neste par", value=False)
    corr_basis = "returns"

with st.expander("Custos e API"):
    cost_rt = st.number_input("Custo round-trip 2 pernas (%)", 0.0, 5.0,
                              core.CUSTO_RT_PADRAO * 100, 0.01) / 100
    rent_aa = st.number_input("Aluguel short (% a.a.)", 0.0, 100.0,
                              core.ALUGUEL_AA_PADRAO * 100, 0.5) / 100
    api_key = st.text_input("Anthropic API Key",
                            value=get_secret("ANTHROPIC_API_KEY", ""),
                            type="password", placeholder="sk-ant-... (opcional)")

cfg = {"z_monitor": z_monitor, "dz_monitor": dz_monitor, "corr_min": corr_min,
       "z_alert": z_alert, "dz_alert": dz_alert, "z_exit": z_exit,
       "max_hold": max_hold, "corr_win": corr_win}

data_key = (long_in.strip(), short_in.strip(), period, mean_win, vol_win,
            corr_win, corr_basis)
if run:
    st.session_state["m_committed"] = data_key
committed = st.session_state.get("m_committed")

if committed is None:
    st.info("Digite o par acima — **só o ticker, sem .SA** — e toque em "
            "**Analisar**.")
    st.stop()

try:
    with st.spinner("Baixando e calculando..."):
        lr, sr, al, ps = load_pair(*committed)
except DataError as err:
    st.error(err.message)
    st.stop()
except ValueError as err:
    st.error(str(err))
    st.stop()

sig = core.classify(ps, cfg, tem_posicao=tem_posicao)
L, S = html.escape(lr.display), html.escape(sr.display)
hold_ref = ps.hl[0] if ps.hl and np.isfinite(ps.hl[0]) else max_hold
z_be = core.breakeven_z(ps.sigma, cost_rt, rent_aa, hold_ref)

if committed != data_key:
    st.warning("Configuração alterada — toque em **Analisar** para atualizar.")

# --- sinal --------------------------------------------------------------------
atraso = int(np.busday_count(ps.last_date.date(), datetime.now().date()))
if atraso <= 0:
    badge = f'<span class="ls-meta">último pregão · {ps.last_date:%d/%m}</span>'
elif atraso == 1:
    badge = f'<span class="ls-meta">fechamento de {ps.last_date:%d/%m} · D-1</span>'
else:
    badge = (f'<span class="ls-meta ls-stale">dado de {ps.last_date:%d/%m} · '
             f'{atraso} pregões atrás</span>')

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
    <span class="chip chip-long">LONG {L}</span>
    <span class="pair-x">x</span>
    <span class="chip chip-short">SHORT {S}</span>
  </div>
  <div style="margin-bottom:8px">{badge}</div>
  <div class="lv-{sig.level} {dir_cls}">
    <div class="sb-label">{sig.label}</div>
    {side_html}
    <div class="sb-desc">{sig.desc}</div>
  </div>
  <div class="gates">{gates_html}</div>
</div>
""", unsafe_allow_html=True)

for note in [n for n in (lr.resolution_note, sr.resolution_note) if n]:
    st.caption(note)
for w in lr.warnings + sr.warnings + ps.problems:
    st.warning(w)
if al.pct_dropped >= 2:
    st.warning(f"{al.n_dropped} pregões descartados ({al.pct_dropped:.1f}%) "
               f"por calendários diferentes.")

# checklist — é justamente a tela que você quer no celular quando dispara
if sig.level == "alert":
    st.markdown("**Checklist antes de executar:**")
    k = f"{lr.symbol}_{sr.symbol}"
    st.checkbox("Liquidez OK", key=f"m_liq_{k}")
    st.checkbox("Spread de tela OK", key=f"m_spr_{k}")
    st.checkbox("Aluguel checado", key=f"m_alu_{k}")
    st.checkbox("Sem evento societário", key=f"m_evt_{k}")

# --- métricas em 2 colunas ----------------------------------------------------
m1, m2 = st.columns(2)
m1.markdown(metric_card("Z-Score", f"{ps.z_now:+.3f}", "", z_tone(ps.z_now, cfg)),
            unsafe_allow_html=True)
mom = {True: "esticando", False: "revertendo", None: "parado"}[sig.esticando]
m2.markdown(metric_card("Delta z · 1 pregão", f"{ps.dz_now:+.3f}",
                        f"{mom} · ruído {ps.dz_noise:.3f}", "accent"),
            unsafe_allow_html=True)

if ps.hl is None:
    hl_v, hl_s, hl_t = "n/d", "amostra insuficiente", "na"
elif not np.isfinite(ps.hl[0]):
    hl_v, hl_s, hl_t = "infinita", "não reverte", "weak"
else:
    _hl, _lo, _hi = ps.hl
    _hi_txt = "inf" if not np.isfinite(_hi) else f"{_hi:.0f}"
    hl_v, hl_s = f"{_hl:.0f}p", f"IC {_lo:.0f}-{_hi_txt}p"
    hl_t = "ok" if _hl <= max_hold else "weak"

edge = abs(ps.z_now) - z_exit - z_be
m3, m4 = st.columns(2)
m3.markdown(metric_card("Meia-vida", hl_v, hl_s, hl_t), unsafe_allow_html=True)
m4.markdown(metric_card("Edge líquido", f"{edge:+.2f} z",
                        f"break-even {z_be:.2f} z",
                        "long" if edge > 0.30 else "short" if edge < 0 else "weak"),
            unsafe_allow_html=True)

m5, m6 = st.columns(2)
if ps.corr_now is None:
    m5.markdown(metric_card(f"Corr {corr_win}p", "n/d", "janela > histórico", "na"),
                unsafe_allow_html=True)
else:
    m5.markdown(metric_card(f"Corr {corr_win}p", f"{ps.corr_now:.3f}",
                            f"retornos · >= {corr_min:.2f}",
                            "ok" if ps.corr_now >= corr_min else "weak"),
                unsafe_allow_html=True)


def pv_card(label, p, ok_txt, bad_txt):
    if p is None:
        return metric_card(label, "n/d", "teste falhou", "na")
    return metric_card(label, f"{p:.3f}", ok_txt if p < 0.05 else bad_txt,
                       "ok" if p < 0.05 else "weak")


m6.markdown(pv_card("Cointegração p", ps.coint_p, "cointegrado", "fraco"),
            unsafe_allow_html=True)
st.markdown(pv_card("ADF do spread p", ps.adf_p, "estacionário",
                    "não estacionário"), unsafe_allow_html=True)

if z_be > z_exit:
    st.error(f"A saída em |z| <= {z_exit:.2f} não cobre o custo "
             f"(break-even {z_be:.2f} z).")

st.divider()

# --- gráficos empilhados ------------------------------------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                    subplot_titles=[f"Z-Score ({mean_win}p/{vol_win}p)",
                                    f"Spread ln({L}) - ln({S})"],
                    vertical_spacing=0.10)

fig.add_trace(go.Bar(x=ps.z.index, y=ps.z, name="Z",
                     marker_color=z_bar_colors(ps.z, cfg)), row=1, col=1)
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
fig.update_yaxes(range=[-top, top], row=1, col=1)

roll_m = ps.spread.rolling(mean_win).mean()
roll_s = ps.spread.rolling(vol_win).std()
fig.add_trace(go.Scatter(x=ps.spread.index, y=ps.spread, name="Spread",
                         line=dict(color=C_ACCENT, width=1.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m, name="Média",
                         line=dict(color=C_GRAPHIC, width=1)), row=2, col=1)
for mult, color in ((z_alert, C_SHORT), (-z_alert, C_LONG)):
    fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m + mult * roll_s,
                             showlegend=False, hoverinfo="skip",
                             line=dict(color=color, width=1, dash="dot")),
                  row=2, col=1)

style_fig(fig, height=460, top_margin=50, legend_y=1.06)
fig.update_annotations(font=dict(size=12, color=C_MUTED))
st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

# --- Claude -------------------------------------------------------------------
st.divider()
st.subheader("Análise Claude")
if not api_key:
    st.caption("Adicione a API Key em **Custos e API** para ativar.")
else:
    akey = f"mia_{lr.symbol}_{sr.symbol}_{hash(frozenset(cfg.items()))}"
    if st.button("Gerar análise", width="stretch"):
        with st.spinner("Claude analisando..."):
            try:
                st.session_state[akey] = core.analyze_with_claude(
                    api_key, L, S, ps, sig, cfg, z_be, conciso=True)
                st.session_state[akey + "_ts"] = datetime.now()
            except Exception as e:
                st.error(f"Erro na chamada à API: {e}")
    if akey in st.session_state:
        ts = st.session_state.get(akey + "_ts")
        if ts:
            st.caption(f"Gerada em {ts:%d/%m %H:%M}")
        st.markdown(st.session_state[akey])

st.caption(f"Yahoo Finance · último pregão {ps.last_date:%d/%m/%Y}")
