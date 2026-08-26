"""LS Dashboard — versão desktop.

Toda a lógica quantitativa vive em core.py; os dados em datasource.py; o
visual em theme.py. Este arquivo é só a camada de apresentação — foi assim
que app.py e mobile.py pararam de divergir.
"""

import html
import os
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
    """Segredo do servidor: st.secrets (secrets.toml) ou variável de ambiente.

    Nunca quebra o app se não houver secrets.toml, e cobre o caso do token
    chegar via `environment:` do docker-compose (arquivo .env na VPS).
    """
    try:
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


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
    # Defaults vivem no session_state (nao em value=): assim os widgets abaixo
    # usam so a key, sem o warning "created with a default value but also had
    # its value set via the Session State API".
    st.session_state.setdefault("in_long", "PETR4")
    st.session_state.setdefault("in_short", "VALE3")

    # A troca precisa acontecer ANTES dos widgets existirem: o Streamlit
    # proíbe reescrever a chave de um widget já instanciado no mesmo run.
    if st.session_state.pop("_do_swap", False):
        a = st.session_state["in_long"]
        b = st.session_state["in_short"]
        st.session_state["in_long"], st.session_state["in_short"] = b, a

    c1, c2 = st.columns(2)
    with c1:
        long_in = st.text_input("LONG", key="in_long",
                                help="Digite só o ticker: petr4, vale3, bova11, "
                                     "aapl, ibov. O sufixo .SA é automático.")
    with c2:
        short_in = st.text_input("SHORT", key="in_short")

    if st.button("⇄ inverter par", width="stretch"):
        st.session_state["_do_swap"] = True
        st.rerun()

    period_label = st.selectbox("Histórico", list(PERIOD_MAP.keys()), index=2)
    period = PERIOD_MAP[period_label]

    run_btn = st.button("▶ Analisar", width="stretch", type="primary")
    st.divider()

    # Posicao fica JUNTO do par, nao enterrada num expander: ela troca o texto
    # do sinal principal entre "CONVERGÊNCIA · encerrar" e "SEM EDGE".
    # Chave por par — senao o check vaza para um par que voce nao tem.
    tem_posicao = st.checkbox(
        "Tenho posição aberta neste par", value=False,
        key=f"pos_{long_in.strip().upper()}_{short_in.strip().upper()}",
        help="Sem posição, 'CONVERGÊNCIA -> encerrar' não faz sentido — o "
             "painel mostra 'SEM EDGE' em vez disso.")
    st.divider()

    # Rotulos mostram o valor CORRENTE (via session_state), nao o validado:
    # com 5 caixas fechadas que mudam o significado de tudo na tela, um rotulo
    # que sempre diz "63/63/252" e pior que nenhum rotulo.
    def _v(key, default):
        return st.session_state.get(key, default)

    # --- janelas -------------------------------------------------------------
    with st.expander(f"Janelas · {_v('sb_mean', SETUP_VALIDADO['mean_win'])}/"
                     f"{_v('sb_vol', SETUP_VALIDADO['vol_win'])}/"
                     f"{_v('sb_corr', SETUP_VALIDADO['corr_win'])}",
                     expanded=False):
        mean_win = st.number_input("Média do spread (pregões)", 5, 504,
                                   SETUP_VALIDADO["mean_win"], key="sb_mean")
        vol_win = st.number_input("Volatilidade (pregões)", 5, 504,
                                  SETUP_VALIDADO["vol_win"], key="sb_vol")
        corr_win = st.number_input("Correlação (pregões)", 20, 504,
                                   SETUP_VALIDADO["corr_win"], key="sb_corr")
        corr_basis = st.radio(
            "Base da correlação",
            ["returns", "levels"], index=0, horizontal=True, key="sb_basis",
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
    with st.expander(f"Gatilhos · {_v('sb_zmon', SETUP_VALIDADO['z_monitor']):.2f}"
                     f" / {_v('sb_zalert', SETUP_VALIDADO['z_alert']):.2f}",
                     expanded=False):
        st.caption("Monitorar")
        z_monitor = st.slider("|z| >=", 0.5, 3.0, SETUP_VALIDADO["z_monitor"],
                              0.05, key="sb_zmon")
        dz_monitor = st.slider("d|z| (1 pregão) >=", 0.0, 0.5,
                               SETUP_VALIDADO["dz_monitor"], 0.01, key="sb_dzmon")
        corr_min = st.slider("Correlação >=", 0.0, 1.0,
                             SETUP_VALIDADO["corr_min"], 0.05, key="sb_corrmin")
        st.caption("Alerta máximo")
        z_alert = st.slider("|z| >= ", 0.5, 3.0, SETUP_VALIDADO["z_alert"],
                            0.05, key="sb_zalert")
        dz_alert = st.slider("d|z| (1 pregão) >= ", 0.0, 0.5,
                             SETUP_VALIDADO["dz_alert"], 0.01, key="sb_dzalert")

    with st.expander(f"Saída · |z|<={_v('sb_zexit', SETUP_VALIDADO['z_exit']):.2f}"
                     f" · {_v('sb_hold', SETUP_VALIDADO['max_hold'])}p",
                     expanded=False):
        z_exit = st.slider("Convergência |z| <=", 0.0, 1.0,
                           SETUP_VALIDADO["z_exit"], 0.05, key="sb_zexit")
        max_hold = st.number_input("Máx. holding (pregões)", 5, 252,
                                   SETUP_VALIDADO["max_hold"], key="sb_hold")

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

    # SEGURANÇA: segredos configurados no servidor NUNCA são enviados ao
    # navegador (nem mascarados num value= — daria pra extrair via DevTools
    # num site público). A tela só informa que estão configurados; o valor
    # real fica no backend.
    server_api_key = get_secret("ANTHROPIC_API_KEY", "")
    server_brapi = get_secret("BRAPI_TOKEN", "")

    with st.expander("Conexões", expanded=False):
        if server_api_key:
            st.caption("Claude AI · ✅ chave configurada no servidor")
            api_key = server_api_key
        else:
            api_key = st.text_input("Anthropic API Key",
                                    type="password", placeholder="sk-ant-...").strip()

        if server_brapi:
            st.caption("brapi.dev · ✅ token configurado no servidor")
            st.session_state["_brapi_token_ui"] = ""
        else:
            brapi_token = st.text_input(
                "brapi.dev Token (PRO)",
                type="password", placeholder="token do brapi.dev",
                help="Dados de preço da B3. Vale só nesta sessão do navegador; "
                     "para fixar de vez, configure no servidor (arquivo .env).")
            # Disponibiliza o token colado pro datasource.
            st.session_state["_brapi_token_ui"] = brapi_token.strip()

    st.divider()
    auto_refresh = st.toggle("Auto-refresh no pregão (60s)", value=False)

cfg = {"z_monitor": z_monitor, "dz_monitor": dz_monitor, "corr_min": corr_min,
       "z_alert": z_alert, "dz_alert": dz_alert, "z_exit": z_exit,
       "max_hold": max_hold, "corr_win": corr_win}

# Gatilhos fora de ordem quebram a maquina de estados em silencio:
# z_monitor > z_alert faz ALERTA disparar antes de MONITORAR; z_exit acima de
# z_monitor faz o ramo de saida engolir tudo e o painel dizer "SEM EDGE" sempre.
ordem = []
if not (z_exit < z_monitor):
    ordem.append(f"saída ({z_exit:.2f}) tem que ser MENOR que monitorar ({z_monitor:.2f})")
if not (z_monitor <= z_alert):
    ordem.append(f"monitorar ({z_monitor:.2f}) não pode passar de alerta ({z_alert:.2f})")
if not (dz_monitor <= dz_alert):
    ordem.append(f"d|z| de monitorar ({dz_monitor:.2f}) não pode passar do de "
                 f"alerta ({dz_alert:.2f})")
if ordem:
    st.error("**Gatilhos fora de ordem** — corrija na barra lateral:\n\n"
             + "\n".join(f"- {o}" for o in ordem))
    st.stop()

# Aviso de drift: você está no setup validado ou em algo que mexeu semana passada?
_fmt = {"z_monitor": "monitorar |z|", "dz_monitor": "monitorar d|z|",
        "corr_min": "corr mín", "z_alert": "alerta |z|", "dz_alert": "alerta d|z|",
        "z_exit": "saída |z|", "max_hold": "máx holding"}
drift = [f"{_fmt[k]} {cfg[k]:g} (validado {v:g})"
         for k, v in SETUP_VALIDADO.items()
         if k in cfg and k in _fmt and abs(cfg[k] - v) > 1e-9]
for nome, atual, val in (("média", mean_win, SETUP_VALIDADO["mean_win"]),
                         ("vol", vol_win, SETUP_VALIDADO["vol_win"]),
                         ("corr", corr_win, SETUP_VALIDADO["corr_win"])):
    if atual != val:
        drift.append(f"janela de {nome} {atual} (validado {val})")
if corr_basis != "returns":
    drift.append("correlação em níveis (validado: log-retornos)")

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

# A PARTIR DAQUI usa-se sempre o parametro EFETIVO do calculo, nunca o widget.
# Em estado stale os dois divergem, e o painel desenhava bandas de uma janela
# sobre barras de z de outra, com o titulo mentindo a janela.
mean_win, vol_win = ps.mean_win, ps.vol_win
corr_win, corr_basis = ps.corr_win, ps.corr_basis
cfg["corr_win"] = ps.corr_win

sig = core.classify(ps, cfg, tem_posicao=tem_posicao)
L, S = html.escape(lr.display), html.escape(sr.display)

# O aluguel so corre ate o holding maximo: um par com meia-vida de 180p que
# voce fecha em 63p nao paga 180 pregoes de BTC. Sem o teto, z_be inflava ~3x.
hold_ref = min(ps.hl[0], max_hold) if ps.hl and np.isfinite(ps.hl[0]) else max_hold
z_be = core.breakeven_z(ps.sigma, cost_rt, rent_aa, hold_ref)
# O edge é economicamente um quarto gate: "esse sinal paga a conta?".
edge = abs(ps.z_now) - z_exit - z_be

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
# 4º chip: o edge líquido. Sem ele a tríade responde "o par esticou?" mas não
# "vale a pena?", que é a pergunta que decide o trade.
gates_html += (
    f'<span class="gate {"ok" if edge > 0 else "fail"}">'
    f'<span class="g-mark">{"OK" if edge > 0 else "X"}</span> '
    f'edge <b>{edge:+.2f} z</b> '
    f'<span class="g-thr">custo {z_be:.2f}</span></span>')

# O aviso de stale vem ANTES do sinal: ler "ALERTA MÁXIMO" em 28px e só depois
# descobrir que é do par anterior é pior do que não mostrar nada.
if stale:
    st.warning("A configuração mudou desde o último cálculo. Clique em "
               "**Analisar** — o painel abaixo ainda é do par anterior.")

st.markdown(f"""
<div class="ls-head{' is-stale' if stale else ''}">
  <div class="ls-pair">
    <span class="chip chip-long">LONG · {L}</span>
    <span class="pair-x">x</span>
    <span class="chip chip-short">SHORT · {S}</span>
  </div>
  {badge}
  <div class="lv-{sig.level} {dir_cls}">
    <div class="sb-label">{sig.label}</div>
    {side_html}
    <div class="sb-desc">{sig.desc}</div>
  </div>
  <div class="gates">{gates_html}</div>
</div>
""", unsafe_allow_html=True)

# Avisos de qualidade de dado consolidados: empilhados inline eles somavam até
# 8 blocos de largura total entre o sinal e o gráfico, desfazendo a subida dele.
notas = [n for n in (lr.resolution_note, sr.resolution_note) if n]
alertas = list(lr.warnings) + list(sr.warnings) + list(ps.problems)
if al.pct_dropped >= 2:
    alertas.append(
        f"Calendários diferentes: {al.n_dropped} pregões descartados "
        f"({al.pct_dropped:.1f}% da amostra) por falta de negócio em um dos "
        f"ativos. Estatísticas calculadas sobre {al.n_used} pregões.")
if alertas:
    with st.expander(f"Qualidade do dado · {len(alertas)} aviso"
                     f"{'s' if len(alertas) > 1 else ''}", expanded=False):
        for w in alertas:
            st.warning(w)
        for n in notas:
            st.caption(n)
elif notas:
    st.caption(" · ".join(notas))

if drift:
    d1, d2 = st.columns([4, 1])
    d1.info("Fora do setup validado do backtest: " + " · ".join(drift))
    if d2.button("Restaurar", width="stretch",
                 help="Volta todos os parâmetros ao setup validado"):
        for k in ("sb_mean", "sb_vol", "sb_corr", "sb_zmon", "sb_dzmon",
                  "sb_corrmin", "sb_zalert", "sb_dzalert", "sb_zexit", "sb_hold"):
            st.session_state.pop(k, None)
        st.rerun()

# checklist por par — chaves com o ticker, senão vaza entre pares
if sig.level == "alert":
    st.markdown("**Checklist do Alerta Máximo — validar antes de executar:**")
    k = f"{lr.symbol}_{sr.symbol}"
    q1, q2, q3, q4 = st.columns(4)
    c_ok = [
        q1.checkbox("Liquidez OK", key=f"chk_liq_{k}"),
        q2.checkbox("Spread de tela OK", key=f"chk_spr_{k}"),
        q3.checkbox("Aluguel checado", key=f"chk_alu_{k}"),
        q4.checkbox("Sem evento societário", key=f"chk_evt_{k}"),
    ]
    # Fecha o loop: marcar as 4 caixas entrega a ordem pronta, que é o próximo
    # passo real. Antes o checklist não produzia nada.
    if all(c_ok):
        _sz = core.sizing(ps.frame, capital)
        _lado = "vender" if sig.side == "short_ratio" else "comprar"
        _lado_s = "comprar" if sig.side == "short_ratio" else "vender"
        st.success(
            f"**Checklist completo — ordem para R$ {capital:,.0f} por perna:**  "
            f"{_lado} **{_sz['q_long']}** {L} a ~R$ {_sz['p_long']:.2f}  ·  "
            f"{_lado_s} **{_sz['q_short']}** {S} a ~R$ {_sz['p_short']:.2f}  ·  "
            f"resíduo de lote {_sz['residuo_pct']:+.2%}".replace(",", "."))


# =============================================================================
# Gráficos — acima das estatísticas: primeiro se enxerga, depois se confere
# =============================================================================
# Ordem interna: Z-Score primeiro. Antes ele nascia abaixo da dobra.
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, row_heights=[0.38, 0.30, 0.32],
    subplot_titles=[f"Z-Score ({mean_win}p / {vol_win}p)",
                    f"Spread ln({L}) - ln({S}) · média {mean_win}p",
                    "Preços normalizados (base 100)"],
    vertical_spacing=0.07)

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

# top_margin=104 + legend_y=1.06: com 86/1.03 a legenda (29px de altura) e o
# título do subplot 1 (18px) se encostavam — é o bug antigo de sobreposição
# legenda × título, em versão atenuada. Agora sobram ~11px entre eles.
style_fig(fig, height=600, top_margin=104, legend_y=1.06)
fig.update_annotations(font=dict(size=13, color=C_MUTED))
st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

st.caption("Faixas: vermelha = zona de venda do ratio · verde = zona de compra "
           "· roxa = banda de saída. Tracejado = monitorar, pontilhado = alerta.")

st.markdown("")


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
    "d|z| · 1 pregão", f"{ps.dz_now:+.3f}",
    f"{mom} · ruído de fundo {ps.dz_noise:.3f}", "neutral"), unsafe_allow_html=True)

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
    # Sem correlação o gate nunca passa e NENHUM sinal pode disparar — o painel
    # dizia calmamente "NEUTRO · aguardar", que lê como "sem oportunidade".
    st.warning(
        f"**Nenhum sinal pode disparar:** a janela de correlação "
        f"({corr_win} pregões) é maior que o histórico disponível "
        f"({ps.n_obs} pregões), então o gate de correlação nunca passa. "
        f"Aumente o **Histórico** na barra lateral ou reduza a **janela de "
        f"correlação** em Janelas.")
    cv, cs, ct = "n/d", f"janela {corr_win}p > {ps.n_obs}p disponíveis", "na"
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
custo_total = cost_rt + rent_aa * hold_ref / 252
st.caption(
    f"**Edge líquido {edge:+.2f} z** — amplitude de |z| {abs(ps.z_now):.2f} até "
    f"a saída em {z_exit:.2f}, contra um break-even de {z_be:.2f} z "
    f"(custo total {custo_total * 100:.2f}% do notional em "
    f"{hold_ref:.0f} pregões)."
    + (f"  ⚠️ Como o break-even ({z_be:.2f}) é maior que a banda de saída "
       f"({z_exit:.2f}), fechar em |z| ≤ {z_exit:.2f} sai no zero a zero — "
       f"revise custos ou a regra de saída."
       if z_be > z_exit else ""))

# Só vira alarme quando há um sinal de verdade para atrapalhar. Antes disparava
# em todo par e todo dia, inclusive sem sinal — o jeito clássico de treinar
# o usuário a ignorar vermelho.
if sig.level in ("alert", "watch") and edge < 0:
    st.error(f"**Sinal sem edge.** O custo ({z_be:.2f} z) consome toda a "
             f"amplitude disponível até a saída. Não opere este par nesta "
             f"configuração de custo.")

st.markdown("")

# =============================================================================
# Abas — detalhe sob demanda
# =============================================================================
tab_e, tab_s, tab_ia, tab_m = st.tabs(
    ["Estatísticas", "Sizing", "Análise IA", "Manual"])

with tab_e:
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Estatísticas do spread canônico**")
        st.dataframe(pd.DataFrame({
            "Métrica": ["Spread atual", f"Média ({mean_win}p)", "sigma do spread",
                        "Z-Score", "d|z| (1 pregão)", "Z máx", "Z mín",
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
        f"{sz['residuo_pct']:+.2%} do capital",
        "weak" if abs(sz["residuo_pct"]) > 0.02 else "pass"), unsafe_allow_html=True)

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
        st.info("Adicione sua Anthropic API Key no expander **Conexões** da barra "
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

**d|z|** — variação do **módulo** do z em 1 pregão: `|z_t| − |z_t-1|`. O gate é
**unilateral** (`>= {dz_monitor:.2f}`), então só passa quando o par está
*esticando*. Isso é fiel à família I do backtest — não é `|Δz|`, que aceitaria
também um par revertendo, trade economicamente oposto. Atenção: d|z| mistura o
movimento de hoje com o efeito da janela *descartando* a observação de
{mean_win} pregões atrás. O card mostra o **ruído de fundo**
({ps.dz_noise:.3f} neste par) — um gatilho abaixo desse valor é ruído, não
momentum.

**Meia-vida de reversão** — quanto tempo o spread leva para percorrer metade do
caminho de volta à média. É o número que responde *quanto tempo meu capital
fica preso*. Leia o **intervalo de confiança**, não o ponto: o estimador é
enviesado para baixo em meias-vidas longas.

**Correlação ({corr_win}p)** — sobre **log-retornos**, não sobre níveis de
preço, exatamente como no backtest (`corr(dln L, dln S)`). Correlação entre
séries de preço em nível é espúria: mede tendência comum. Medido em
PETR4×VALE3: 0,77 em níveis contra −0,08 em retornos. O gatilho de
{SETUP_VALIDADO['corr_min']:.2f} é restritivo de propósito.

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

- **Monitorar** — `|z| >= {z_monitor:.2f}` · `d|z| >= {dz_monitor:.2f}` ·
  `corr >= {corr_min:.2f}`
- **Alerta máximo** — `|z| >= {z_alert:.2f}` · `d|z| >= {dz_alert:.2f}` ·
  `corr >= {corr_min:.2f}` + checklist
- **Saída** — `|z| <= {z_exit:.2f}` ou {max_hold} pregões de holding. Sem stop
  e sem saída por cruzamento do zero — igual ao backtest.

Quando o sinal é NEUTRO, a tríade de check no topo mostra **qual condição
falhou** — não só que o conjunto não passou.

Conferido linha a linha contra `run_jarvis_pair()` do `research_ls_b3`
(família I), que foi a configuração levada ao teste cego de 2024+.

### O que o painel ainda não modela

O backtest tem duas regras que este painel não implementa:

1. **Exclusão por perdas** — 3 trades com PnL líquido negativo consecutivos no
   mesmo par bloqueiam o par; a reabilitação exige 2 convergências virtuais
   consecutivas. Isso exige histórico de trades por par.
2. **Elegibilidade point-in-time** — ADTV de 60 dias >= R$ 1M nas duas pernas
   no dia do sinal, com tiers de slippage por faixa de ADTV (10/20/40 bps).
   Aqui o custo é um parâmetro único que você informa.

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
