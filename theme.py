"""Design system compartilhado — desktop e mobile importam daqui.

Regra de ouro da paleta (de 9 matizes para 4):

    verde / vermelho  = DIREÇÃO (long / short) e nada mais
    roxo              = o ratio e o cromo da UI
    azul / cinza      = qualidade estatística (corr, cointegração, ADF)
    intensidade       = OPACIDADE + espessura de borda, JAMAIS um matiz novo

Isso elimina laranja, teal, ciano e amarelo da escala de sinal — e resolve de
uma vez o "arco-íris", a divergência desktop↔mobile (onde as cores do z-score
significavam coisas OPOSTAS) e boa parte do problema de daltonismo.

Contrastes verificados contra --surface-raised (#232334):
    --text-strong  10,7:1     --text-muted  6,9:1
    --text-subtle   5,5:1  (substitui #7f849c, que reprovava AA com 4,2:1)
    --text-graphic  3,4:1  (mínimo para objetos gráficos, WCAG 1.4.11)
"""

# --- superfícies (4, não 7) --------------------------------------------------
C_BASE   = "#1e1e2e"
C_RAISED = "#232334"
C_SUNKEN = "#181825"
C_LINE   = "#313244"

# --- texto -------------------------------------------------------------------
C_TEXT    = "#cdd6f4"
C_MUTED   = "#a6adc8"
C_SUBTLE  = "#9399b2"
C_GRAPHIC = "#6c7086"

# --- direção (2 matizes, só isso) --------------------------------------------
# Verde escurecido de #a6e3a1 para #94d68f: o original tinha 11,0:1 contra
# 7,1:1 do vermelho, e o alerta de venda parecia ~55% menos luminoso.
C_LONG  = "#94d68f"
C_SHORT = "#f38ba8"
C_FLAT  = "#45475a"

# --- cromo e estatística -----------------------------------------------------
C_ACCENT    = "#cba6f7"
C_STAT_OK   = "#89b4fa"
C_STAT_WEAK = "#9399b2"

LVL_WATCH = 0.45      # intensidade "monitorar"
LVL_ALERT = 1.00      # intensidade "alerta máximo"


def rgba(hex_color: str, alpha: float) -> str:
    """#rrggbb + alfa -> 'rgba(r,g,b,a)'. Como a intensidade vira opacidade."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


CSS = f"""
<style>
    /* O header do Streamlit (stHeader) e FIXO com sizes.headerHeight = 3.75rem
       (60px) e carrega o menu e o botao Deploy no canto superior direito.
       Reduzir o padding joga o conteudo POR BAIXO dele e o texto aparece
       cortado. 4.5rem e o valor que o proprio Streamlit usa no modo compacto
       com header; 3.5rem ainda passava 4px por baixo.
       A formula do padding nao olha `layout`, entao wide e centered usam o
       mesmo numero — desktop e mobile ficam iguais. */
    .block-container {{ padding-top: 4.5rem; padding-bottom: 2.5rem; }}
    /* Fundo solido no header para nada aparecer por baixo dele ao rolar. */
    header[data-testid="stHeader"] {{ background: {C_BASE}; }}
    h1, h2, h3 {{ letter-spacing: -0.02em; }}

    /* ---------- escala tipográfica: 6 degraus, não 13 ---------- */
    /* 0.75 / 0.875 / 1 / 1.25 / 1.75 / 2.25 rem */

    /* ---------- cabeçalho + sinal, fundidos num bloco só ---------- */
    .ls-head {{
        border: 1px solid {C_LINE}; border-radius: 14px;
        padding: 16px 20px; margin-bottom: 14px;
        background: {rgba(C_LINE, 0.35)};
        box-shadow: 0 4px 14px rgba(0,0,0,0.22);
    }}
    /* Config alterada e ainda nao recalculada: o bloco inteiro esmaece, para
       nao se ler um "ALERTA MAXIMO" que e do par anterior. */
    .ls-head.is-stale {{ opacity: 0.5; filter: saturate(0.5); }}
    .ls-pair {{
        display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
        margin-bottom: 10px;
    }}
    .chip {{
        display: inline-block; padding: 5px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.875rem; white-space: nowrap;
    }}
    .chip-long  {{ background: {rgba(C_LONG, 0.10)};  color: {C_LONG};
                   border: 1px solid {rgba(C_LONG, 0.45)}; }}
    .chip-short {{ background: {rgba(C_SHORT, 0.10)}; color: {C_SHORT};
                   border: 1px solid {rgba(C_SHORT, 0.45)}; }}
    /* C_GRAPHIC e token de objeto grafico (3:1), nao de glifo. */
    .pair-x   {{ color: {C_SUBTLE}; font-weight: 800; font-size: 1rem; }}
    /* Em linha propria, alinhado a esquerda. Com margin-left:auto ele ia para
       o canto superior DIREITO — debaixo dos botoes do header do Streamlit. */
    .ls-meta  {{ display: block; color: {C_MUTED}; font-size: 0.875rem;
                 margin: 0 0 10px; }}
    .ls-stale {{ color: {C_SHORT}; font-weight: 700; }}

    /* ---------- sinal: a informação mais importante, com o maior peso ------ */
    .sb-label {{ font-size: 1.75rem; font-weight: 800; line-height: 1.15;
                 color: {C_TEXT}; }}
    .sb-side  {{ font-size: 1rem; color: {C_MUTED}; margin-top: 2px; }}
    /* C_SUBTLE sobre o fundo composto do .lv-alert (#423346) da 4,17:1 e
       reprova AA em 14px. C_MUTED da 5,30:1. */
    .sb-desc  {{ font-size: 0.875rem; color: {C_MUTED}; margin-top: 4px; }}

    /* nível codificado por borda + fundo, não só pelo emoji dentro do texto */
    /* padding vertical: so com padding-left o fundo colorido cola no texto. */
    .lv-alert {{ border-left: 5px solid {C_SHORT}; padding: 10px 14px;
                 background: {rgba(C_SHORT, 0.14)}; border-radius: 0 10px 10px 0; }}
    .lv-alert.dir-long {{ border-left-color: {C_LONG};
                          background: {rgba(C_LONG, 0.14)}; }}
    .lv-alert .sb-label {{ color: {C_SHORT}; }}
    .lv-alert.dir-long .sb-label {{ color: {C_LONG}; }}

    .lv-watch {{ border-left: 3px solid {rgba(C_SHORT, 0.55)}; padding: 10px 14px;
                 background: {rgba(C_SHORT, 0.06)}; border-radius: 0 10px 10px 0; }}
    .lv-watch.dir-long {{ border-left-color: {rgba(C_LONG, 0.55)};
                          background: {rgba(C_LONG, 0.06)}; }}
    .lv-watch .sb-label {{ color: {C_SHORT}; }}
    .lv-watch.dir-long .sb-label {{ color: {C_LONG}; }}

    .lv-exit  {{ border-left: 3px solid {C_ACCENT}; padding: 10px 14px;
                 background: {rgba(C_ACCENT, 0.08)}; border-radius: 0 10px 10px 0; }}
    .lv-exit .sb-label {{ color: {C_ACCENT}; }}

    .lv-flat  {{ border-left: 3px solid {C_FLAT}; padding: 10px 14px;
                 background: {rgba(C_FLAT, 0.20)}; border-radius: 0 10px 10px 0; }}
    .lv-flat .sb-label {{ color: {C_MUTED}; }}

    /* ---------- tríade de check ---------- */
    .gates {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }}
    .gate {{
        font-size: 0.875rem; font-variant-numeric: tabular-nums;
        padding: 4px 10px; border-radius: 8px; border: 1px solid {C_LINE};
        background: {rgba(C_SUNKEN, 0.55)};
    }}
    .gate b {{ color: {C_TEXT}; }}
    .gate .g-thr {{ color: {C_SUBTLE}; }}
    .gate.ok   {{ border-color: {rgba(C_STAT_OK, 0.45)}; }}
    .gate.ok   .g-mark {{ color: {C_STAT_OK}; font-weight: 800; }}
    .gate.fail {{ border-color: {rgba(C_SHORT, 0.55)}; background: {rgba(C_SHORT, 0.07)}; }}
    .gate.fail .g-mark {{ color: {C_SHORT}; font-weight: 800; }}

    /* ---------- cards de métrica ---------- */
    .metric-card {{
        background: {C_RAISED}; border: 1px solid {C_LINE};
        border-left: 3px solid {C_FLAT};
        border-radius: 12px; padding: 12px 14px; height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }}
    /* height:100% no card so funciona se a cadeia de wrappers do Streamlit
       tambem esticar — stColumn estica, mas os divs internos sao `auto`,
       e height:100% contra pai auto computa auto. Sem isso os cards ficam
       com fundos de altura irregular. */
    div[data-testid="stColumn"] > div,
    div[data-testid="stColumn"] div[data-testid="stVerticalBlock"],
    div[data-testid="stColumn"] div[data-testid="stElementContainer"]:has(.metric-card),
    div[data-testid="stColumn"] div[data-testid="stMarkdown"]:has(.metric-card),
    div[data-testid="stColumn"] div[data-testid="stMarkdownContainer"]:has(.metric-card)
        {{ height: 100%; }}

    /* min-height e relativo ao proprio font-size: 2.1em de 12px = 25,2px, mas
       uma linha com line-height 1.6 ja mede 19,2px e duas medem 38,4px — o
       rotulo de 2 linhas estourava 13px. line-height fixo resolve. */
    .metric-label {{
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: {C_MUTED};
        white-space: normal; line-height: 1.25; min-height: 2.5em;
    }}
    .metric-value {{
        font-size: 1.25rem; font-weight: 700; color: {C_TEXT};
        margin-top: 2px; font-variant-numeric: tabular-nums;
    }}
    .metric-sub {{
        font-size: 0.75rem; color: {C_SUBTLE}; margin-top: 2px;
        line-height: 1.35; min-height: 1.35em; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis;
    }}
    .metric-card.tone-long  {{ border-left-color: {C_LONG}; }}
    .metric-card.tone-long  .metric-value {{ color: {C_LONG}; }}
    .metric-card.tone-short {{ border-left-color: {C_SHORT}; }}
    .metric-card.tone-short .metric-value {{ color: {C_SHORT}; }}
    .metric-card.tone-accent {{ border-left-color: {C_ACCENT}; }}
    .metric-card.tone-ok    {{ border-left-color: {C_STAT_OK}; }}
    .metric-card.tone-weak  {{ border-left-color: {C_STAT_WEAK}; }}
    /* "aprovado" que NAO e direcao: verde ja significa long nesta tela. */
    .metric-card.tone-pass  {{ border-left-color: {C_STAT_OK}; }}
    .metric-card.tone-pass  .metric-value {{ color: {C_STAT_OK}; }}
    .metric-card.tone-na    {{ border-left-color: {C_GRAPHIC}; }}
    .metric-card.tone-na    .metric-value {{ color: {C_SUBTLE}; }}

    /* ---------- estado vazio ---------- */
    .empty-state {{
        border: 1px dashed {C_FLAT}; border-radius: 14px;
        padding: 40px 24px; text-align: center;
        color: {C_MUTED}; background: {rgba(C_LINE, 0.25)}; font-size: 0.9375rem;
    }}
    .empty-state b {{ color: {C_ACCENT}; }}

    /* ---------- widgets do streamlit ---------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {C_LINE}; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px 9px 0 0; padding: 8px 16px;
        color: {C_MUTED}; font-weight: 600; background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {C_ACCENT} !important; background: {rgba(C_ACCENT, 0.08)};
    }}
    /* overflow visible: com hidden o balao de valor do slider era cortado, e
       todos os sliders de gatilho vivem dentro de expanders. */
    div[data-testid="stExpander"] {{
        border: 1px solid {C_LINE}; border-radius: 10px;
        background: {rgba(C_LINE, 0.25)}; overflow: visible;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {C_LINE}; border-radius: 12px; overflow: hidden;
    }}
    section[data-testid="stSidebar"] {{
        background: {C_SUNKEN}; border-right: 1px solid {C_LINE};
    }}
    section[data-testid="stSidebar"] hr {{ margin: 0.6rem 0; }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {C_ACCENT} 0%, {C_STAT_OK} 100%);
        color: {C_SUNKEN}; font-weight: 700; border: none; border-radius: 10px;
    }}
    .stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); }}
    footer {{ visibility: hidden; }}
</style>
"""

PLOTLY_CONFIG = {"displayModeBar": False}


# =============================================================================
# Componentes
# =============================================================================
def metric_card(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    return (f'<div class="metric-card tone-{tone}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>')


def z_tone(z: float, cfg: dict) -> str:
    """Tom do z — usado pelo card E pelo gráfico.

    Antes eram duas escalas diferentes: o card cortava em ±z_monitor com 3 tons
    e o gráfico cortava em ±z_alert com 5. Com z = +1,2 o card ficava vermelho
    e a barra ficava laranja, na mesma tela.
    """
    if z >= cfg["z_monitor"]:
        return "short"
    if z <= -cfg["z_monitor"]:
        return "long"
    return "neutral"


def z_bar_colors(z_series, cfg: dict) -> list:
    """Cores das barras do z: 2 matizes × 2 intensidades. Nunca um matiz novo."""
    out = []
    for v in z_series.fillna(0):
        if v >= cfg["z_alert"]:
            out.append(rgba(C_SHORT, LVL_ALERT))
        elif v >= cfg["z_monitor"]:
            out.append(rgba(C_SHORT, LVL_WATCH))
        elif v <= -cfg["z_alert"]:
            out.append(rgba(C_LONG, LVL_ALERT))
        elif v <= -cfg["z_monitor"]:
            out.append(rgba(C_LONG, LVL_WATCH))
        else:
            out.append(C_FLAT)
    return out


def style_fig(fig, height: int, top_margin: int = 60, show_legend: bool = True,
              legend_y: float = 1.03):
    """Layout Plotly consistente.

    Dois detalhes que mudam a leitura:
      * spikemode="across" em vez de hovermode="x unified" — o unified só
        unifica DENTRO do subplot sob o cursor, então ao inspecionar o z-score
        você perdia o ratio, que é exatamente a comparação cruzada que
        justifica empilhar os painéis.
      * margin b=10 — b=0 cortava os ticks do eixo x do painel inferior.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=rgba(C_LINE, 0.25),
        font=dict(color=C_TEXT, size=12),
        hovermode="x",
        hoverlabel=dict(bgcolor=C_SUNKEN, bordercolor=C_FLAT,
                        font=dict(color=C_TEXT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=legend_y, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        showlegend=show_legend,
        margin=dict(l=8, r=8, t=top_margin, b=10),
    )
    # automargin: com margin l=8 e automargin False (padrao), rotulos de eixo
    # como "-2.5" ou "0.4523" (~35px) sao clipados na borda do SVG.
    fig.update_xaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE,
                     automargin=True,
                     showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor=C_GRAPHIC, spikethickness=1)
    fig.update_yaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE,
                     automargin=True)
    return fig
