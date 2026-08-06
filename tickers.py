"""Normalizacao de tickers para o Yahoo Finance (B3 + mercados globais).

O usuario NUNCA precisa digitar ".SA". Ele digita "petr4" e funciona.
"""

import re
from typing import NamedTuple

# Raiz B3: 4 caracteres (1a letra, demais alfanumericos p/ BDRs nao-patrocinados
# como M1TA34/A1MD34) + 1-2 digitos + sufixo opcional F (fracionario) ou Q (leilao).
_B3_RE = re.compile(r"^([A-Z][A-Z0-9]{3})([0-9]{1,2})([FQ]?)$")

_FIAT = {"USD", "BRL", "EUR", "GBP", "JPY"}

# Alias amigaveis -> simbolo Yahoo. Evita que o usuario precise saber o codigo.
_ALIASES = {
    "IBOV": "^BVSP", "IBOVESPA": "^BVSP", "BVSP": "^BVSP",
    "SP500": "^GSPC", "SPX": "^GSPC", "S&P500": "^GSPC",
    "DOLAR": "USDBRL=X", "USDBRL": "USDBRL=X", "DOL": "USDBRL=X",
    "EURBRL": "EURBRL=X", "BTC": "BTC-USD", "IFIX": "^IFIX",
}


class Symbol(NamedTuple):
    """Resultado da normalizacao.

    raw        : o que o usuario digitou (para mensagens de erro)
    display    : rotulo curto para a UI, sem ruido tecnico ("PETR4", "AAPL")
    candidates : simbolos Yahoo a tentar, em ordem de probabilidade
    market     : "B3" | "US" | "INDEX" | "FX" | "CRYPTO" | "OTHER"
    note       : aviso a exibir ao usuario (ou "")
    """
    raw: str
    display: str
    candidates: tuple
    market: str
    note: str

    @property
    def primary(self) -> str:
        return self.candidates[0]


def normalize_ticker(user_input: str) -> Symbol:
    """Converte entrada livre do usuario em simbolo(s) Yahoo Finance.

    O usuario digita "petr4", "PETR4 SA", " petr4.sa " ou "AAPL" e tudo funciona.

    Regras (nesta ordem):
      1. Limpeza: uppercase, strip, remove espacos internos e pontuacao solta.
      2. "PETR4 SA" / "PETR4.SA" / "PETR4SA" -> reconhece sufixo ja presente,
         nunca duplica.
      3. Simbolos com ^, = ou - sao repassados intactos (indice, cambio, cripto).
      4. Alias conhecidos (IBOV, DOLAR, SP500...) viram o codigo Yahoo.
      5. Padrao B3 (4 chars + 1-2 digitos) -> recebe ".SA".
         Cobre acoes (PETR4), units (SANB11, TAEE11), FIIs (KNRI11),
         ETFs (BOVA11, SMAL11, IVVB11) e BDRs (AAPL34, M1TA34).
      6. Sufixo F (fracionario) e removido: no Yahoo, PETR4F.SA existe mas
         devolve ~1 barra por mes -- serie inutil para pairs trading.
      7. Qualquer outra coisa (AAPL, SPY, BRK-B) e tratada como simbolo global
         e repassada sem sufixo.
      8. Fallback: candidates tem 2 itens quando ha ambiguidade real, para o
         fetch tentar o segundo se o primeiro nao retornar dados.

    Nao faz nenhuma chamada de rede -- e pura, barata e testavel.

    >>> normalize_ticker("petr4").primary
    'PETR4.SA'
    >>> normalize_ticker("AAPL").primary
    'AAPL'
    """
    raw = (user_input or "").strip()
    if not raw:
        return Symbol(raw, "", (), "OTHER", "Ticker vazio.")

    # 1. limpeza: uppercase + remove TODO espaco interno (inclui "PETR4 SA")
    s = re.sub(r"\s+", "", raw.upper())
    s = s.strip(".,;:/")

    # 2. sufixo ".SA" ja presente (com ponto, com espaco ja colapsado, ou colado)
    had_sa = False
    if s.endswith(".SA"):
        s, had_sa = s[:-3], True
    elif s.endswith("SA") and _B3_RE.match(s[:-2]):
        # "PETR4SA" (usuario digitou com espaco, colapsado no passo 1)
        s, had_sa = s[:-2], True

    # 3. indices / cambio / cripto: repassa intacto, nunca leva .SA
    if s.startswith("^"):
        return Symbol(raw, s, (s,), "INDEX", "")
    if "=" in s:
        return Symbol(raw, s, (s,), "FX", "")
    if "-" in s and not had_sa:
        mkt = "CRYPTO" if s.rsplit("-", 1)[-1] in _FIAT else "OTHER"
        return Symbol(raw, s, (s,), mkt, "")

    # 4. alias amigaveis
    if s in _ALIASES:
        y = _ALIASES[s]
        mkt = "INDEX" if y.startswith("^") else "FX" if "=" in y else "CRYPTO"
        return Symbol(raw, s, (y,), mkt, f"{s} interpretado como {y}.")

    # outro sufixo de bolsa ja informado (.L, .TO, .MC...): respeita
    if "." in s:
        return Symbol(raw, s, (s,), "OTHER", "")

    # 5/6. padrao B3
    m = _B3_RE.match(s)
    if m:
        root, num, mod = m.groups()
        base = f"{root}{num}"
        note = ""
        if mod == "F":
            note = (f"{base}F e o fracionario; o Yahoo praticamente nao tem "
                    f"historico dele. Usando {base}.")
        elif mod == "Q":
            note = f"{base}Q e codigo de leilao. Usando {base}."
        return Symbol(raw, base, (f"{base}.SA",), "B3", note)

    # 7. simbolo global (AAPL, SPY, BRK-B, MSFT)
    if had_sa:
        # usuario forcou .SA num simbolo fora do padrao B3 -> respeita, mas
        # deixa o simbolo cru como fallback
        return Symbol(raw, s, (f"{s}.SA", s), "B3", "")

    # 8. ambiguidade: 5-6 chars sem digito final pode ser B3 exotico
    fallback = (s,) if len(s) <= 4 else (s, f"{s}.SA")
    return Symbol(raw, s, fallback, "US", "")


# ---------------------------------------------------------------------------
# CASOS DE TESTE -- verificados contra a API real do Yahoo em 2026-08-06.
#
#   entrada        -> primary        mercado   observacao
#   "petr4"        -> "PETR4.SA"     B3        caso principal do usuario
#   "PETR4"        -> "PETR4.SA"     B3
#   "  petr4  "    -> "PETR4.SA"     B3        strip
#   "PETR4.SA"     -> "PETR4.SA"     B3        ja com sufixo, NAO duplica
#   "petr4.sa"     -> "PETR4.SA"     B3        minusculo com sufixo
#   "PETR4 SA"     -> "PETR4.SA"     B3        espaco interno
#   "PETR4F"       -> "PETR4.SA"     B3        fracionario -> remove F + avisa
#   "VALE3"        -> "VALE3.SA"     B3
#   "BOVA11"       -> "BOVA11.SA"    B3        ETF terminado em 11
#   "KNRI11"       -> "KNRI11.SA"    B3        FII
#   "SANB11"       -> "SANB11.SA"    B3        unit
#   "AAPL34"       -> "AAPL34.SA"    B3        BDR patrocinado
#   "M1TA34"       -> "M1TA34.SA"    B3        BDR c/ DIGITO NA RAIZ (Meta)
#   "AAPL"         -> "AAPL"         US        sem sufixo
#   "BRK-B"        -> "BRK-B"        OTHER     classe de acao com hifen
#   "^BVSP"        -> "^BVSP"        INDEX
#   "ibov"         -> "^BVSP"        INDEX     alias + avisa a traducao
#   "dolar"        -> "USDBRL=X"     FX        alias
#   ""             -> ()             OTHER     note="Ticker vazio."
#
# NAO-COLISAO verificada: tickers B3 crus (PETR4, VALE3, ITUB4, BOVA11, KNRI11,
# SANB11, WEGE3) dao 404 no Yahoo. Nao existe o risco de "resolveu para outro
# ativo em silencio" -- ou vem o ativo certo, ou vem 404.
# ---------------------------------------------------------------------------
