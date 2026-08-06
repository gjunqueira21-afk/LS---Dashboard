# Long / Short Dashboard

Painel de pairs trading na B3: spread canônico, z-score, meia-vida de reversão,
cointegração, custo em unidades de z e análise via Claude.

## Arquitetura

| Arquivo         | Papel |
|-----------------|-------|
| `core.py`       | Motor quantitativo — **fonte única de verdade** da estratégia |
| `datasource.py` | Download resiliente, cache adaptativo, alinhamento auditado |
| `tickers.py`    | Normalização de ticker (você nunca digita `.SA`) |
| `theme.py`      | Design system: paleta, CSS, componentes, layout de gráfico |
| `app.py`        | Apresentação **desktop** |
| `mobile.py`     | Apresentação **celular** |

`app.py` e `mobile.py` são só camada de apresentação. Toda a matemática e todas
as cores vêm dos módulos compartilhados — antes as duas telas tinham cópias da
lógica e já haviam divergido (as cores do z-score significavam coisas opostas
em cada versão).

## Especificação canônica

Um único spread para sinal, teste estatístico, gráfico e dimensionamento:

```
spread = ln(LONG) - ln(SHORT)        # beta = 1 IMPOSTO, notional-neutro
z      = (spread - média) / desvio
ADF    = adfuller(spread)            # válido: nada foi estimado da amostra
coint  = coint(ln LONG, ln SHORT)    # Engle-Granger, valores críticos corretos
corr   = correlação de LOG-RETORNOS  # nunca de níveis de preço
sizing = R$ igual em cada perna
```

O beta OLS continua sendo calculado e exibido, mas é **informativo** — não entra
no sinal, no teste nem no sizing.

### Por que assim

- **Um spread, não três.** O painel antigo rodava o z sobre `ln(L/S)` (beta=1),
  exibia um card de beta OLS mandando montar posição beta-neutra, e validava
  com ADF um terceiro objeto (`L − β·S` em níveis de preço). O selo
  "estacionário" validava uma série que ninguém operava.
- **ADF legítimo.** `adfuller` sobre resíduo de beta estimado por OLS usa a
  distribuição errada — é o erro clássico do Engle-Granger em dois passos, com
  falso positivo de ~15% em vez de 5%. Com beta imposto, `adfuller` é correto.
- **Correlação em retornos.** Entre séries de preço em nível a correlação é
  espúria: mede tendência comum. Medido no par PETR4×VALE3: **0,77 em níveis**
  (passaria um gatilho de 0,75) contra **−0,08 em retornos**.
- **Custo em unidades de z.** Round-trip nas duas pernas + aluguel no holding,
  convertido para a mesma escala do sinal. Responde *esse sinal paga a conta?*.

## Tickers

Digite só o ticker. O sufixo `.SA` é resolvido automaticamente:

| Você digita | Resolve para | |
|---|---|---|
| `petr4`, `PETR4 SA`, `petr4.sa` | `PETR4.SA` | ações, units, FIIs, ETFs, BDRs |
| `bova11`, `knri11`, `m1ta34` | `…SA` | inclusive BDR com dígito na raiz |
| `aapl`, `spy`, `brk-b` | intacto | mercado americano |
| `ibov`, `dolar`, `sp500` | `^BVSP`, `USDBRL=X`, `^GSPC` | apelidos |

Tickers B3 crus dão 404 no Yahoo, então nunca há risco de resolver em silêncio
para o ativo errado — ou vem o certo, ou vem erro explícito.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py       # desktop
streamlit run mobile.py    # celular
```

## Publicar (Streamlit Community Cloud)

Duas apps separadas a partir do mesmo repositório — uma apontando para `app.py`,
outra para `mobile.py`. Cada uma ganha um URL próprio.

1. https://share.streamlit.io → login com GitHub
2. **New app** → repositório `gjunqueira21-afk/LS---Dashboard`
3. **Main file path**: `app.py` ou `mobile.py`
4. **Advanced settings → Secrets** (opcional):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. **Deploy**. Repita para o segundo arquivo.

Para usar no celular como app: abra o URL do `mobile.py` e use
*Adicionar à Tela de Início* (Safari) / *Instalar app* (Chrome).

## Cache

O TTL é adaptativo, não fixo — dado diário só muda de verdade durante o pregão:

| Quando | Granularidade |
|---|---|
| Pregão B3 (seg–sex, 10:00–18:30) | 60s |
| Resto do dia útil | 15 min |
| Fim de semana | 6h |

O histórico máximo é baixado uma vez por ticker e recortado em memória: trocar
o seletor de período é instantâneo e não toca a rede.

## Limitações conhecidas

- **Δz usa o módulo.** "Esticando" e "revertendo" — dois trades economicamente
  opostos — caem no mesmo rótulo. A direção é exibida, mas não separa o sinal.
  Separar exigiria saber qual das duas semânticas o backtest validou.
- **Sem livro de posições.** O painel não sabe se você tem posição aberta além
  do checkbox manual; `max_hold` não é comparado a uma data de entrada.
- **Sem backtest embutido.** Convergência do z não é evidência de reversão: num
  passeio aleatório a regra converge em ~90% dos casos com ~68% de trades
  vencedores e expectância negativa. Só um backtest com PnL líquido e t-stat
  responde isso.
- **`cache_bucket` não conhece feriados da B3** — usa dia da semana e horário.

> Rodar em produção sombra antes de execução real.
