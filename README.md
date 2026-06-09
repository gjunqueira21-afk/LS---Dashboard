# 📊 Long / Short Dashboard

Dashboard de análise de pares (Long/Short) com Z-Score, cointegração, hedge ratio
e análise via Claude AI, além de um painel de Fundamentos (brapi.dev).

## Versões

| Arquivo       | Uso                | Layout                          |
|---------------|--------------------|---------------------------------|
| `app.py`      | **Desktop**        | Largo, sidebar, 2 painéis (LS + Fundamentos) |
| `mobile.py`   | **Celular**        | Coluna única, controles no topo, gráficos empilhados |

As duas versões são independentes e usam a mesma lógica quantitativa.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py       # desktop
streamlit run mobile.py    # mobile
```

## Publicar (Streamlit Community Cloud)

Você pode publicar **duas apps separadas** a partir deste mesmo repositório —
uma apontando para `app.py` (desktop) e outra para `mobile.py` (celular).
Cada uma ganha um URL próprio.

1. Acesse https://share.streamlit.io e faça login com o GitHub.
2. **New app** → selecione o repositório `gjunqueira21-afk/LS---Dashboard`.
3. Em **Main file path**, escolha:
   - `app.py` para a versão desktop, ou
   - `mobile.py` para a versão celular.
4. Em **Advanced settings → Secrets**, cole suas chaves (opcional):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   BRAPI_TOKEN = "seu_token_brapi"
   ```
5. **Deploy**. Repita o processo para o segundo arquivo.

## 📲 Adicionar à tela inicial do celular (como app)

Depois que o `mobile.py` estiver publicado, abra o URL no navegador do celular:

**iPhone (Safari):** botão Compartilhar → *Adicionar à Tela de Início*.
**Android (Chrome):** menu ⋮ → *Adicionar à tela inicial* / *Instalar app*.

O dashboard passa a abrir em tela cheia, como um aplicativo nativo.

## ⏰ Evitar a hibernação do Streamlit Cloud

No plano gratuito, o app "dorme" após dias sem acesso. Para mantê-lo sempre ativo,
configure um monitor gratuito (ex: [UptimeRobot](https://uptimerobot.com)) com um
ping HTTP ao URL do app a cada ~5 minutos.
