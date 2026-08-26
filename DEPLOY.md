# Deploy na VPS — LS Dashboard (Docker + Traefik + HTTPS)

O painel roda num container Docker e entra no **Traefik** já existente na VPS
via labels — o certificado HTTPS (Let's Encrypt) é automático.

URL: **https://longshort.marketwatchrf.com** (registro A → IP da VPS).

## Subir / atualizar

```bash
cd ~/LS---Dashboard
git pull
docker compose up -d --build
```

## Segredos (token brapi + chave Anthropic)

Os segredos ficam num arquivo **`.env`** na raiz do projeto, **na VPS**
(ignorado pelo git e pelo build da imagem). O docker compose injeta os
valores no container:

```bash
cat > ~/LS---Dashboard/.env <<'EOF'
BRAPI_TOKEN=seu-token-pro-do-brapi
ANTHROPIC_API_KEY=sk-ant-sua-chave
EOF
chmod 600 ~/LS---Dashboard/.env
docker compose up -d          # recria o container com as variáveis
```

Com os segredos no servidor, a barra lateral mostra "✅ configurado no
servidor" e **nunca envia o valor ao navegador** (site é público). Sem
`.env`, o app mostra campos mascarados pra colar manualmente — válidos só
naquela sessão do navegador.

> Alternativa: `.streamlit/secrets.toml` (formato TOML: `BRAPI_TOKEN = "..."`)
> também funciona — o diretório `.streamlit/` é montado no container.
> **Atenção:** se existir um *diretório* chamado `secrets.toml` (criado por um
> bind mount antigo do Docker), remova antes: `rm -rf .streamlit/secrets.toml`.

## Senha de acesso (Basic Auth via Traefik)

O site pede usuário e senha antes de abrir (janela do navegador). As
credenciais ficam no `.env` como hash — a senha nunca aparece em texto puro
nem vai pro git.

```bash
cd ~/LS---Dashboard
# gera o hash (digite a senha 2x; ela fica oculta)
HASH=$(openssl passwd -apr1)
# grava no .env (troque "gustavo" pelo usuário que quiser)
echo "LS_AUTH_USERS=gustavo:$HASH" >> .env
docker compose up -d
```

- Trocar a senha: apague a linha `LS_AUTH_USERS` do `.env`, repita os
  comandos acima e rode `docker compose up -d`.
- Mais de um usuário: separe por vírgula
  (`LS_AUTH_USERS=ana:hash1,beto:hash2`).
- Sem `LS_AUTH_USERS` no `.env`, o `docker compose up` falha de propósito com
  uma mensagem clara — melhor do que subir o site aberto sem querer.

## Comandos do dia a dia

| Ação | Comando |
|---|---|
| Logs do app | `docker compose logs -f app` |
| Reiniciar | `docker compose restart` |
| Parar | `docker compose down` |
| Atualizar (novo commit) | `git pull && docker compose up -d --build` |
| Status | `docker compose ps` |
| Ver se env chegou no container | `docker compose exec app printenv \| grep -E 'BRAPI\|ANTHROPIC'` |

## Solução de problemas

- **"token invalido/ausente" ao Analisar:** confira o `.env` (sem aspas, sem
  espaços em volta do `=`) e rode `docker compose up -d` pra recriar o
  container — `restart` sozinho **não** relê o `.env`.
- **HTTPS não sobe:** confirme o DNS (`dig +short longshort.marketwatchrf.com`
  deve responder o IP da VPS) e veja os logs do Traefik.
- **502:** `docker compose logs app`.
