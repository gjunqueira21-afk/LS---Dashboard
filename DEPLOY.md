# Deploy na VPS — LS Dashboard (Docker + Caddy + HTTPS)

Guia para subir o painel Long/Short numa VPS Linux (Ubuntu/Debian) com HTTPS
automático. O Caddy cuida do certificado (Let's Encrypt) e o Docker isola o app.

## Pré-requisitos

- Uma VPS Linux com acesso `sudo`.
- Um domínio (ou subdomínio) com um registro **A** apontando para o **IP da VPS**.
  Ex.: `painel.seudominio.com  →  203.0.113.10`
- Portas **80** e **443** liberadas no firewall da VPS.

---

## 1. Instalar o Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER      # rodar docker sem sudo
# saia e entre de novo no SSH pra valer o grupo (ou rode: newgrp docker)
```

Confirme:

```bash
docker --version
docker compose version
```

## 2. Clonar o repositório

```bash
git clone https://github.com/gjunqueira21-afk/LS---Dashboard.git
cd LS---Dashboard
```

## 3. Configurar o domínio no Caddy

Edite o `Caddyfile` e troque `painel.seudominio.com` pelo seu domínio real:

```bash
nano Caddyfile
```

## 4. Configurar a API key (análise via Claude)

Crie o arquivo de secrets (ele é ignorado pelo git — fica só no servidor):

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<'EOF'
ANTHROPIC_API_KEY = "sk-ant-COLE-SUA-CHAVE-AQUI"
EOF
chmod 600 .streamlit/secrets.toml
```

> Se não for usar a IA, pode pular este passo — o app funciona sem a chave
> (só os gráficos e estatísticas).

## 5. Liberar as portas no firewall

Se usar `ufw`:

```bash
sudo ufw allow 80
sudo ufw allow 443
sudo ufw reload
```

## 6. Subir o app

```bash
docker compose up -d --build
```

Aguarde ~1 minuto (build + Caddy pegando o certificado). Acesse:

```
https://painel.seudominio.com
```

Pronto! 🎉

---

## Comandos do dia a dia

| Ação | Comando |
|---|---|
| Ver logs do app | `docker compose logs -f app` |
| Ver logs do Caddy (HTTPS) | `docker compose logs -f caddy` |
| Parar | `docker compose down` |
| Reiniciar | `docker compose restart` |
| **Atualizar** (após novo commit) | `git pull && docker compose up -d --build` |
| Status | `docker compose ps` |

## Solução de problemas

- **HTTPS não sobe / erro de certificado:** confirme que o A record do domínio
  aponta pro IP da VPS (`dig +short painel.seudominio.com`) e que as portas 80 e
  443 estão abertas. O Caddy precisa da porta 80 pra validar o certificado.
- **App não abre (502):** veja `docker compose logs app`. Normalmente é erro de
  dependência ou de código; o container reinicia sozinho.
- **Trocar a API key:** edite `.streamlit/secrets.toml` e rode
  `docker compose restart app`.
- **Ver se está de pé sem domínio:** `curl -I http://localhost:8501/_stcore/health`
  de dentro da VPS deve responder `200 OK`.
