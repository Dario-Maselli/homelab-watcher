# 🧭 homelab-watcher
Git → Docker Compose redeployer for your homelab… with Vault secrets… repo-controlled stacks… Windows friendly… self-healing retries… and clear notifications

<p align="left"> <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.13.5%2B-blue"></a> <a href="#"><img alt="OS" src="https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-informational"></a> <a href="#"><img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-brightgreen"></a> <a href="#"><img alt="Vault" src="https://img.shields.io/badge/Secrets-HashiCorp%20Vault-7c3aed"></a> </p>
---

## ✨ What it does

1. Watches your homelab Git repo for new commits

2. Keeps a deploy copy that is separate from your dev checkout

3. Reads desired stacks from your repo: .homelab/stacks.yaml

4. Pulls env from Vault KV v2 or files… optionally writes .env beside the compose file

5. Validates docker compose config then pulls and brings stacks up

6. Persists previous state and brings removed stacks down

7. Notifies via Discord and Email… retries on transient failures

---

## 🧩 Repo layout it expects
### Your homelab repo defines stacks… not the watcher

```
./homelab/
├── stacks.yaml
├── automation/
│ └── docker-compose.yml
├── authentication/
│ └── docker-compose.yml
├── core/
│ └── docker-compose.yml
├── monitoring/
│ └── docker-compose.yml
├── automation/
│ └── docker-compose.yml
```
### Example .homelab/stacks.yaml:
```yaml
stacks:
  - dir: automation
    compose: docker-compose.yml

  - dir: authentication
    compose: docker-compose.yml
    env:
      backend: vault
      secret_path: kv/homelab/authentication
      data_key: env
      optional: true
      materialize: ".env"
      project_name: authentication

  - dir: databases
    compose: docker-compose.yml
    env:
      backend: vault
      secret_path: kv/homelab/databases
      data_key: env
      optional: true
      materialize: ".env"

  - dir: media
    compose: docker-compose.yaml

  - dir: monitoring
    compose: docker-compose.yml
    env:
      backend: vault
      secret_path: kv/homelab/monitoring
      data_key: env
      optional: true
      materialize: ".env"
```

> `env.materialize` writes a file in the stack folder so env_file: works without extra wiring `project_name` fixes the Compose project name… makes teardown deterministic

---

## 📦 Requirements
- Python 3.13 or newer
- Docker Desktop or Docker Engine
- git available on PATH
- Optional… HashiCorp Vault for secrets

### Install Python deps:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Install
- Python 3.13 or newer
- Docker Desktop or Docker Engine
- git available on PATH
- Optional… HashiCorp Vault for secrets

### Install Python deps:
```bash
# Windows
git clone https://github.com/Dario-Maselli/homelab-watcher.git
cd homelab-watcher
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux or macOS
git clone https://github.com/Dario-Maselli/homelab-watcher.git
cd homelab-watcher
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configure
Create .env next to watcher.py or next to watcher.yaml

You can also point HOMELAB_ENV_FILE to an absolute path

```env
# Vault
VAULT_ADDR=http://127.0.0.1:8200
VAULT_TOKEN=replace-with-a-real-token
VAULT_UNSEAL_KEY=optional-single-node-lab

# Notifications
DISCORD_WEBHOOK=
EMAIL_ENABLED=false
EMAIL_FROM=homelab@yourdomain.com
EMAIL_TO=you@yourdomain.com,alerts@yourdomain.com
EMAIL_SMTP_HOST=smtp.yourmailserver.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=homelab@yourdomain.com
EMAIL_PASSWORD=super-secret
EMAIL_STARTTLS=true

# Watcher knobs
LOG_STEPS=true
POLL_INTERVAL_SECONDS=20
DOCKER_READY_TIMEOUT=120
COMPOSE_TIMEOUT=300
VAULT_WAIT_TIMEOUT=120
DOWN_REMOVE_VOLUMES=false

# Git
REPO_URL=git@github.com:Dario-Maselli/homelab.git
BRANCH=main

# Where the deploy copy lives
DEPLOY_PATH=D:\homelab-deploy

# Optional worktree mode
# SOURCE_REPO_PATH=D:\7-GitHubRepos\homelab
```

### Create watcher.yaml:

```yaml
poll_interval_seconds: 30
log_steps: true

notify:
  discord_webhook: ${DISCORD_WEBHOOK}
  email:
    enabled: ${EMAIL_ENABLED}
    smtp_host: ${EMAIL_SMTP_HOST}
    smtp_port: ${EMAIL_SMTP_PORT}
    username: ${EMAIL_USERNAME}
    password: ${EMAIL_PASSWORD}
    from: ${EMAIL_FROM}
    to: ${EMAIL_TO}
    starttls: ${EMAIL_STARTTLS}

projects:
  - name: homelab
    repo_url: ${REPO_URL}
    branch: ${BRANCH}
    deploy_path: ${DEPLOY_PATH}
    # mode: worktree
    # source_repo_path: ${SOURCE_REPO_PATH}
```
---

## 🔐 Vault quick start
Enable KV v2 and write secrets as either one blob under key env or as key by key

### Install Python deps:
```bash
# once
vault secrets enable -path=kv kv-v2

# NOTE: I made use of the UI to create the env files

# single blob… watcher uses data_key: env
vault kv put kv/homelab/monitoring env=@./monitoring.env

# or discrete keys
vault kv put kv/homelab/databases POSTGRES_USER=app POSTGRES_PASSWORD=secret TZ=Africa/Johannesburg
```

Watcher will:

- wait for Vault readiness if a stack needs non-optional secrets

- attempt one unseal if VAULT_UNSEAL_KEY is provided

- skip optional secrets when sealed without spamming

---

## ▶️ Run
```bash
# Windows
.\.venv\Scripts\Activate.ps1
python watcher.py watcher.yaml
```
```bash
# Linux or macOS
source .venv/bin/activate
python watcher.py watcher.yaml
```
You will see a startup banner… then periodic logs as it polls and deploys

---

## 🧯 Windows service options
### Task Scheduler
1. Create task that runs at startup
2. Program python
3. Arguments watcher.py watcher.yaml
4. Start in your watcher folder
5. Run with highest privilege

### NSSM
```powershell
nssm install homelab-watcher "C:\Path\To\Python\python.exe" "C:\Path\To\watcher.py" "C:\Path\To\watcher.yaml"
nssm set homelab-watcher AppDirectory "C:\Path\To\homelab-watcher"
nssm start homelab-watcher
```
---

## 🔄 How removals work
Watcher stores the last applied set of stacks per project
```bash
Windows  %USERPROFILE%\.homelab_watcher\state\<project>_stacks.json
Linux    ~/.homelab_watcher/state/<project>_stacks.json
```

On the next cycle after a commit
1. Loads desired state from .homelab/stacks.yaml in the deploy copy
2. Compares against the saved file
3. Anything missing is brought down with

```bash
docker compose -f <compose> -p <project_name> down --remove-orphans
```

If you also want volumes gone set DOWN_REMOVE_VOLUMES=true

Pro tip… set a `project_name` for each stack or ensure the env used for deploy has `COMPOSE_PROJECT_NAME=…` so up and down agree

---

## 🔔 Notifications
- Discord via `DISCORD_WEBHOOK`
- Email via `EMAIL_*` vars

### Watcher notifies
- on startup and when Docker is not ready
- when a stack deploys
- when a stack fails five times
- when a removed stack is brought down
---

## 🧠 Modes for safety while you develop
### Clone
Keeps an isolated clone under `DEPLOY_PATH` that tracks the remote branch… development in your local checkout never gets touched

### Worktree mode
Set `SOURCE_REPO_PATH` and switch project `mode` to `worktree`… watcher creates a detached worktree under `DEPLOY_PATH` pinned to `origin/<branch>`… your main checkout remains free to develop on any branch

---

## 🧪 Troubleshooting
### Vault is sealed
You will see a wait message… unseal or provide `VAULT_UNSEAL_KEY`… optional stacks proceed without noise

### Env file missing
Watcher writes cache copies here

```bash
%USERPROFILE%\.homelab_watcher\cache\env\<project>\<stack>\.env
```

If your compose uses `env_file:` keep `materialize: ".env"` in the stack config

### Discord silent
Confirm the variable is actually set then test

```powershell
Invoke-WebRequest -Method Post -ContentType "application/json" -Body '{"content":"test…"}' -Uri $env:DISCORD_WEBHOOK
```

### Git auth fails
Prefer SSH `git@github.com:…` with your key loaded in the agent or use PAT with `https://` and a credential helper

### Removed stack did not go down
Add `project_name` in the stack entry or ensure the env defines `COMPOSE_PROJECT_NAME`… commit… watcher will capture it on the next cycle and remove it after you delete it from desired state
---

## 📘 Reference

Key env vars

| Name                    | Meaning                 | Example                          |
| ----------------------- | ----------------------- | -------------------------------- |
| REPO\_URL               | homelab repo remote     | `git@github.com:you/homelab.git` |
| BRANCH                  | branch to deploy        | `main`                           |
| DEPLOY\_PATH            | path of deploy copy     | `D:\homelab-deploy`              |
| SOURCE\_REPO\_PATH      | enable worktree mode    | `D:\Repos\homelab`               |
| VAULT\_ADDR             | Vault URL               | `http://127.0.0.1:8200`          |
| VAULT\_TOKEN            | Vault token             | `hvs…`                           |
| VAULT\_UNSEAL\_KEY      | single node convenience | one unseal key                   |
| DISCORD\_WEBHOOK        | Discord target          | URL                              |
| EMAIL\_\*               | SMTP config             | values                           |
| POLL\_INTERVAL\_SECONDS | poll cadence            | `20`                             |
| DOCKER\_READY\_TIMEOUT  | docker wait sec         | `120`                            |
| COMPOSE\_TIMEOUT        | compose retries window  | `300`                            |
| VAULT\_WAIT\_TIMEOUT    | vault wait sec          | `120`                            |
| DOWN\_REMOVE\_VOLUMES   | include `-v` on down    | `true` or `false`                |


---

## ✅ Philosophy
1. Infra desired state belongs in the infra repo… not in a separate app config

2. Deploys must be safe for development… so the watcher never flips branches in your dev checkout

3. Secrets live in Vault… materialized only when needed… easily portable to a VPS or another host


---

Happy shipping!

---