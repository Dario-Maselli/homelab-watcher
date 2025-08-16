#!/usr/bin/env python3
"""
watcher.py
Git → Docker Compose redeployer with Vault-backed env and async-safe git:

- Project modes:
  - CLONE  … deploy from an isolated clone under deploy_path
  - WORKTREE … deploy from a detached git worktree linked to source_repo_path
- Never switches your dev repo branch… only fetches in the source… resets the deploy copy only
- Vault KV v2 or file-based env per stack… writes a local cached .env per stack
- If a stack does not need an env… omit env spec… or set env.optional: true to ignore missing
- Auto-discovery of compose stacks when not listed
- Docker readiness wait with backoff… compose retries
- Discord and email notifications
- Step logging with LOG_STEPS
"""

import os, re, sys, time, json, random, socket, traceback, subprocess, smtplib, shutil
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.utils import formatdate
from typing import Dict, List, Optional, Tuple
import yaml
from dotenv import load_dotenv

# ---------- basics

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def host() -> str:
    return socket.gethostname()

def as_bool(val, default=False) -> bool:
    if val is None:
        return default
    return str(val).lower() in ("true", "1", "yes", "y", "on")

LOG_STEPS = as_bool(os.environ.get("LOG_STEPS", "false"))

def log(msg: str, step: bool = False) -> None:
    if step and not LOG_STEPS:
        return
    print(f"[{now()}] {msg}", flush=True)

def run(cmd: str, cwd: Optional[Path] = None, check: bool = True, capture: bool = True) -> str:
    if LOG_STEPS:
        print(f"[{now()}] RUN: {cmd} (cwd={cwd})", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=os.environ.copy(),
    )
    if check and result.returncode != 0:
        output = result.stdout or ""
        raise RuntimeError(f"command failed [{result.returncode}]… {cmd}\n{output}")
    return result.stdout if capture else ""

# ---------- notifications

def _split_csv(val: str) -> List[str]:
    if not val:
        return []
    return [p.strip() for p in str(val).split(",") if p and p.strip()]

def send_discord(webhook: str, message: str) -> None:
    if not webhook:
        return
    try:
        import urllib.request
        data = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(webhook, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        log("Sent Discord notification", step=True)
    except Exception:
        pass

def send_email(cfg: dict, subject: str, body: str) -> None:
    if not cfg or not as_bool(cfg.get("enabled")):
        return
    tos = _split_csv(cfg.get("to", "")) if isinstance(cfg.get("to"), str) else (cfg.get("to") or [])
    tos = [t for t in tos if t]
    if not tos:
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = cfg["from"]
    msg["To"] = ", ".join(tos)
    msg["Date"] = formatdate(localtime=True)
    with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as s:
        if as_bool(cfg.get("starttls", True)):
            s.starttls()
        if cfg.get("username"):
            s.login(cfg["username"], cfg["password"])
        s.sendmail(cfg["from"], tos, msg.as_string())
    log("Sent email notification", step=True)

# ---------- docker helpers

def docker_cli_name() -> str:
    try:
        out = run("docker compose version")
        if out:
            return "docker compose"
    except Exception:
        pass
    if shutil.which("docker-compose"):
        return "docker-compose"
    return "docker compose"

def wait_for_docker(max_wait_seconds: int = 600) -> bool:
    log("Waiting for Docker engine", step=True)
    start = time.time()
    delay = 2.0
    while True:
        try:
            out = run("docker info", check=True)
            if "Server Version" in out or "Storage Driver" in out:
                return True
        except Exception:
            pass
        if time.time() - start >= max_wait_seconds:
            return False
        time.sleep(min(delay, 20) + random.uniform(0, 0.75))
        delay *= 1.7

def compose_up(stack_dir: Path, compose_file: str, env_file: Optional[Path]) -> None:
    cli = docker_cli_name()
    base = f'{cli} -f "{compose_file}"'
    if env_file and env_file.exists():
        base = f'{base} --env-file "{env_file}"'
        log(f"Using env file: {env_file}", step=True)
    log(f"Compose up in {stack_dir} using {compose_file}", step=True)
    run(f"{base} pull --quiet", cwd=stack_dir, check=False)
    run(f"{base} up -d --remove-orphans", cwd=stack_dir)

def compose_up_with_retry(stack_dir: Path, compose_file: str, env_file: Optional[Path], total_timeout: int = 300) -> None:
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            compose_up(stack_dir, compose_file, env_file)
            return
        except Exception:
            if time.time() - start > total_timeout:
                raise
            log(f"Retry compose_up attempt {attempt}", step=True)
            time.sleep(min(5 + attempt, 20))

# ---------- git helpers

def ensure_clone(repo_url: str, branch: str, deploy_path: Path) -> None:
    """Keep an isolated clone in deploy_path… never touches your dev checkout."""
    if not deploy_path.exists():
        log(f"Cloning {repo_url} into {deploy_path}")
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        run(f'git clone --branch {branch} --single-branch "{repo_url}" "{deploy_path}"')
    else:
        try:
            current = run("git remote get-url origin", cwd=deploy_path).strip()
            if current != repo_url:
                log(f"Updating origin URL in {deploy_path}", step=True)
                run(f'git remote set-url origin "{repo_url}"', cwd=deploy_path)
        except Exception:
            pass

def update_clone_to_origin(branch: str, deploy_path: Path) -> Tuple[str, str]:
    """Fetch and hard reset the isolated clone to origin/branch."""
    run("git fetch --prune origin", cwd=deploy_path)
    local = run("git rev-parse HEAD", cwd=deploy_path).strip()
    remote = run(f"git rev-parse origin/{branch}", cwd=deploy_path).strip()
    if local != remote:
        log(f"Resetting clone to origin/{branch}", step=True)
        run(f"git reset --hard origin/{branch}", cwd=deploy_path)
    return local, remote

def worktree_exists(source_repo: Path, deploy_path: Path) -> bool:
    try:
        out = run("git worktree list --porcelain", cwd=source_repo)
        for line in out.splitlines():
            if line.startswith("worktree "):
                wt_path = Path(line.split(" ", 1)[1].strip())
                if wt_path.resolve() == deploy_path.resolve():
                    return True
    except Exception:
        pass
    return deploy_path.exists() and (deploy_path / ".git").exists()

def ensure_worktree(repo_url: str, branch: str, source_repo: Path, deploy_path: Path) -> None:
    """
    Create a detached worktree at deploy_path tracking origin/branch…
    We only git fetch in source_repo… we never checkout or switch your dev branch.
    """
    if not (source_repo.exists() and (source_repo / ".git").exists()):
        raise SystemExit(f"FATAL: SOURCE_REPO_PATH is not a valid repo: {source_repo}")

    # keep origin accurate if a repo_url was provided
    if repo_url:
        try:
            current = run("git remote get-url origin", cwd=source_repo).strip()
            if current != repo_url:
                log(f"Updating origin URL in source repo {source_repo}", step=True)
                run(f'git remote set-url origin "{repo_url}"', cwd=source_repo)
        except Exception:
            pass

    run(f"git fetch origin {branch}", cwd=source_repo)

    if not worktree_exists(source_repo, deploy_path):
        log(f"Adding worktree at {deploy_path} -> origin/{branch}")
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        # detached worktree at the remote branch tip
        run(f'git worktree add --detach "{deploy_path}" origin/{branch}', cwd=source_repo)

def update_worktree_to_origin(branch: str, deploy_path: Path, source_repo: Path) -> Tuple[str, str]:
    """Fetch in source repo… then hard reset the worktree to the fetched remote tip… dev repo is never switched."""
    run(f"git fetch origin {branch}", cwd=source_repo)
    local = run("git rev-parse HEAD", cwd=deploy_path).strip()
    remote = run(f"git rev-parse origin/{branch}", cwd=source_repo).strip()
    if local != remote:
        log(f"Resetting worktree to origin/{branch}", step=True)
        run(f"git reset --hard {remote}", cwd=deploy_path)
    return local, remote

# ---------- config and discovery

ENV_ONLY_PATTERN = re.compile(r"^\$\{([^}]+)\}$")
COMPOSE_FILENAMES = ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml")

def interpolate_env(obj):
    if isinstance(obj, dict):
        return {k: interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate_env(x) for x in obj]
    if isinstance(obj, str):
        m = ENV_ONLY_PATTERN.match(obj)
        if m:
            return os.environ.get(m.group(1), "")
        return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), obj)
    return obj

def discover_stacks(root: Path, max_depth: int = 2, excludes: Optional[List[str]] = None) -> List[Tuple[Path, str]]:
    excludes = set(excludes or [".git", ".github", ".gitea", ".vscode", "__pycache__"])
    results: List[Tuple[Path, str]] = []
    def walk(dir_path: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for child in dir_path.iterdir():
                if child.name in excludes:
                    continue
                if child.is_dir():
                    for fname in COMPOSE_FILENAMES:
                        if (child / fname).exists():
                            results.append((child, fname))
                            break
                    walk(child, depth + 1)
        except PermissionError:
            pass
    walk(root, 0)
    dedup: Dict[str, Tuple[Path, str]] = {}
    for d, f in results:
        dedup[str(d.resolve())] = (d, f)
    return list(dedup.values())

def default_base_dir() -> Path:
    p = os.environ.get("HOMELAB_BASE_DIR")
    if p:
        return Path(p).expanduser()
    return Path.home() / ".homelab_watcher"

def resolve_env_file(cfg_path: Optional[Path]) -> Path:
    o = os.environ.get("HOMELAB_ENV_FILE")
    if o:
        return Path(o).expanduser()
    if cfg_path:
        c = cfg_path.parent / ".env"
        if c.exists():
            return c
    c = Path(__file__).resolve().parent / ".env"
    if c.exists():
        return c
    c = Path.cwd() / ".env"
    if c.exists():
        return c
    return default_base_dir() / ".env"

# ---------- env backends

def _env_cache_root() -> Path:
    return (default_base_dir() / "cache" / "env").expanduser()

def write_env_cache(project: str, stack: str, content: str) -> Path:
    root = _env_cache_root() / safe_name(project) / safe_name(stack)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".env"
    path.write_text(content, encoding="utf-8")
    return path

def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x or "")

def fetch_env_file(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None

def vault_kv2_read(secret_path: str, addr: Optional[str], token: Optional[str], timeout: int = 10) -> Dict[str, str]:
    """
    Read from Vault KV v2: GET {addr}/v1/<mount>/data/<path>
    secret_path looks like 'kv/homelab/monitoring'
    Returns dict from .data.data
    """
    if not secret_path or "/" not in secret_path:
        raise RuntimeError(f"vault secret_path must look like 'kv/namespace/...' not '{secret_path}'")
    addr = addr or os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
    token = token or os.environ.get("VAULT_TOKEN", "")
    mount, rel = secret_path.split("/", 1)
    url = f"{addr.rstrip('/')}/v1/{mount}/data/{rel}"
    import urllib.request, urllib.error
    req = urllib.request.Request(url, headers={"X-Vault-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Vault HTTP {e.code} on {url}\n{body}") from None
    except Exception as e:
        raise RuntimeError(f"Vault fetch failed: {e}") from None
    data = (payload or {}).get("data", {}).get("data", {})
    if not isinstance(data, dict):
        raise RuntimeError(f"Vault returned unexpected payload for {secret_path}")
    return {str(k): "" if v is None else str(v) for k, v in data.items()}

def build_env_blob_from_map(kv: Dict[str, str]) -> str:
    # single key named env means a full .env blob
    if "env" in kv and len(kv) == 1:
        return kv["env"]
    lines = []
    for k, v in kv.items():
        if re.search(r"\s", k):
            raise RuntimeError(f"Invalid env key with whitespace: {k}")
        lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"

def get_env_for_stack(project_name: str, stack_name: str, stack_dir: Path, env_spec: Optional[dict]) -> Optional[Path]:
    """
    Resolve env for a stack.
    env_spec:
      backend: vault | file
      optional: true to ignore missing secret or file
      # vault
      secret_path: kv/homelab/monitoring
      data_key: env   … optional… if present use only that key as blob
      addr: override VAULT_ADDR
      token: override VAULT_TOKEN
      # file
      path: relative/or/absolute .env path
    """
    if not env_spec:
        return None

    backend = (env_spec.get("backend") or "file").lower()
    optional = as_bool(env_spec.get("optional", False))

    if backend == "file":
        p = env_spec.get("path")
        if not p:
            return None
        content = fetch_env_file(str(stack_dir / p) if not os.path.isabs(p) else p)
        if content is None:
            if optional:
                log(f"[{project_name}/{stack_name}] optional env file not found… continuing", step=True)
                return None
            raise RuntimeError(f"[{project_name}/{stack_name}] env file not found: {p}")
        return write_env_cache(project_name, stack_name, content)

    if backend == "vault":
        secret_path = env_spec.get("secret_path", "").strip()
        addr = env_spec.get("addr")
        token = env_spec.get("token")
        try:
            kv = vault_kv2_read(secret_path, addr, token)
        except Exception as e:
            if optional:
                log(f"[{project_name}/{stack_name}] optional vault fetch failed… {e}", step=True)
                return None
            raise
        data_key = env_spec.get("data_key")
        if data_key:
            if data_key not in kv:
                if optional:
                    log(f"[{project_name}/{stack_name}] optional key '{data_key}' missing… continuing", step=True)
                    return None
                raise RuntimeError(f"[{project_name}/{stack_name}] vault path {secret_path} missing key '{data_key}'")
            blob = str(kv[data_key])
            return write_env_cache(project_name, stack_name, blob)
        blob = build_env_blob_from_map(kv)
        return write_env_cache(project_name, stack_name, blob)

    raise RuntimeError(f"[{project_name}/{stack_name}] unsupported env backend: {backend}")

# ---------- main

def main():
    # config path
    if len(sys.argv) >= 2:
        cfg_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        cfg_path = (default_base_dir() / "watcher.yml").resolve()

    # env path
    env_file = resolve_env_file(cfg_path if cfg_path.exists() else None)
    load_dotenv(dotenv_path=str(env_file))

    # refresh step flag
    global LOG_STEPS
    LOG_STEPS = as_bool(os.environ.get("LOG_STEPS", "false"))

    banner = [
        "=== homelab watcher starting ===",
        f" host:        {host()}",
        f" env file:    {env_file}",
        f" config path: {cfg_path if cfg_path.exists() else '<not found>'}",
        f" log steps:   {LOG_STEPS}",
    ]
    for line in banner:
        print(line, flush=True)

    # load config
    poll = int(os.environ.get("POLL_INTERVAL_SECONDS", 20))
    projects = []
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg = interpolate_env(cfg)
        if "log_steps" in cfg:
            globals()["LOG_STEPS"] = as_bool(cfg.get("log_steps"))
        poll = int(cfg.get("poll_interval_seconds", poll))
        notify = cfg.get("notify", {}) or {}
        discord_webhook = notify.get("discord_webhook", os.environ.get("DISCORD_WEBHOOK", ""))
        email_cfg = notify.get("email", {
            "enabled": os.environ.get("EMAIL_ENABLED", "false"),
            "smtp_host": os.environ.get("EMAIL_SMTP_HOST", ""),
            "smtp_port": os.environ.get("EMAIL_SMTP_PORT", "587"),
            "username": os.environ.get("EMAIL_USERNAME", ""),
            "password": os.environ.get("EMAIL_PASSWORD", ""),
            "from": os.environ.get("EMAIL_FROM", "homelab-watcher@local"),
            "to": os.environ.get("EMAIL_TO", ""),
            "starttls": os.environ.get("EMAIL_STARTTLS", "true"),
        })
        projects = cfg.get("projects", [])
    else:
        discord_webhook = os.environ.get("DISCORD_WEBHOOK", "")
        email_cfg = {
            "enabled": os.environ.get("EMAIL_ENABLED", "false"),
            "from": os.environ.get("EMAIL_FROM"),
            "to": os.environ.get("EMAIL_TO"),
            "smtp_host": os.environ.get("EMAIL_SMTP_HOST"),
            "smtp_port": os.environ.get("EMAIL_SMTP_PORT", "587"),
            "username": os.environ.get("EMAIL_USERNAME"),
            "password": os.environ.get("EMAIL_PASSWORD"),
            "starttls": os.environ.get("EMAIL_STARTTLS", "true"),
        }

    if not projects:
        projects = [{
            "name": os.environ.get("PROJECT_NAME", "homelab"),
            "repo_url": os.environ.get("REPO_URL", ""),
            "branch": os.environ.get("BRANCH", "main"),
            "deploy_path": os.environ.get("DEPLOY_PATH", str(default_base_dir() / "deploy" / "homelab")),
            "mode": "clone",
        }]

    print(" projects:", flush=True)
    for p in projects:
        print(f"  - {p.get('name')}  mode={p.get('mode','clone')}  repo={p.get('repo_url')}  branch={p.get('branch')}"
              f"  deploy_path={p.get('deploy_path') or p.get('path')}  source_repo={p.get('source_repo_path','')}", flush=True)

    last_seen: Dict[str, str] = {}
    pending: Dict[str, str] = {}
    last_docker_warn_ts = 0

    send_discord(discord_webhook, f"homelab watcher started on {host()} at {now()}")

    while True:
        # docker readiness
        ready = wait_for_docker(max_wait_seconds=int(os.environ.get("DOCKER_READY_TIMEOUT", 120)))
        if not ready:
            if time.time() - last_docker_warn_ts > 300:
                msg = f"Docker not ready on {host()} at {now()}… will retry"
                log(msg)
                send_discord(discord_webhook, msg)
                send_email(email_cfg, "[homelab] docker not ready", msg)
                last_docker_warn_ts = time.time()
            time.sleep(poll)
            continue

        for proj in projects:
            name = proj.get("name", "project")
            try:
                repo_url = proj.get("repo_url", "")
                branch = proj.get("branch", "main")
                deploy_path_str = proj.get("deploy_path") or proj.get("path") or ""
                if not deploy_path_str:
                    raise SystemExit(f"FATAL: project '{name}' missing deploy_path/path")
                deploy_path = Path(deploy_path_str).expanduser().resolve()
                mode = (proj.get("mode") or "clone").lower()
                source_repo_path = Path(proj.get("source_repo_path")).expanduser().resolve() \
                    if proj.get("source_repo_path") else None

                if not repo_url:
                    raise SystemExit(f"FATAL: project '{name}' has empty repo_url")

                # fetch and compute hashes without touching your dev checkout
                if mode == "worktree":
                    if not source_repo_path:
                        raise SystemExit(f"FATAL: project '{name}' mode=worktree requires source_repo_path")
                    ensure_worktree(repo_url, branch, source_repo_path, deploy_path)
                    local, remote = update_worktree_to_origin(branch, deploy_path, source_repo_path)
                else:
                    ensure_clone(repo_url, branch, deploy_path)
                    local, remote = update_clone_to_origin(branch, deploy_path)

                if last_seen.get(name) is None:
                    last_seen[name] = local

                needs_deploy = pending.get(name) or (remote != local)

                if needs_deploy:
                    pending[name] = remote or "pending"

                    # discover stacks
                    stacks_cfg = proj.get("stacks", [])
                    stacks: List[Tuple[Path, str, Optional[dict]]] = []
                    if stacks_cfg:
                        for s in stacks_cfg:
                            d = deploy_path / s.get("dir", ".")
                            fname = s.get("compose", "")
                            if not fname:
                                for c in COMPOSE_FILENAMES:
                                    if (d / c).exists():
                                        fname = c
                                        break
                            if fname and (d / fname).exists():
                                stacks.append((d, fname, s.get("env")))
                    else:
                        for d, fname in discover_stacks(deploy_path, max_depth=int(os.environ.get("DISCOVERY_DEPTH", 2))):
                            stacks.append((d, fname, None))

                    if stacks:
                        for d, fname, env_spec in stacks:
                            log(f"[{name}] stack: {d} / {fname}", step=True)
                            env_path = get_env_for_stack(name, d.name or "stack", d, env_spec) if env_spec else None
                            compose_up_with_retry(d, fname, env_path, total_timeout=int(os.environ.get("COMPOSE_TIMEOUT", 300)))
                    else:
                        log(f"[{name}] no compose stacks found under {deploy_path}… skipping deploy")

                    msg = f"[{name}] deployed {(remote or 'unknown')[:7]} on {host()} at {now()}… was {(local or 'unknown')[:7]}"
                    log(msg)
                    send_discord(discord_webhook, msg)
                    send_email(email_cfg, f"[homelab] {name} updated", msg)
                    last_seen[name] = remote or local
                    pending.pop(name, None)
                else:
                    log(f"[{name}] no changes (local={local[:7]} remote={remote[:7]})")

            except SystemExit as se:
                print(str(se), flush=True)
                return
            except Exception as e:
                tb = traceback.format_exc()
                err = f"[{name}] deployment failed at {now()} on {host()}… {e}\n{tb}"
                log(err)
                try:
                    send_discord(discord_webhook, err[:1900])
                except Exception:
                    pass
                try:
                    send_email(email_cfg, f"[homelab] {name} deployment failed", err)
                except Exception:
                    pass

        time.sleep(int(poll))

if __name__ == "__main__":
    main()
