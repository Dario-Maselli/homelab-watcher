#!/usr/bin/env python3
"""
watcher.py
Git → Docker Compose redeployer with Vault/file env, repo-controlled stacks,
stateful removals, and Windows-friendly paths.

Key behaviors:
  • Two project modes:
      CLONE      … deploy from an isolated clone under deploy_path
      WORKTREE   … deploy from a detached git worktree tied to source_repo_path
  • Repo-controlled stacks:
      The watcher reads a stacks file from the deploy copy of the homelab repo.
      Search order: .homelab/stacks.yaml, .homelab/stacks.yml, stacks.yaml, stacks.yml
      Override via HOMELAB_STACKS_FILE (absolute or relative to repo root).
  • Env per stack (optional):
      backend: vault | file
      optional: true to ignore missing secret/file
      materialize: ".env" (or any filename) to write next to compose (for env_file:)
      vault: secret_path, data_key, addr, token
      file:  path
  • Vault-aware:
      - waits if required Vault secrets are needed and Vault is sealed/unready
      - optional secrets are skipped while sealed (no spam)
      - optional local auto-unseal via VAULT_UNSEAL_KEY (self-hosted convenience)
  • Stateful cleanups:
      - persists "desired stacks" per project
      - when a stack disappears from desired state, runs `compose down` to remove it
  • Docker readiness wait, compose retries (max 5) with final notifications
  • Discord + email notifications with basic retry and 429 handling
  • Step logging with LOG_STEPS
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

def send_discord(webhook: str, message: str, username: str = None) -> None:
    if not webhook:
        log("Discord webhook not configured", step=True)
        return
    import urllib.request, urllib.error
    import time as _time

    def _post_once(payload: dict) -> Tuple[bool, str]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "homelab-watcher/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                code = getattr(resp, "status", 204)
                if 200 <= code < 300:
                    return True, f"ok {code}"
                return False, f"non-2xx {code}"
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            return False, f"http {e.code} {body.strip()}"
        except Exception as e:
            return False, f"error {type(e).__name__}: {e}"

    msg = str(message or "")
    maxlen = 1900
    chunks = [msg[i:i+maxlen] for i in range(0, len(msg), maxlen)] or [""]

    for idx, chunk in enumerate(chunks, 1):
        payload = {"content": chunk}
        if username:
            payload["username"] = username
        attempts = 0
        while True:
            attempts += 1
            ok, info = _post_once(payload)
            if ok:
                log(f"Sent Discord notification ({idx}/{len(chunks)})", step=True)
                break
            if "http 429" in info.lower():
                wait_s = 2.0
                try:
                    head = urllib.request.Request(webhook, method="HEAD")
                    with urllib.request.urlopen(head, timeout=5) as r:
                        ra = r.headers.get("Retry-After")
                        if ra:
                            wait_s = float(ra)
                except Exception:
                    pass
                log(f"Discord 429… sleeping {wait_s}s", step=False)
                _time.sleep(wait_s)
            else:
                if attempts >= 3:
                    log(f"Discord webhook failed after {attempts} attempts: {info}", step=False)
                    break
                backoff = min(2 * attempts, 10)
                log(f"Discord post failed ({info})… retrying in {backoff}s", step=False)
                _time.sleep(backoff)

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
    # validate interpolated config so YAML/env mistakes show up clearly
    run(f"{base} config -q", cwd=stack_dir)
    log(f"Compose up in {stack_dir} using {compose_file}", step=True)
    run(f"{base} pull --quiet", cwd=stack_dir, check=False)
    run(f"{base} up -d --remove-orphans", cwd=stack_dir)

def compose_up_with_retry(
    stack_dir: Path,
    compose_file: str,
    env_file: Optional[Path],
    project_name: str,
    stack_name: str,
    discord_webhook: str,
    email_cfg: dict,
    max_retries: int = 5,
) -> None:
    attempt = 0
    while True:
        attempt += 1
        try:
            compose_up(stack_dir, compose_file, env_file)
            return
        except Exception as e:
            log(f"[{project_name}/{stack_name}] compose_up failed on attempt {attempt}… {e}")
            if attempt >= max_retries:
                msg = (
                    f"[{project_name}/{stack_name}] deploy failed after {attempt} attempts on {host()} at {now()}…\n"
                    f"{e}"
                )
                try:
                    send_discord(discord_webhook, msg[:1900])
                except Exception:
                    pass
                try:
                    send_email(email_cfg, f"[homelab] {project_name}/{stack_name} deploy failed", msg)
                except Exception:
                    pass
                raise
            time.sleep(min(5 + attempt, 20))
            log(f"[{project_name}/{stack_name}] retrying compose_up attempt {attempt + 1}", step=True)

# ---------- git helpers

def ensure_clone(repo_url: str, branch: str, deploy_path: Path) -> None:
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

def _looks_like_git_url(u: str) -> bool:
    return str(u).startswith(("git@", "https://", "http://"))

def ensure_worktree(repo_url: str, branch: str, source_repo: Path, deploy_path: Path) -> None:
    if not (source_repo.exists() and (source_repo / ".git").exists()):
        raise SystemExit(f"FATAL: SOURCE_REPO_PATH is not a valid repo: {source_repo}")
    if repo_url and _looks_like_git_url(repo_url):
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
        run(f'git worktree add --detach "{deploy_path}" origin/{branch}', cwd=source_repo)

def update_worktree_to_origin(branch: str, deploy_path: Path, source_repo: Path) -> Tuple[str, str]:
    run(f"git fetch origin {branch}", cwd=source_repo)
    local = run("git rev-parse HEAD", cwd=deploy_path).strip()
    remote = run(f"git rev-parse origin/{branch}", cwd=source_repo).strip()
    if local != remote:
        log(f"Resetting worktree to origin/{branch}", step=True)
        run(f"git reset --hard {remote}", cwd=deploy_path)
    return local, remote

# ---------- config & discovery

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

# ---------- compose file resolution (forgiving)

def resolve_compose_file(dir_path: Path, fname_hint: Optional[str]) -> Optional[str]:
    # if a hint is given… use it if it exists
    if fname_hint:
        p = dir_path / fname_hint
        if p.exists():
            return fname_hint
        # try common alternates if hint is wrong (yml vs yaml etc.)
        for alt in COMPOSE_FILENAMES:
            if (dir_path / alt).exists():
                log(f"[resolve] compose '{fname_hint}' not found in {dir_path}… using '{alt}'", step=True)
                return alt
        log(f"[resolve] no compose file found in {dir_path} for hint '{fname_hint}'… skipping", step=True)
        return None
    # no hint… auto-discover
    for c in COMPOSE_FILENAMES:
        if (dir_path / c).exists():
            return c
    log(f"[resolve] no compose file found in {dir_path}… skipping", step=True)
    return None

# ---------- env backends

def _env_cache_root() -> Path:
    return (default_base_dir() / "cache" / "env").expanduser()

def write_env_cache(project: str, stack: str, content: str) -> Path:
    root = _env_cache_root() / safe_name(project) / safe_name(stack)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".env"
    path.write_text(content, encoding="utf-8")
    return path

def write_materialized_env(stack_dir: Path, filename: str, content: str) -> Path:
    target = stack_dir / filename
    target.write_text(content, encoding="utf-8")
    log(f"Materialized env file at {target}", step=True)
    return target

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
    if not secret_path or "/" not in secret_path:
        raise RuntimeError(f"vault secret_path must look like 'kv/namespace/…' not '{secret_path}'")
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
    if "env" in kv and len(kv) == 1:
        return kv["env"]
    lines = []
    for k, v in kv.items():
        if re.search(r"\s", k):
            raise RuntimeError(f"Invalid env key with whitespace: {k}")
        lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"

def get_env_for_stack(project_name: str, stack_name: str, stack_dir: Path, env_spec: Optional[dict]) -> Optional[Path]:
    if not env_spec:
        return None

    def _finish(content: str) -> Path:
        cache_path = write_env_cache(project_name, stack_name, content)
        mat_name = env_spec.get("materialize")
        if mat_name:
            write_materialized_env(stack_dir, mat_name, content)
        return cache_path

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
        return _finish(content)

    if backend == "vault":
        secret_path = (env_spec.get("secret_path") or "").strip()
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
            return _finish(str(kv[data_key]))
        return _finish(build_env_blob_from_map(kv))

    raise RuntimeError(f"[{project_name}/{stack_name}] unsupported env backend: {backend}")

# ---------- Vault health / optional auto-unseal

class VaultSealedError(RuntimeError): pass

def _vault_addr() -> str:
    return os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200").rstrip("/")

def vault_health(timeout: int = 5) -> dict:
    import urllib.request, urllib.error, json as _json
    url = f"{_vault_addr()}/v1/sys/health"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            data = _json.loads(body) if body else {}
            return data if isinstance(data, dict) else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        try:
            return json.loads(body) if body else {"sealed": True}
        except Exception:
            return {"sealed": True}
    except Exception:
        return {}

def vault_try_unseal(unseal_key: str, timeout: int = 5) -> bool:
    if not unseal_key:
        return False
    import urllib.request, urllib.error
    url = f"{_vault_addr()}/v1/sys/unseal"
    payload = json.dumps({"key": unseal_key}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8","ignore")
            data = json.loads(body) if body else {}
            return bool(data and data.get("sealed") is False)
    except Exception:
        return False

def wait_for_vault_if_required(stacks_cfg: list, discord_webhook: str, email_cfg: dict) -> bool:
    """
    Returns True when:
      - no vault-backed stacks require secrets, or
      - Vault becomes ready (sealed=false, initialized=true, standby or active)
    Returns False if Vault stayed sealed/unavailable after waiting.
    Behavior:
      - If any stack has env.backend == 'vault' and optional != true, we wait.
      - If VAULT_UNSEAL_KEY is set and Vault is sealed, we attempt an unseal once per cycle.
      - Wait window controlled by VAULT_WAIT_TIMEOUT (default 120s).
    """
    need_vault = False
    for s in stacks_cfg or []:
        env = (s.get("env") or {})
        if str(env.get("backend","")).lower() == "vault" and not as_bool(env.get("optional", False)):
            need_vault = True
            break
    if not need_vault:
        return True

    total = int(os.environ.get("VAULT_WAIT_TIMEOUT", "120"))
    start = time.time()
    warned = False
    unseal_tried = False
    while time.time() - start < total:
        h = vault_health()
        sealed = bool(h.get("sealed", False))
        initialized = h.get("initialized", True)
        if initialized and not sealed:
            if warned:
                msg = f"Vault is ready on {host()} at {now()}"
                log(msg)
                try: send_discord(discord_webhook, msg)
                except Exception: pass
                try: send_email(email_cfg, "[homelab] vault ready", msg)
                except Exception: pass
            return True
        if not warned:
            msg = f"Vault is not ready (sealed={sealed}, initialized={initialized})… waiting"
            log(msg)
            try: send_discord(discord_webhook, msg)
            except Exception: pass
            warned = True
        if sealed and not unseal_tried:
            key = os.environ.get("VAULT_UNSEAL_KEY", "")
            if key:
                if vault_try_unseal(key):
                    log("Attempted auto-unseal… success", step=True)
                    time.sleep(1.0)
                    continue
                else:
                    log("Attempted auto-unseal… failed", step=True)
                unseal_tried = True
        time.sleep(3.0)

    msg = f"Vault still not ready after {int(time.time()-start)}s… skipping deploy cycle for vault-required stacks"
    log(msg)
    try: send_discord(discord_webhook, msg)
    except Exception: pass
    try: send_email(email_cfg, "[homelab] vault not ready", msg)
    except Exception: pass
    return False

# ---------- deployment state (persist previous stacks)

def _state_root() -> Path:
    p = default_base_dir() / "state"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _state_file_for(project: str) -> Path:
    return _state_root() / f"{safe_name(project)}_stacks.json"

def _norm_key(stack_dir: Path, compose_file: str) -> str:
    return str((stack_dir.resolve() / compose_file).as_posix()).lower()

def _parse_env_for_project_name(env_path: Optional[Path], default_name: str) -> str:
    try:
        if env_path and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if not line or line.strip().startswith("#"):
                    continue
                if line.startswith("COMPOSE_PROJECT_NAME="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return default_name

def load_prev_stacks_state(project: str) -> Dict[str, dict]:
    sf = _state_file_for(project)
    if not sf.exists():
        return {}
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}

def save_current_stacks_state(project: str, items: List[dict]) -> None:
    out = {}
    for it in items:
        key = _norm_key(Path(it["dir"]), it["compose"])
        out[key] = it
    _state_file_for(project).write_text(json.dumps(out, indent=2), encoding="utf-8")

def compose_down(
    stack_dir: Path,
    compose_file: str,
    project_name: Optional[str],
    env_file: Optional[Path] = None,
    remove_volumes: bool = False,
) -> None:
    cli = docker_cli_name()
    base = f'{cli} -f "{compose_file}"'
    if project_name:
        base = f'{base} -p "{project_name}"'
    elif env_file and env_file.exists():
        base = f'{base} --env-file "{env_file}"'
    cmd = f"{base} down --remove-orphans"
    if remove_volumes:
        cmd = f"{cmd} -v"
    log(f"Compose down in {stack_dir} for project '{project_name or stack_dir.name}'", step=True)
    run(cmd, cwd=stack_dir)

# ---------- repo stacks loader

REPO_STACKS_CANDIDATES = [
    ".homelab/stacks.yaml",
    ".homelab/stacks.yml",
    "stacks.yaml",
    "stacks.yml",
]

def find_repo_stacks_file(repo_root: Path) -> Optional[Path]:
    override = os.environ.get("HOMELAB_STACKS_FILE", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = repo_root / override
        if p.exists():
            return p
    for rel in REPO_STACKS_CANDIDATES:
        p = repo_root / rel
        if p.exists():
            return p
    return None

def load_repo_stacks(repo_root: Path) -> Optional[List[dict]]:
    stacks_path = find_repo_stacks_file(repo_root)
    if not stacks_path:
        return None
    try:
        data = yaml.safe_load(stacks_path.read_text(encoding="utf-8")) or {}
        data = interpolate_env(data)
        stacks = data.get("stacks")
        if isinstance(stacks, list):
            norm: List[dict] = []
            for s in stacks:
                if not isinstance(s, dict):
                    continue
                norm.append({
                    "dir": s.get("dir", "."),
                    "compose": s.get("compose", ""),
                    "env": s.get("env"),
                    "project_name": s.get("project_name"),
                })
            return norm
        if isinstance(data, list):
            return data
    except Exception as e:
        log(f"Failed to load repo stacks file: {e}")
    return None

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

                # sync repo copy
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

                    # choose stacks source: repo file → watcher.yaml → auto
                    stacks_cfg_repo = load_repo_stacks(deploy_path)
                    stacks_cfg_yaml = proj.get("stacks", [])

                    stacks_list: List[Tuple[Path, str, Optional[dict]]] = []
                    chosen = None
                    if stacks_cfg_repo:
                        chosen = "repo stacks file"
                        for s in stacks_cfg_repo:
                            d = deploy_path / s.get("dir", ".")
                            fname = resolve_compose_file(d, s.get("compose", ""))
                            if fname:
                                env_spec = s.get("env") or {}
                                if s.get("project_name"):
                                    env_spec = dict(env_spec)
                                    env_spec["project_name"] = s["project_name"]
                                stacks_list.append((d, fname, env_spec or None))
                    elif stacks_cfg_yaml:
                        chosen = "watcher.yaml stacks"
                        for s in stacks_cfg_yaml:
                            d = deploy_path / s.get("dir", ".")
                            fname = resolve_compose_file(d, s.get("compose", ""))
                            if fname:
                                stacks_list.append((d, fname, s.get("env")))
                    else:
                        chosen = "auto-discovery"
                        for d, fname in discover_stacks(deploy_path, max_depth=int(os.environ.get("DISCOVERY_DEPTH", 2))):
                            stacks_list.append((d, fname, None))

                    log(f"[{name}] using {chosen}", step=True)

                    # If any required Vault env is needed, wait (and try auto-unseal if configured)
                    probe_cfg = [{"env": env_spec or {}} for _, _, env_spec in stacks_list]
                    if not wait_for_vault_if_required(probe_cfg, discord_webhook, email_cfg):
                        # Skip this cycle entirely if a required vault secret cannot be read yet
                        last_seen[name] = remote or local
                        pending.pop(name, None)
                        continue

                    # Build desired_items (also resolve env; optional vault env skipped when sealed)
                    desired_items: List[dict] = []
                    h = vault_health()
                    sealed_now = bool(h.get("sealed", False))

                    for d, fname, env_spec in stacks_list:
                        env_cache = None
                        mat_name = None
                        explicit_pn = None

                        if isinstance(env_spec, dict):
                            explicit_pn = env_spec.get("project_name") or None
                            if str(env_spec.get("backend","")).lower() == "vault" and sealed_now and as_bool(env_spec.get("optional", False)):
                                log(f"[{name}/{d.name}] vault is sealed and env is optional… skipping env fetch this cycle", step=True)
                            else:
                                try:
                                    env_cache = get_env_for_stack(name, d.name or "stack", d, env_spec)
                                    mat_name = env_spec.get("materialize")
                                except Exception as e:
                                    if str(env_spec.get("backend","")).lower() == "vault" and "Vault is sealed" in str(e):
                                        raise VaultSealedError("vault is sealed") from e
                                    if not as_bool(env_spec.get("optional", False)):
                                        raise
                                    log(f"[{name}/{d.name}] optional env fetch failed… {e}", step=True)

                        proj_name_eff = explicit_pn or _parse_env_for_project_name(env_cache, d.name)
                        desired_items.append({
                            "dir": str(d.resolve()),
                            "compose": fname,
                            "project_name": proj_name_eff,
                            "env_cache": str(env_cache) if env_cache else None,
                            "materialized_name": mat_name,
                        })

                    # Load previous, compute removals BEFORE new ups
                    prev = load_prev_stacks_state(name)
                    prev_keys = set(prev.keys())
                    curr_keys = set(_norm_key(Path(it["dir"]), it["compose"]) for it in desired_items)
                    removed_keys = prev_keys - curr_keys

                    if removed_keys:
                        log(f"[{name}] stacks removed since last deploy: {len(removed_keys)}", step=True)
                        remove_vols = as_bool(os.environ.get("DOWN_REMOVE_VOLUMES", "false"))
                        for k in sorted(removed_keys):
                            it = prev[k]
                            try:
                                compose_down(
                                    stack_dir=Path(it["dir"]),
                                    compose_file=it["compose"],
                                    project_name=it.get("project_name"),
                                    env_file=Path(it["env_cache"]) if it.get("env_cache") else None,
                                    remove_volumes=remove_vols,
                                )
                                send_discord(discord_webhook, f"[{name}] removed stack {it['dir']} ({it['compose']}) on {host()} at {now()}")
                            except Exception as e:
                                msg = f"[{name}] failed to remove old stack {it['dir']} ({it['compose']}): {e}"
                                log(msg)
                                try: send_discord(discord_webhook, msg[:1900])
                                except Exception: pass
                                try: send_email(email_cfg, f"[homelab] {name} remove old stack failed", msg)
                                except Exception: pass

                    # Deploy current desired
                    if desired_items:
                        for d, fname, env_spec in stacks_list:
                            log(f"[{name}] stack: {d} / {fname}", step=True)
                            # Find the env_cache we already computed for that stack to avoid re-fetch
                            key = _norm_key(d, fname)
                            env_cache = None
                            for it in desired_items:
                                if _norm_key(Path(it["dir"]), it["compose"]) == key:
                                    env_cache = Path(it["env_cache"]) if it.get("env_cache") else None
                                    break
                            compose_up_with_retry(
                                d,
                                fname,
                                env_cache,
                                project_name=name,
                                stack_name=d.name or "stack",
                                discord_webhook=discord_webhook,
                                email_cfg=email_cfg,
                                max_retries=5,
                            )
                    else:
                        log(f"[{name}] no compose stacks found under {deploy_path}… skipping deploy")

                    msg = f"[{name}] deployed {(remote or 'unknown')[:7]} on {host()} at {now()}… was {(local or 'unknown')[:7]}"
                    log(msg)
                    send_discord(discord_webhook, msg)
                    send_email(email_cfg, f"[homelab] {name} updated", msg)

                    # Persist desired state AFTER successful cycle
                    save_current_stacks_state(name, desired_items)

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
