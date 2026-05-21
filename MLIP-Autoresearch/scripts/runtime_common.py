from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

GENERATION_RE = re.compile(r"^generation_\d+$")
RUNNABLE_UNIT_RE = re.compile(r"^generation_\d+/proposal_\d+$")

IMPLEMENTATION_STATES = {
    "implementation_needed",
    "implemented",
    "remote_smoke_pending",
    "launch_ready",
    "repairing",
    "abandoned",
}

RUN_STATES = {
    "not_started",
    "running",
    "terminal_success",
    "terminal_failure",
    "terminal_timeout",
    "terminal_abandoned",
}

EPS = 1e-12
DEFAULT_Q_VERSION = "rmd17_iso17_v1"
DEFAULT_Q_DATASETS = ("rmd17", "iso17")
DEFAULT_Q_WEIGHTS = {"rmd17": 0.65, "iso17": 0.35}


def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict:
    return load_json(CONFIG_PATH, {})


def _candidate_runtime_roots(config: dict) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("RESEARCH_RUNTIME_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    configured = config.get("staging_runtime_root") or config.get("local_staging_runtime_root")
    if configured:
        path = Path(configured)
        roots.append(path if path.is_absolute() else (ROOT / configured).resolve())
    configured_runtime = config.get("runtime_root", "../research_runtime")
    roots.append((ROOT / configured_runtime).resolve())
    roots.append((ROOT.parent / "research_runtime").resolve())
    roots.append((ROOT.parent.parent / "research_runtime").resolve())
    # de-duplicate preserving order
    out = []
    seen = set()
    for p in roots:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def staging_runtime_root(config: dict | None = None) -> Path:
    config = config or load_config()
    candidates = _candidate_runtime_roots(config)
    for path in candidates:
        if path.exists():
            return path
    # if none exists, return the configured fallback for debugging/error text
    return candidates[0]


CONFIG = load_config()
STAGING_RUNTIME_ROOT = staging_runtime_root(CONFIG)
RUNTIME_ROOT = STAGING_RUNTIME_ROOT  # compatibility alias
GENERATIONS = STAGING_RUNTIME_ROOT / "generations"
BASE_UNIT = STAGING_RUNTIME_ROOT / "base_unit"
SEED_UNIT = STAGING_RUNTIME_ROOT / "seed_unit"
LEDGER = STAGING_RUNTIME_ROOT / "ledger"
FRONTIER = LEDGER / "frontier.jsonl"
PROPOSALS = STAGING_RUNTIME_ROOT / "proposals"
ROUND_STATE = LEDGER / "round_state.json"


def runtime_mode(config: dict | None = None) -> str:
    config = config or load_config()
    return os.environ.get("MLIP_AUTORESEARCH_RUNTIME", config.get("runtime", "remote"))


def assert_runtime_surface(*, require_root: bool = True) -> tuple[dict, str, Path]:
    config = load_config()
    runtime = runtime_mode(config)
    root = staging_runtime_root(config)

    if runtime not in {"local", "remote"}:
        raise SystemExit("Invalid runtime. Set config.json field 'runtime' or MLIP_AUTORESEARCH_RUNTIME to 'local' or 'remote'.")

    if require_root and not root.exists():
        raise SystemExit(
            "Staging runtime root is not visible on the current filesystem.\n"
            f"Resolved staging root: {root}\n"
            "Set RESEARCH_RUNTIME_ROOT explicitly or fix runtime_common root resolution."
        )
    return config, runtime, root


def assert_staging_shape(root: Path | None = None) -> None:
    root = root or STAGING_RUNTIME_ROOT
    expected = [root / "benchmark", root / "ledger", root / "base_unit"]
    existing = [p for p in expected if p.exists()]
    if len(existing) < 2:
        raise SystemExit(
            "Resolved staging runtime root does not look like a research_runtime.\n"
            f"Resolved root: {root}\n"
            f"Existing markers: {[str(p) for p in existing]}"
        )


def local_python_path(config: dict | None = None) -> str:
    config = config or load_config()
    env_python = os.environ.get("MLIP_AUTORESEARCH_LOCAL_PYTHON")
    if env_python:
        return env_python
    path = str(config.get("local_python", "")).strip()
    if not path:
        raise SystemExit("Missing local_python in config.json.")
    return str(Path(path).expanduser())


def local_python_prelude(config: dict | None = None) -> str:
    config = config or load_config()
    local_python = local_python_path(config)
    conda_sh = str(config.get("conda_sh", "")).strip()
    conda_env = str(config.get("conda_env", "")).strip()
    if local_python:
        return (
            f'echo "[local] which python: {local_python}"; '
            f'"{local_python}" -V; '
            f'"{local_python}" -c "import sys; print(\"sys.executable=\", sys.executable)"'
        )
    if conda_sh and conda_env:
        return (
            f'source "{conda_sh}"; conda activate "{conda_env}"; '
            'echo "[local] which python: $(which python)"; python -V; '
            'python -c "import sys; print(\"sys.executable=\", sys.executable)"'
        )
    raise SystemExit("No valid local execution shell configured.")


def conda_sh_path(config: dict | None = None) -> str:
    config = config or load_config()
    return str(config.get("conda_sh", ""))


def conda_env_name(config: dict | None = None) -> str:
    config = config or load_config()
    return str(config.get("conda_env", ""))


def local_shell_diagnostics_prefix(config: dict | None = None) -> str:
    config = config or load_config()
    conda_sh = shlex.quote(conda_sh_path(config))
    conda_env = shlex.quote(conda_env_name(config))
    return (
        f"source {conda_sh} && conda activate {conda_env} && "
        "echo '[local-exec] which python:' && which python && "
        "echo '[local-exec] python -V:' && python -V && "
        "python - <<'PY'\n"
        "import sys\n"
        "print('[local-exec] sys.executable:', sys.executable)\n"
        "PY\n"
    )


def run_local_shell(command: str, *, cwd: Path | None = None, capture_output: bool = False, check: bool = False):
    config = load_config()
    prefix = local_shell_diagnostics_prefix(config)
    full = prefix + command
    return subprocess.run(
        ["bash", "-lc", full],
        cwd=cwd,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def load_json_text(text: str | None, default: Any):
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def remote_config(config: dict | None = None) -> dict:
    config = config or load_config()
    remote = config.get("remote", {})
    required = ["host", "port", "user", "workdir", "runtime_root"]
    missing = [k for k in required if not remote.get(k)]
    if missing:
        raise SystemExit(f"Missing remote config fields: {missing}")
    return remote


def remote_target(config: dict | None = None) -> str:
    remote = remote_config(config)
    return f"{remote['user']}@{remote['host']}"


def rsync_ssh_arg(config: dict | None = None) -> str:
    remote = remote_config(config)
    return f"ssh -o StrictHostKeyChecking=no -p {remote['port']}"


def ssh_command(config: dict | None = None) -> list[str]:
    config = config or load_config()
    remote = remote_config(config)
    command: list[str] = []
    password = remote.get("password")
    if password:
        command.extend(["sshpass", "-p", str(password)])
    command.extend(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            str(remote["port"]),
            remote_target(config),
        ]
    )
    return command


def run_remote_bash(script: str, *, config: dict | None = None, capture_output: bool = True):
    config = config or load_config()
    remote_command = f"bash -lc {shlex.quote(script)}"
    return subprocess.run(
        [*ssh_command(config), remote_command],
        text=True,
        capture_output=capture_output,
    )


def rsync_to_remote(local_path: Path, remote_path: str, *, delete: bool = False, config: dict | None = None):
    config = config or load_config()
    command = []
    password = remote_config(config).get("password")
    if password:
        command.extend(["sshpass", "-p", str(password)])
    command.extend(["rsync", "-az", "-e", rsync_ssh_arg(config)])
    if delete:
        command.append("--delete")
    command.extend([f"{local_path}/", f"{remote_target(config)}:{remote_path}/"])
    return subprocess.run(command, text=True, capture_output=True)


def rsync_from_remote(remote_path: str, local_path: Path, *, config: dict | None = None):
    config = config or load_config()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = []
    password = remote_config(config).get("password")
    if password:
        command.extend(["sshpass", "-p", str(password)])
    command.extend(["rsync", "-az", "-e", rsync_ssh_arg(config), f"{remote_target(config)}:{remote_path}", str(local_path)])
    return subprocess.run(command, text=True, capture_output=True)


def remote_runtime_root(config: dict | None = None) -> str:
    config = config or load_config()
    remote = remote_config(config)
    return str(Path(remote["workdir"]) / remote["runtime_root"])


def remote_unit_path(unit_name: str, config: dict | None = None) -> str:
    return str(Path(remote_runtime_root(config)) / resolve_unit_rel(unit_name))


def resolve_unit_rel(name: str) -> Path:
    if name == "base_unit":
        return Path("base_unit")
    if name == "seed_unit":
        return Path("seed_unit")
    if not RUNNABLE_UNIT_RE.fullmatch(name):
        raise SystemExit("invalid runnable unit name: expected 'base_unit', 'seed_unit', or generation_###/proposal_###")
    return Path("generations") / name


def resolve_source_unit(source: str | None, runtime_root: Path | None = None) -> Path:
    runtime_root = runtime_root or STAGING_RUNTIME_ROOT
    if not source or source == "base_unit":
        return runtime_root / "base_unit"
    if source == "seed_unit":
        return runtime_root / "seed_unit"
    candidate_path = Path(source).expanduser()
    if candidate_path.is_absolute() or any(sep in source for sep in ("/", "\\")) and not RUNNABLE_UNIT_RE.fullmatch(source):
        resolved = candidate_path.resolve()
        if not resolved.exists():
            raise SystemExit(f"source path not found: {resolved}")
        return resolved
    return resolve_unit(source, runtime_root)


def resolve_unit(name: str, runtime_root: Path | None = None) -> Path:
    runtime_root = runtime_root or STAGING_RUNTIME_ROOT
    if name == "base_unit":
        return runtime_root / "base_unit"
    if name == "seed_unit":
        return runtime_root / "seed_unit"
    if not RUNNABLE_UNIT_RE.fullmatch(name):
        raise SystemExit("invalid runnable unit name: expected 'base_unit', 'seed_unit', or generation_###/proposal_###")
    candidate = runtime_root / "generations" / name
    if candidate.exists():
        return candidate
    raise SystemExit(f"unit not found: {name}")


def unit_label(unit_root: Path, runtime_root: Path | None = None) -> str:
    runtime_root = runtime_root or STAGING_RUNTIME_ROOT
    if unit_root == runtime_root / "base_unit":
        return "base_unit"
    if unit_root == runtime_root / "seed_unit":
        return "seed_unit"
    if unit_root.is_relative_to(runtime_root / "generations"):
        return str(unit_root.relative_to(runtime_root / "generations"))
    return str(unit_root)


def infer_control_from_proposal_name(proposal_file: str | None) -> bool:
    if not proposal_file:
        return False
    lowered = proposal_file.lower()
    return "control" in lowered or "replicate" in lowered


def default_implementation_status(
    unit_root: Path,
    *,
    source_unit: str,
    proposal_file: str | None,
    control_replicate: bool,
) -> dict:
    return {
        "implementation_state": "implemented" if control_replicate else "implementation_needed",
        "source_unit": source_unit,
        "proposal_file": proposal_file,
        "control_replicate": bool(control_replicate),
        "changed_files": [],
        "repair_attempts": 0,
        "same_failure_class_repairs": 0,
        "last_failure_class": None,
        "remote_synced": False,
        "remote_smoke_passed": False,
        "remote_path": None,
        "smoke_log_local": None,
        "smoke_log_remote": None,
        "last_actor": "create_unit.py",
        "last_updated_utc": now_utc(),
    }


def default_run_status() -> dict:
    return {
        "run_state": "not_started",
        "launch_count": 0,
        "retry_count": 0,
        "pid": None,
        "host": None,
        "launch_log_local": None,
        "launch_log_remote": None,
        "failure_class": None,
        "launched_at_utc": None,
        "finished_at_utc": None,
        "last_actor": "create_unit.py",
        "last_state_change_utc": now_utc(),
    }


def implementation_status_path(unit_root: Path) -> Path:
    return unit_root / "implementation_status.json"


def run_status_path(unit_root: Path) -> Path:
    return unit_root / "run_status.json"


def round_state_path(runtime_root: Path | None = None) -> Path:
    runtime_root = runtime_root or STAGING_RUNTIME_ROOT
    return runtime_root / "ledger" / "round_state.json"


def load_implementation_status(unit_root: Path) -> dict:
    return load_json(implementation_status_path(unit_root), {})


def load_run_status(unit_root: Path) -> dict:
    return load_json(run_status_path(unit_root), {})


def load_round_state(runtime_root: Path | None = None) -> dict:
    return load_json(round_state_path(runtime_root), {})


def write_implementation_status(unit_root: Path, payload: dict) -> None:
    state = payload.get("implementation_state")
    if state not in IMPLEMENTATION_STATES:
        raise SystemExit(f"Invalid implementation_state: {state!r}")
    payload.setdefault("last_updated_utc", now_utc())
    save_json(implementation_status_path(unit_root), payload)


def update_implementation_status(unit_root: Path, **updates) -> dict:
    payload = load_implementation_status(unit_root)
    payload.update(updates)
    payload["last_updated_utc"] = now_utc()
    state = payload.get("implementation_state")
    if state not in IMPLEMENTATION_STATES:
        raise SystemExit(f"Invalid implementation_state: {state!r}")
    save_json(implementation_status_path(unit_root), payload)
    return payload


def write_run_status(unit_root: Path, payload: dict) -> None:
    state = payload.get("run_state")
    if state not in RUN_STATES:
        raise SystemExit(f"Invalid run_state: {state!r}")
    payload.setdefault("last_state_change_utc", now_utc())
    save_json(run_status_path(unit_root), payload)


def update_run_status(unit_root: Path, **updates) -> dict:
    payload = load_run_status(unit_root)
    payload.update(updates)
    payload["last_state_change_utc"] = now_utc()
    state = payload.get("run_state")
    if state not in RUN_STATES:
        raise SystemExit(f"Invalid run_state: {state!r}")
    save_json(run_status_path(unit_root), payload)
    return payload


def update_round_state(runtime_root: Path | None = None, **updates) -> dict:
    runtime_root = runtime_root or STAGING_RUNTIME_ROOT
    payload = load_round_state(runtime_root)
    payload.update(updates)
    payload["last_transition_utc"] = now_utc()
    save_json(round_state_path(runtime_root), payload)
    return payload


def assert_launch_ready(unit_root: Path) -> dict:
    status = load_implementation_status(unit_root)
    impl_state = status.get("implementation_state")
    if impl_state != "launch_ready":
        raise SystemExit(f"Unit is not launch-ready. implementation_state={impl_state!r}")
    if not status.get("remote_synced"):
        raise SystemExit("Unit is not launch-ready. remote_synced != true")
    if not status.get("remote_smoke_passed"):
        raise SystemExit("Unit is not launch-ready. remote_smoke_passed != true")
    run = load_run_status(unit_root)
    if run.get("run_state") not in {"not_started", "terminal_failure", "terminal_timeout"}:
        raise SystemExit(f"Unit is not launchable from run_state={run.get('run_state')!r}")
    return status


def sync_status_files_to_remote(unit_root: Path, unit_name: str, *, config: dict | None = None, include_meta: bool = True):
    config = config or load_config()
    remote_path = remote_unit_path(unit_name, config)
    remote_files_dir = remote_path
    files = [
        implementation_status_path(unit_root),
        run_status_path(unit_root),
    ]
    if include_meta:
        files.append(unit_root / "unit_meta.json")
    for path in files:
        if not path.exists():
            continue
        command = []
        password = remote_config(config).get("password")
        if password:
            command.extend(["sshpass", "-p", str(password)])
        command.extend(["rsync", "-az", "-e", rsync_ssh_arg(config), str(path), f"{remote_target(config)}:{remote_files_dir}/{path.name}"])
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            raise SystemExit(f"Failed to sync status file {path.name} to remote:\n{result.stderr}")


def sync_round_state_to_remote(*, config: dict | None = None):
    config = config or load_config()
    path = ROUND_STATE
    if not path.exists():
        return
    remote_path = f"{remote_runtime_root(config)}/ledger/round_state.json"
    command = []
    password = remote_config(config).get("password")
    if password:
        command.extend(["sshpass", "-p", str(password)])
    command.extend(["rsync", "-az", "-e", rsync_ssh_arg(config), str(path), f"{remote_target(config)}:{remote_path}"])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"Failed to sync round_state.json to remote:\n{result.stderr}")


def fetch_remote_json(remote_path: str, *, config: dict | None = None) -> dict | None:
    config = config or load_config()
    script = f"if [ -f {shlex.quote(remote_path)} ]; then cat {shlex.quote(remote_path)}; else echo '__MISSING__'; fi"
    result = run_remote_bash(script, config=config, capture_output=True)
    if result.returncode != 0:
        return None
    text = (result.stdout or "").strip()
    if text == "__MISSING__" or not text:
        return None
    return load_json_text(text, None)


def remote_status_paths(unit_name: str, config: dict | None = None) -> dict[str, str]:
    remote_path = remote_unit_path(unit_name, config)
    return {
        "implementation_status": f"{remote_path}/implementation_status.json",
        "run_status": f"{remote_path}/run_status.json",
    }


def classify_failure_from_text(text: str) -> str:
    lowered = text.lower()
    if "mat1 and mat2 shapes cannot be multiplied" in lowered or "shape" in lowered and "cannot be multiplied" in lowered:
        return "shape_mismatch"
    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return "oom"
    if "modulenotfounderror" in lowered:
        return "missing_dependency"
    if "does not require grad" in lowered or "does not have a grad_fn" in lowered:
        return "force_gradient_disconnected"
    if "killed" in lowered:
        return "killed"
    if "benchmark_root missing" in lowered:
        return "benchmark_root_missing"
    return "unknown_failure"


def _metric_float(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    import math
    if not math.isfinite(out):
        return None
    return out


def q_from_metrics(dataset: str, metrics: dict, anchors: dict) -> float | None:
    existing = _metric_float(metrics, "Q_dataset")
    if existing is not None:
        return existing

    mixed_force = _metric_float(metrics, "mixed_force_mae")
    mixed_energy = _metric_float(metrics, "mixed_energy_mae")
    gap = _metric_float(metrics, "gap_penalty")
    if mixed_force is None or mixed_energy is None:
        return None
    gap = 0.0 if gap is None else gap

    anchor = (anchors or {}).get("seed_anchor", {}).get(dataset, {})
    anchor_force = _metric_float(anchor, "mixed_force_mae")
    anchor_energy = _metric_float(anchor, "mixed_energy_mae")
    if not anchor_force or not anchor_energy or anchor_force <= 0.0 or anchor_energy <= 0.0:
        return None

    import math
    return (
        0.75 * math.log((anchor_force + EPS) / (mixed_force + EPS))
        + 0.25 * math.log((anchor_energy + EPS) / (mixed_energy + EPS))
        - 0.10 * gap
    )


def q_dataset(metrics: dict, dataset: str | None = None) -> float | None:
    existing = _metric_float(metrics, "Q_dataset")
    if existing is not None:
        return existing
    if dataset is None:
        return None
    anchors = load_json(RUNTIME_ROOT / "benchmark" / "anchors.json", {})
    return q_from_metrics(dataset, metrics, anchors)


def active_q_schema(anchors: dict | None = None) -> dict:
    anchors = anchors if anchors is not None else load_json(RUNTIME_ROOT / "benchmark" / "anchors.json", {})
    version = anchors.get("active_q_version") or DEFAULT_Q_VERSION
    schema = (anchors.get("q_versions") or {}).get(version, {})
    datasets = schema.get("datasets") or list(DEFAULT_Q_DATASETS)
    weights = schema.get("weights") or dict(DEFAULT_Q_WEIGHTS)
    return {"version": version, "datasets": list(datasets), "weights": {str(k): float(v) for k, v in weights.items()}}


def q_total_from_map(q_values: dict[str, float | None], anchors: dict | None = None) -> float | None:
    schema = active_q_schema(anchors)
    total = 0.0
    import math

    for dataset in schema["datasets"]:
        value = q_values.get(dataset)
        if value is None:
            return None
        value_f = float(value)
        if not math.isfinite(value_f):
            return None
        total += float(schema["weights"].get(dataset, 0.0)) * value_f
    return total


def q_total(q_rmd17: float | None, q_iso17: float | None, q_mad10k: float | None = None, anchors: dict | None = None) -> float | None:
    q_values = {"rmd17": q_rmd17, "iso17": q_iso17, "mad10k": q_mad10k}
    return q_total_from_map(q_values, anchors)


def q_fields_for_unit(unit_root: Path, anchors: dict | None = None) -> dict:
    anchors = anchors if anchors is not None else load_json(RUNTIME_ROOT / "benchmark" / "anchors.json", {})
    q_values = {}
    for dataset in active_q_schema(anchors)["datasets"]:
        q_values[dataset] = q_from_metrics(dataset, load_json(unit_root / "outputs" / dataset / "benchmark_metrics.json", {}), anchors)
    out = {
        "Q_rmd17": q_values.get("rmd17"),
        "Q_iso17": q_values.get("iso17"),
        "Q_mad10k": q_values.get("mad10k"),
        "Q_total": q_total_from_map(q_values, anchors),
        "benchmark_version": active_q_schema(anchors)["version"],
    }
    return out


def legacy_q_total(q_rmd17: float | None, q_iso17: float | None) -> float | None:
    if q_rmd17 is None or q_iso17 is None:
        return None
    import math
    qr = float(q_rmd17)
    qi = float(q_iso17)
    if not math.isfinite(qr) or not math.isfinite(qi):
        return None
    return 0.65 * qr + 0.35 * qi


def metrics_compact_dossier(dataset: str, metrics: dict) -> dict:
    if not metrics:
        return {}
    out = dict(metrics)
    out["Q_dataset"] = q_from_metrics(dataset, metrics, load_json(RUNTIME_ROOT / "benchmark" / "anchors.json", {}))
    return out
