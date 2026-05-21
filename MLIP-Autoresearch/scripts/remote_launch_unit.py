from __future__ import annotations

import argparse
import re
import shlex
from datetime import datetime, timezone

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    assert_launch_ready,
    load_config,
    load_run_status,
    remote_unit_path,
    resolve_unit,
    run_remote_bash,
    sync_status_files_to_remote,
    update_run_status,
)


def parse_process_free_gpu_ids(text: str) -> list[str]:
    all_ids: list[str] = []
    seen: set[str] = set()
    busy: set[str] = set()
    in_processes = False
    for raw in text.splitlines():
        line = raw.strip()
        match = re.match(r"^\|\s*(\d+)\s+", line)
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            all_ids.append(match.group(1))
        if "Processes:" in line:
            in_processes = True
            continue
        if in_processes and "No running processes found" not in line:
            process_match = re.match(r"^\|\s*(\d+)\s+", line)
            if process_match:
                busy.add(process_match.group(1))
    return [gpu_id for gpu_id in all_ids if gpu_id not in busy]


def resolve_gpu_value(requested: str | None, config: dict) -> str:
    if requested is not None:
        gpu_value = str(requested).strip()
        if not re.fullmatch(r"\d+(,\d+)*", gpu_value):
            raise SystemExit(f"Invalid --gpu value: {gpu_value!r}. Expected numeric GPU id or comma-separated numeric ids.")
        return gpu_value

    probe = run_remote_bash("nvidia-smi", config=config)
    gpu_pool = parse_process_free_gpu_ids(probe.stdout or "") if probe.returncode == 0 else []
    if not gpu_pool:
        raise SystemExit("No process-free remote GPU found. Pass --gpu explicitly or wait for a GPU to become free.")
    return gpu_pool[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--dataset", choices=["rmd17", "iso17", "mad10k", "both", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()

    config = load_config()
    policies = config.get("policies", {})
    epochs = int(args.epochs if args.epochs is not None else policies.get("default_batch_launch_epochs", 8))
    batch_size = int(args.batch_size if args.batch_size is not None else policies.get("default_batch_launch_batch_size", 8))

    unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    assert_launch_ready(unit_root)
    current = load_run_status(unit_root)

    remote_python = f"{config.get('remote_conda_env', '').rstrip('/')}/bin/python"
    if not config.get("remote_conda_env"):
        raise SystemExit("Missing remote_conda_env in config.json.")

    remote_path = remote_unit_path(args.unit, config)
    log_remote = f"{remote_path}/outputs/launch.log"
    pid_remote = f"{remote_path}/outputs/launch.pid"
    gpu_value = resolve_gpu_value(args.gpu, config)
    gpu_export = f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_value)}"
    max_samples = f" --max-samples {args.max_samples}" if args.max_samples is not None else ""

    script = f"""
set -euo pipefail
UNIT={shlex.quote(remote_path)}
PY={shlex.quote(remote_python)}
test -d "$UNIT"
test -f "$UNIT/main.py"
mkdir -p "$UNIT/outputs"
cd "$UNIT"
python3 - <<'PYCHK'
import json
import pathlib
root = pathlib.Path.cwd()
cfg = json.loads((root / 'config.json').read_text())
bench = (root / cfg['benchmark_root']).resolve()
if not bench.exists():
    raise SystemExit(f'benchmark_root missing: {{bench}}')
PYCHK
rm -f {shlex.quote(pid_remote)}
{gpu_export}
(
  echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"
  exec "$PY" "$UNIT/main.py" --dataset {shlex.quote(args.dataset)} --epochs {epochs} --batch-size {batch_size}{max_samples}
) > {shlex.quote(log_remote)} 2>&1 &
echo $! | tee {shlex.quote(pid_remote)}
"""
    result = run_remote_bash(script, config=config)
    if result.returncode != 0:
        update_run_status(
            unit_root,
            run_state="terminal_failure",
            failure_class=f"remote_launch_exit_{result.returncode}",
            launch_log_remote=log_remote,
            last_actor="remote_launch_unit.py",
        )
        sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)
        raise SystemExit(result.returncode)

    pid = (result.stdout or "").strip().splitlines()[-1] if (result.stdout or "").strip() else None
    update_run_status(
        unit_root,
        run_state="running",
        launch_count=int(current.get("launch_count", 0) or 0) + 1,
        pid=pid,
        host=config["remote"]["host"],
        launch_log_local=None,
        launch_log_remote=log_remote,
        failure_class=None,
        launched_at_utc=datetime.now(timezone.utc).isoformat(),
        finished_at_utc=None,
        last_actor="remote_launch_unit.py",
    )
    sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)
    print(pid or "")


if __name__ == "__main__":
    main()
