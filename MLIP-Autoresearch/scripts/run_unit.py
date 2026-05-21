from __future__ import annotations

import argparse
import socket
import subprocess
from pathlib import Path

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    assert_launch_ready,
    assert_runtime_surface,
    load_run_status,
    local_python_path,
    resolve_unit,
    update_run_status,
)


def run_checked(command: list[str], *, cwd: Path, unit_root: Path) -> None:
    current = load_run_status(unit_root)
    update_run_status(
        unit_root,
        run_state="running",
        launch_count=int(current.get("launch_count", 0) or 0) + 1,
        host=socket.gethostname(),
        failure_class=None,
        last_actor="run_unit.py",
    )
    result = subprocess.run(command, cwd=cwd)
    if result.returncode == 0:
        update_run_status(unit_root, run_state="terminal_success", failure_class=None, last_actor="run_unit.py")
    else:
        update_run_status(unit_root, run_state="terminal_failure", failure_class=f"exit_{result.returncode}", last_actor="run_unit.py")
    raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default=None)
    parser.add_argument("--generation", default=None)
    parser.add_argument("--dataset", choices=["rmd17", "iso17", "mad10k", "both", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    unit_name = args.unit or args.generation
    if not unit_name:
        raise SystemExit("Provide --unit (or legacy --generation).")

    config, runtime, _ = assert_runtime_surface(require_root=True)
    if runtime != "local":
        raise SystemExit("run_unit.py is only for runtime=local. Use remote_* scripts for runtime=remote.")

    unit_root = resolve_unit(unit_name, STAGING_RUNTIME_ROOT)
    if unit_name != "base_unit":
        assert_launch_ready(unit_root)

    main_py = unit_root / "main.py"
    py = str(local_python_path(config))
    cmd = [
        py,
        str(main_py),
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    run_checked(cmd, cwd=unit_root, unit_root=unit_root)


if __name__ == "__main__":
    main()
