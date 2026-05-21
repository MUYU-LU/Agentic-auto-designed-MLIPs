from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    load_config,
    load_implementation_status,
    remote_unit_path,
    resolve_unit,
    rsync_to_remote,
    run_remote_bash,
    sync_status_files_to_remote,
    update_implementation_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    config = load_config()
    unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    status = load_implementation_status(unit_root)
    if status.get("implementation_state") not in {"implemented", "launch_ready", "repairing"}:
        raise SystemExit("Unit is not ready for remote sync. Expected implementation_state in {implemented, launch_ready, repairing}.")

    remote_path = remote_unit_path(args.unit, config)
    ensure_remote = run_remote_bash(f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}", config=config)
    if ensure_remote.returncode != 0:
        raise SystemExit(ensure_remote.returncode)

    result = rsync_to_remote(unit_root, remote_path, delete=args.delete, config=config)
    if result.returncode != 0:
        update_implementation_status(unit_root, last_failure_class=f"remote_sync_exit_{result.returncode}", last_actor="remote_sync_unit.py")
        raise SystemExit(result.returncode)

    verify = run_remote_bash(
        f"test -d {shlex.quote(remote_path)} && "
        f"test -f {shlex.quote(remote_path + '/main.py')} && "
        f"test -f {shlex.quote(remote_path + '/config.json')} && echo OK",
        config=config,
    )
    if verify.returncode != 0 or "OK" not in (verify.stdout or ""):
        update_implementation_status(unit_root, last_failure_class="remote_sync_verify_failed", last_actor="remote_sync_unit.py")
        raise SystemExit(1)

    generation_remote = str(Path(remote_path).parent)
    layout_check = run_remote_bash(
        "bad=$(find "
        f"{shlex.quote(generation_remote)} "
        "-mindepth 1 -maxdepth 1 -type d -name 'generation_*' -print -quit); "
        "test -z \"$bad\" || { echo remote_layout_nested_generation:$bad; exit 2; }",
        config=config,
    )
    if layout_check.returncode != 0:
        update_implementation_status(unit_root, last_failure_class="remote_sync_layout_nested_generation", last_actor="remote_sync_unit.py")
        raise SystemExit(layout_check.returncode)

    new_state = "launch_ready" if status.get("implementation_state") == "launch_ready" else "implemented"
    new_smoke = bool(status.get("remote_smoke_passed", False)) if new_state == "launch_ready" else False
    update_implementation_status(
        unit_root,
        implementation_state=new_state,
        remote_synced=True,
        remote_smoke_passed=new_smoke,
        remote_path=remote_path,
        last_failure_class=None,
        last_actor="remote_sync_unit.py",
    )
    sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)
    print(remote_path)


if __name__ == "__main__":
    main()
