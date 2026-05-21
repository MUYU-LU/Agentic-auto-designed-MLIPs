from __future__ import annotations

import argparse
import shlex
from datetime import datetime, timezone

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    load_config,
    load_run_status,
    remote_unit_path,
    resolve_unit,
    rsync_from_remote,
    run_remote_bash,
    sync_status_files_to_remote,
    update_run_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--reason", choices=["timeout", "stalled", "repair"], default="repair")
    args = parser.parse_args()

    config = load_config()
    unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    run = load_run_status(unit_root)
    remote_path = remote_unit_path(args.unit, config)
    log_remote = f"{remote_path}/outputs/launch.log"
    log_local = unit_root / "outputs" / "launch.log"

    script = f"""
set -euo pipefail
UNIT={shlex.quote(remote_path)}
PIDFILE="$UNIT/outputs/launch.pid"
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" || true)
  if [ -n "${{PID:-}}" ]; then
    kill "$PID" 2>/dev/null || true
    sleep 2
    kill -9 "$PID" 2>/dev/null || true
  fi
fi
pkill -f "$UNIT/main.py" 2>/dev/null || true
exit 0
"""
    run_remote_bash(script, config=config)
    rsync_from_remote(log_remote, log_local, config=config)

    new_state = "terminal_timeout" if args.reason == "timeout" else "terminal_failure"
    update_run_status(
        unit_root,
        run_state=new_state,
        failure_class=args.reason,
        launch_log_local=str(log_local),
        launch_log_remote=log_remote,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        last_actor="remote_stop_unit.py",
    )
    sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)
    print(unit_root)


if __name__ == "__main__":
    main()
