from __future__ import annotations

import argparse

from runtime_common import STAGING_RUNTIME_ROOT, resolve_unit, update_implementation_status, update_run_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--actor", default="main_agent")
    args = parser.parse_args()

    unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    update_implementation_status(
        unit_root,
        implementation_state="abandoned",
        last_failure_class=args.reason,
        remote_smoke_passed=False,
        last_actor=args.actor,
    )
    update_run_status(
        unit_root,
        run_state="terminal_abandoned",
        failure_class=args.reason,
        finished_at_utc=None,
        last_actor=args.actor,
    )
    print(unit_root)


if __name__ == "__main__":
    main()
