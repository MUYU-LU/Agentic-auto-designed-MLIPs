from __future__ import annotations

import argparse

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    load_config,
    resolve_unit,
    sync_round_state_to_remote,
    sync_status_files_to_remote,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default=None)
    parser.add_argument("--round-state", action="store_true")
    args = parser.parse_args()

    config = load_config()
    if args.unit:
        unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
        sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)
    if args.round_state:
        sync_round_state_to_remote(config=config)
    if not args.unit and not args.round_state:
        raise SystemExit("Provide --unit and/or --round-state")
    print("ok")


if __name__ == "__main__":
    main()
