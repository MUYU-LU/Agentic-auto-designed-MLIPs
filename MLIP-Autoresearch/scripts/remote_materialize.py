from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from runtime_common import GENERATIONS, local_python_path, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=False, help="generation_### or generation_###/proposal_###")
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--proposal-file", default=None)
    parser.add_argument("--selection-file", default=None, help="Materialize every missing selected proposal in this selection.json.")
    parser.add_argument("--sync", action="store_true", help="Sync the materialized unit to the configured remote runtime.")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    py = str(local_python_path(load_config()))

    if args.selection_file:
        selection_path = Path(args.selection_file)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        target_generation = selection.get("target_generation")
        source_unit = args.source_path or selection.get("source_unit")
        proposal_directory = Path(selection.get("proposal_directory") or selection_path.parent)
        if not target_generation or not source_unit:
            raise SystemExit("selection file must define target_generation and source_unit")
        rc = 0
        for row in selection.get("selected_proposals", []):
            proposal_id = row.get("id")
            proposal_path = row.get("path")
            if not proposal_id:
                continue
            target = f"{target_generation}/{proposal_id}"
            if (GENERATIONS / target).exists():
                continue
            proposal_file = str((proposal_directory / proposal_path).resolve()) if proposal_path else None
            cmd = [py, str(scripts / "create_unit.py"), "--target", target, "--source-path", source_unit]
            if proposal_file:
                cmd.extend(["--proposal-file", proposal_file])
            result = subprocess.run(cmd)
            rc = max(rc, result.returncode)
            if result.returncode != 0:
                continue
            if args.sync:
                sync = subprocess.run([py, str(scripts / "remote_sync_unit.py"), "--unit", target])
                rc = max(rc, sync.returncode)
        raise SystemExit(rc)

    if not args.target:
        raise SystemExit("Provide --target or --selection-file")

    create_cmd = [py, str(scripts / "create_unit.py"), "--target", args.target]
    if args.source_path:
        create_cmd.extend(["--source-path", args.source_path])
    if args.proposal_file:
        create_cmd.extend(["--proposal-file", args.proposal_file])
    result = subprocess.run(create_cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if args.sync and "/" in args.target:
        sync = subprocess.run([py, str(scripts / "remote_sync_unit.py"), "--unit", args.target])
        raise SystemExit(sync.returncode)


if __name__ == "__main__":
    main()
