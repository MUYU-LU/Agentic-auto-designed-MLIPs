from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from runtime_common import GENERATIONS, STAGING_RUNTIME_ROOT, load_config, load_implementation_status, load_json, load_round_state, load_run_status


def parse_gpu_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        gpu_id: str | None = None
        if line.isdigit():
            gpu_id = line
        else:
            match = re.match(r"^\|\s*(\d+)\s+", line)
            if match:
                gpu_id = match.group(1)
        if gpu_id is not None and gpu_id not in seen:
            seen.add(gpu_id)
            ids.append(gpu_id)
    return ids


def parse_available_gpu_ids(text: str) -> list[str]:
    all_ids = parse_gpu_ids(text)
    busy: set[str] = set()
    in_processes = False
    for raw in text.splitlines():
        line = raw.strip()
        if "Processes:" in line:
            in_processes = True
            continue
        if not in_processes or "No running processes found" in line:
            continue
        match = re.match(r"^\|\s*(\d+)\s+", line)
        if match:
            busy.add(match.group(1))
    return [gpu_id for gpu_id in all_ids if gpu_id not in busy]


def parse_manual_gpus(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids or any(not item.isdigit() for item in ids):
        raise SystemExit(f"Invalid --gpus value: {value!r}. Expected comma-separated numeric GPU ids.")
    return ids


def selected_ids_for_generation(generation: str, explicit_selection_file: str | None) -> set[str]:
    selection_path = Path(explicit_selection_file) if explicit_selection_file else None
    if selection_path is None:
        round_state = load_round_state(STAGING_RUNTIME_ROOT)
        if round_state.get("current_generation") == generation and round_state.get("active_selection_file"):
            selection_path = Path(round_state["active_selection_file"])
    if not selection_path or not selection_path.exists():
        return set()
    selection = load_json(selection_path, {})
    if selection.get("target_generation") and selection.get("target_generation") != generation:
        raise SystemExit(
            f"selection target_generation={selection.get('target_generation')!r} does not match --generation={generation!r}"
        )
    ids = {str(row.get("id")) for row in selection.get("selected_proposals", []) if isinstance(row, dict) and row.get("id")}
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", required=True)
    parser.add_argument("--dataset", choices=["rmd17", "iso17", "mad10k", "both", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpus", default="auto", help="Comma-separated remote GPU ids or 'auto'.")
    parser.add_argument("--selection-file", default=None)
    args = parser.parse_args()

    generation_root = GENERATIONS / args.generation
    if not generation_root.exists():
        raise SystemExit(f"generation not found: {generation_root}")

    selected_ids = selected_ids_for_generation(args.generation, args.selection_file)
    if selected_ids:
        missing = sorted(pid for pid in selected_ids if not (generation_root / pid).exists())
        if missing:
            raise SystemExit(f"Selected units are not materialized: {', '.join(args.generation + '/' + pid for pid in missing)}")

    units = []
    for unit_root in sorted(generation_root.glob("proposal_*")):
        if selected_ids and unit_root.name not in selected_ids:
            continue
        impl = load_implementation_status(unit_root)
        run = load_run_status(unit_root)
        if impl.get("implementation_state") == "launch_ready" and run.get("run_state") in {"not_started", "terminal_failure", "terminal_timeout"}:
            units.append(unit_root)

    if selected_ids:
        not_ready = []
        for proposal_id in sorted(selected_ids):
            unit_root = generation_root / proposal_id
            impl = load_implementation_status(unit_root)
            run = load_run_status(unit_root)
            if impl.get("implementation_state") != "launch_ready":
                not_ready.append(f"{args.generation}/{proposal_id}: implementation_state={impl.get('implementation_state')!r}")
            elif not impl.get("remote_synced") or not impl.get("remote_smoke_passed"):
                not_ready.append(f"{args.generation}/{proposal_id}: remote gate incomplete")
            elif run.get("run_state") not in {"not_started", "terminal_failure", "terminal_timeout", "terminal_success", "terminal_abandoned"}:
                not_ready.append(f"{args.generation}/{proposal_id}: run_state={run.get('run_state')!r}")
        if not_ready:
            raise SystemExit("Selected units are not batch-ready:\n" + "\n".join(not_ready))

    if not units:
        raise SystemExit("No launch-ready units found for batch launch.")

    gpu_pool: list[str | None]
    if args.gpus == "auto":
        config = load_config()
        from runtime_common import run_remote_bash
        probe = run_remote_bash("nvidia-smi", config=config)
        gpu_pool = parse_available_gpu_ids(probe.stdout or "") if probe.returncode == 0 else []
        if not gpu_pool:
            raise SystemExit("Auto GPU discovery did not find process-free numeric GPU ids. Pass --gpus explicitly.")
    else:
        gpu_pool = parse_manual_gpus(args.gpus)

    script_path = Path(__file__).resolve().parent / "remote_launch_unit.py"
    rc = 0
    for idx, unit_root in enumerate(units):
        unit_name = str(unit_root.relative_to(GENERATIONS))
        gpu = gpu_pool[idx % len(gpu_pool)]
        cmd = [sys.executable, str(script_path), "--unit", unit_name, "--dataset", args.dataset]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        if gpu is not None:
            cmd += ["--gpu", str(gpu)]
        result = subprocess.run(cmd)
        rc = max(rc, result.returncode)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
