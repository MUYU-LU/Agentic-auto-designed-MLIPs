from __future__ import annotations

import argparse
from pathlib import Path

from runtime_common import (
    BASE_UNIT,
    GENERATIONS,
    STAGING_RUNTIME_ROOT,
    default_implementation_status,
    default_run_status,
    active_q_schema,
    infer_control_from_proposal_name,
    load_json,
    resolve_unit,
    save_json,
    unit_label,
    write_implementation_status,
    write_run_status,
)


def iter_units() -> list[Path]:
    units = []
    if BASE_UNIT.exists():
        units.append(BASE_UNIT)
    for generation_dir in sorted(GENERATIONS.glob("generation_*")):
        for proposal_dir in sorted(generation_dir.glob("proposal_*")):
            if proposal_dir.is_dir():
                units.append(proposal_dir)
    return units


def source_unit_of(unit_root: Path) -> str:
    meta = load_json(unit_root / "unit_meta.json", {})
    return meta.get("source_unit", "base_unit")


def proposal_file_of(unit_root: Path) -> str | None:
    meta = load_json(unit_root / "unit_meta.json", {})
    return meta.get("proposal_file")


def has_local_outputs(unit_root: Path) -> bool:
    return all((unit_root / "outputs" / dataset / "benchmark_metrics.json").exists() for dataset in active_q_schema()["datasets"])


def source_code_hashes(unit_root: Path, source_root: Path) -> bool:
    for rel in ["model/model.py", "model/train.py"]:
        a = unit_root / rel
        b = source_root / rel
        if a.exists() and b.exists() and a.read_bytes() != b.read_bytes():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write normalized status files instead of printing planned actions.")
    args = parser.parse_args()

    for unit_root in iter_units():
        unit_name = unit_label(unit_root, STAGING_RUNTIME_ROOT)
        source_name = source_unit_of(unit_root)
        source_root = resolve_unit(source_name, STAGING_RUNTIME_ROOT) if unit_name != "base_unit" else BASE_UNIT
        proposal_file = proposal_file_of(unit_root)
        control = infer_control_from_proposal_name(proposal_file)

        impl_path = unit_root / "implementation_status.json"
        run_path = unit_root / "run_status.json"

        if not impl_path.exists():
            has_diff = unit_name != "base_unit" and source_code_hashes(unit_root, source_root)
            payload = default_implementation_status(
                unit_root,
                source_unit=source_name,
                proposal_file=proposal_file,
                control_replicate=control,
            )
            if has_diff:
                payload["implementation_state"] = "implemented"
            if args.apply:
                write_implementation_status(unit_root, payload)
            else:
                print(f"[plan] backfill implementation_status for {unit_name}: {payload['implementation_state']}")

        if not run_path.exists():
            payload = default_run_status()
            if has_local_outputs(unit_root):
                payload["run_state"] = "terminal_success"
            if args.apply:
                write_run_status(unit_root, payload)
            else:
                print(f"[plan] backfill run_status for {unit_name}: {payload['run_state']}")


if __name__ == "__main__":
    main()
