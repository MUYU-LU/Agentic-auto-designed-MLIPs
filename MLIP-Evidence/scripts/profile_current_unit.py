#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CAPABILITY_PATTERNS = {
    "absolute-positions": ["atoms.positions", "positions.reshape", "positions.view", "pos.reshape", "pos.view"],
    "relative-geometry": ["positions[:, none", "positions[none, :", "rij", "distance", "norm(", "pair_vec"],
    "cutoff-locality": ["cutoff", "neighbor", "neighbour", "radius", "radius_graph"],
    "message-passing": ["message", "interaction", "aggregate", "scatter", "propagate"],
    "angular-triplets": ["triplet", "angle", "theta", "three-body", "many-body"],
    "vector-features": ["vector", "direction", "equivariant", "irreps", "spherical"],
    "equivariance": ["equivariant", "e3nn", "irreps", "tensor product", "o3."],
    "force-from-autograd": ["autograd.grad", "create_graph=true", "get_forces"],
    "atomref": ["atomref", "species_bias", "reference energy", "atomic energy bias", "element_bias"],
    "long-range-head": ["electrostatics", "ewald", "long-range", "charge", "dipole", "coulomb"],
    "batching": ["batch_size", "dataloader", "collate", "batch_idx", "batch["],
    "multi-dataset": ["rmd17", "iso17", "dataset both", "mixed", "sample_ratio"],
    "uncertainty": ["ensemble", "variance", "uncertainty", "calibration"],
}

UNIT_FILES = [
    "config.json",
    "config.yaml",
    "model/model.py",
    "model/train.py",
    "model/eval.py",
    "model/dataloader.py",
    "benchmark/metrics.py",
    "main.py",
]

STATUS_FILES = [
    "unit_meta.json",
    "implementation_status.json",
    "run_status.json",
]


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_unreadable_json": str(path)}


def has_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(pattern.lower() in lower for pattern in patterns)


def infer_runtime_root(unit_root: Path) -> Path | None:
    if unit_root.name in {"base_unit", "seed_unit"}:
        return unit_root.parent
    parts = unit_root.parts
    if "generations" in parts:
        idx = parts.index("generations")
        return Path(*parts[:idx])
    return None


def read_metrics_and_history(unit_root: Path) -> dict:
    outputs = unit_root / "outputs"
    result: dict[str, Any] = {"datasets": {}, "top_level_train_history": None}
    if not outputs.exists():
        return result
    top_history = outputs / "train_history.json"
    if top_history.exists():
        result["top_level_train_history"] = load_json(top_history, {})
    for dataset_dir in sorted(p for p in outputs.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        result["datasets"][dataset] = {
            "benchmark_metrics": load_json(dataset_dir / "benchmark_metrics.json", None),
            "train_history": load_json(dataset_dir / "train_history.json", None),
        }
    return result


def read_runtime_state(runtime_root: Path | None, unit_root: Path) -> dict:
    if runtime_root is None:
        return {}
    state: dict[str, Any] = {
        "round_state": load_json(runtime_root / "ledger" / "round_state.json", {}),
        "frontier_tail": [],
        "selection_files": [],
    }
    frontier = runtime_root / "ledger" / "frontier.jsonl"
    if frontier.exists():
        rows = []
        for line in frontier.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        state["frontier_tail"] = rows
    for selection in sorted(runtime_root.glob("proposals/**/selection.json"))[-5:]:
        state["selection_files"].append({"path": str(selection), "content": load_json(selection, {})})
    generation_root = unit_root.parent if unit_root.parent.name.startswith("generation_") else None
    if generation_root is not None:
        local_selection = generation_root / "selection.json"
        if local_selection.exists():
            state["selection_files"].append({"path": str(local_selection), "content": load_json(local_selection, {})})
    return state


def infer_bottleneck(status: dict, runtime_state: dict) -> str:
    impl = status.get("implementation_status", {}) or {}
    run = status.get("run_status", {}) or {}
    impl_state = impl.get("implementation_state")
    run_state = run.get("run_state")
    if run_state == "running":
        return "running"
    if run_state in {"terminal_failure", "terminal_timeout"}:
        return "repair"
    if run_state == "terminal_abandoned" or impl_state == "abandoned":
        return "abandoned"
    if impl_state in {None, "implementation_needed"}:
        return "implementation"
    if impl_state == "implemented" and not impl.get("remote_synced"):
        return "sync"
    if impl_state in {"implemented", "remote_smoke_pending"} and not impl.get("remote_smoke_passed"):
        return "smoke"
    if impl_state == "launch_ready" and run_state in {None, "not_started", "terminal_failure", "terminal_timeout"}:
        return "launch"
    workflow = (runtime_state.get("round_state") or {}).get("workflow_state")
    return workflow or "unknown"


def profile_unit(unit_root: Path) -> dict:
    unit_root = unit_root.expanduser().resolve()
    runtime_root = infer_runtime_root(unit_root)
    files = {}
    combined_parts = []
    for relpath in UNIT_FILES:
        text = load_text(unit_root / relpath)
        if text:
            files[relpath] = {
                "bytes": len(text.encode("utf-8")),
                "lines": text.count("\n") + 1,
            }
            combined_parts.append(f"\n# {relpath}\n{text}")

    combined = "\n".join(combined_parts)
    capabilities = {name: has_any(combined, patterns) for name, patterns in CAPABILITY_PATTERNS.items()}
    present = [name for name, enabled in capabilities.items() if enabled]
    missing = [name for name, enabled in capabilities.items() if not enabled]

    status = {relpath.removesuffix(".json"): load_json(unit_root / relpath, {}) for relpath in STATUS_FILES}
    metrics = read_metrics_and_history(unit_root)
    runtime_state = read_runtime_state(runtime_root, unit_root)
    current_bottleneck = infer_bottleneck(status, runtime_state)

    phase = "absolute-coordinate baseline"
    if capabilities["relative-geometry"]:
        phase = "relative geometry"
    if capabilities["message-passing"]:
        phase = "local message passing"
    if capabilities["equivariance"]:
        phase = "equivariant local MLIP"
    if capabilities["long-range-head"]:
        phase = f"{phase} with long-range head"

    return {
        "unit_root": str(unit_root),
        "runtime_root": str(runtime_root) if runtime_root else None,
        "phase_guess": phase,
        "current_bottleneck": current_bottleneck,
        "files_read": files,
        "status_files": status,
        "metrics_and_history": metrics,
        "runtime_state": runtime_state,
        "capabilities": capabilities,
        "present": present,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the runnable MLIP unit capability and process state.")
    parser.add_argument("--unit-root", required=True, help="Path to base_unit or a generated unit directory.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    profile = profile_unit(Path(args.unit_root).expanduser().resolve())
    print(json.dumps(profile, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
