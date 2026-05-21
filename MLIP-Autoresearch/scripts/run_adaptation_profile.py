from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from ase.io import iread


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_runtime_root() -> Path:
    # Local skill layout: MLIP-Autoresearch/scripts/<this file> -> repo root -> research_runtime.
    return Path(__file__).resolve().parents[2] / "research_runtime"


def resolve_unit(unit: str, runtime_root: Path) -> Path:
    if unit in {"base_unit", "seed_unit"}:
        root = runtime_root / unit
    elif "/" in unit:
        root = runtime_root / "generations" / unit
    else:
        raise SystemExit("invalid unit name; expected base_unit, seed_unit, or generation_###/proposal_###")
    if not root.exists():
        raise SystemExit(f"unit does not exist: {root}")
    return root


def sample_from_atoms(atoms, device: torch.device) -> dict:
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", {})
    return {
        "numbers": torch.tensor(atoms.numbers, dtype=torch.long, device=device),
        "positions": torch.tensor(atoms.positions, dtype=torch.float32, device=device),
        "energy": torch.tensor(float(results["energy"]), dtype=torch.float32, device=device),
        "forces": torch.tensor(results["forces"], dtype=torch.float32, device=device),
        "subset": atoms.info.get("subset", "unknown"),
        "n_atoms": len(atoms),
    }


def iter_split(split_dir: Path):
    for path in sorted(split_dir.glob("*.extxyz")):
        for index, atoms in enumerate(iread(path, format="extxyz", index=":")):
            yield path, index, atoms


def mae(values: list[float]) -> float | None:
    return None if not values else float(sum(abs(v) for v in values) / len(values))


def rmse(values: list[float]) -> float | None:
    return None if not values else float((sum(v * v for v in values) / len(values)) ** 0.5)


def finalize_bucket(bucket: dict) -> dict:
    return {
        "frames": bucket["frames"],
        "failed": bucket["failed"],
        "energy_mae_eV": mae(bucket["energy_errors_eV"]),
        "energy_rmse_eV": rmse(bucket["energy_errors_eV"]),
        "energy_mae_meV_per_atom": None
        if not bucket["energy_errors_meV_per_atom"]
        else mae(bucket["energy_errors_meV_per_atom"]),
        "energy_rmse_meV_per_atom": None
        if not bucket["energy_errors_meV_per_atom"]
        else rmse(bucket["energy_errors_meV_per_atom"]),
        "force_mae_eV_per_A": mae(bucket["force_component_errors_eV_per_A"]),
        "force_rmse_eV_per_A": rmse(bucket["force_component_errors_eV_per_A"]),
    }


def evaluate_model(model, split_dir: Path, device: torch.device, max_frames: int | None = None, predictions_path: Path | None = None) -> dict:
    model.eval()
    overall = defaultdict(list)
    overall["frames"] = 0
    overall["failed"] = 0
    by_subset: dict[str, dict] = {}
    rows = []

    def bucket_for(subset: str) -> dict:
        if subset not in by_subset:
            by_subset[subset] = defaultdict(list)
            by_subset[subset]["frames"] = 0
            by_subset[subset]["failed"] = 0
        return by_subset[subset]

    for frame_number, (path, index, atoms) in enumerate(iter_split(split_dir)):
        if max_frames is not None and frame_number >= max_frames:
            break
        subset = str(atoms.info.get("subset", "unknown"))
        bucket = bucket_for(subset)
        overall["frames"] += 1
        bucket["frames"] += 1
        row = {"file": str(path), "index": index, "subset": subset, "status": "ok", "error": ""}
        try:
            sample = sample_from_atoms(atoms, device)
            pred_energy, pred_forces = model(sample)
            ref_energy = sample["energy"]
            ref_forces = sample["forces"]
            energy_error = float((pred_energy.detach() - ref_energy.detach()).cpu().item())
            energy_error_mev_atom = 1000.0 * energy_error / max(int(sample["n_atoms"]), 1)
            force_errors = (pred_forces.detach() - ref_forces.detach()).reshape(-1).cpu().tolist()
            for target in (overall, bucket):
                target["energy_errors_eV"].append(energy_error)
                target["energy_errors_meV_per_atom"].append(energy_error_mev_atom)
                target["force_component_errors_eV_per_A"].extend(float(v) for v in force_errors)
            row.update(
                {
                    "n_atoms": int(sample["n_atoms"]),
                    "pred_energy_eV": float(pred_energy.detach().cpu().item()),
                    "ref_energy_eV": float(ref_energy.detach().cpu().item()),
                    "energy_error_eV": energy_error,
                    "energy_error_meV_per_atom": energy_error_mev_atom,
                }
            )
        except Exception as exc:
            overall["failed"] += 1
            bucket["failed"] += 1
            row["status"] = "failed"
            row["error"] = repr(exc)
        rows.append(row)

    if predictions_path:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "file",
                "index",
                "subset",
                "status",
                "error",
                "n_atoms",
                "pred_energy_eV",
                "ref_energy_eV",
                "energy_error_eV",
                "energy_error_meV_per_atom",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return {
        "overall": finalize_bucket(overall),
        "by_subset": {subset: finalize_bucket(bucket) for subset, bucket in sorted(by_subset.items())},
    }


def import_unit_modules(unit_root: Path):
    sys.path.insert(0, str(unit_root))
    try:
        train_module = importlib.import_module("model.train")
        model_module = importlib.import_module("model.model")
        return train_module, model_module
    finally:
        try:
            sys.path.remove(str(unit_root))
        except ValueError:
            pass


def run_native_train(train_module, profile_root: Path, output_dir: Path, epochs: int, batch_size: int, max_train_frames: int | None):
    if not hasattr(train_module, "train"):
        raise RuntimeError("model.train has no train(...) function")
    return train_module.train(
        dataset="mad10k",
        train_dir=profile_root / "train",
        val_dir=profile_root / "val",
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        max_samples=max_train_frames,
    )


def instantiate_model(model_module, device: torch.device):
    if not hasattr(model_module, "EvolutionMLIP"):
        raise RuntimeError("model.model has no EvolutionMLIP class")
    return model_module.EvolutionMLIP().to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--runtime-root", type=Path, default=default_runtime_root())
    parser.add_argument("--profile-root", type=Path, default=None)
    parser.add_argument("--profile-name", default="mad10k_adaptation")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-val-frames", type=int, default=None)
    parser.add_argument("--max-test-frames", type=int, default=None)
    args = parser.parse_args()

    runtime_root = args.runtime_root.expanduser().resolve()
    unit_root = resolve_unit(args.unit, runtime_root)
    profile_root = args.profile_root.expanduser().resolve() if args.profile_root else runtime_root / "benchmark" / "profiles" / "mad10k"
    manifest = profile_root / "split_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"missing profile manifest: {manifest}")

    output_dir = unit_root / "outputs" / "profiles" / args.profile_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    payload = {
        "schema": "adaptation_profile.v1",
        "unit": args.unit,
        "profile_name": args.profile_name,
        "profile_root": str(profile_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "trainer_mode": "unit_native_train",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_frames": args.max_train_frames,
        "max_val_frames": args.max_val_frames,
        "max_test_frames": args.max_test_frames,
        "note": "This profile measures whether the runnable unit's model/train code can adapt to MAD-10k. It is not part of main Q_total.",
    }
    save_json(summary_path, payload)

    try:
        train_module, model_module = import_unit_modules(unit_root)
        train_result = run_native_train(
            train_module,
            profile_root,
            output_dir,
            args.epochs,
            args.batch_size,
            args.max_train_frames,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = instantiate_model(model_module, device)
        model.load_state_dict(torch.load(train_result["model_path"], map_location=device))
        val = evaluate_model(model, profile_root / "val", device, max_frames=args.max_val_frames, predictions_path=output_dir / "val_predictions.csv")
        test = evaluate_model(model, profile_root / "test", device, max_frames=args.max_test_frames, predictions_path=output_dir / "test_predictions.csv")
        payload.update(
            {
                "status": "success",
                "model_path": train_result.get("model_path"),
                "history_path": train_result.get("history_path"),
                "device": str(device),
                "val": val,
                "test": test,
                "headline": {
                    "test_force_mae_eV_per_A": test["overall"]["force_mae_eV_per_A"],
                    "test_energy_mae_meV_per_atom": test["overall"]["energy_mae_meV_per_atom"],
                    "test_failure_rate": test["overall"]["failed"] / max(test["overall"]["frames"], 1),
                },
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "failure",
                "failure_class": "mad10k_adaptation_failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        save_json(summary_path, payload)
        raise SystemExit(1)

    save_json(summary_path, payload)
    print(summary_path)


if __name__ == "__main__":
    main()
