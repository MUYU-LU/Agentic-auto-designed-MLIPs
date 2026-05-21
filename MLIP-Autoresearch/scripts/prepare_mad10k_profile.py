from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from runtime_common import STAGING_RUNTIME_ROOT, save_json

DEFAULT_MAD_ROOT = Path(os.environ.get("MLIP_MAD10K_RAW_DIR", "<MAD10K_RAW_DIR>"))
SUBSET_RE = re.compile(r"(?:^|\s)subset=([^\s]+)")


def iter_extxyz_frames(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            first = handle.readline()
            if not first:
                return
            if not first.strip():
                continue
            n_atoms = int(first.strip())
            comment = handle.readline()
            atom_lines = [handle.readline() for _ in range(n_atoms)]
            yield first + comment + "".join(atom_lines), comment


def frame_subset(comment: str) -> str:
    match = SUBSET_RE.search(comment)
    return match.group(1) if match else "unknown"


def count_subsets(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, comment in iter_extxyz_frames(path):
        counts[frame_subset(comment)] += 1
    return counts


def allocate_stratified(counts: Counter[str], target_total: int) -> dict[str, int]:
    total = sum(counts.values())
    if target_total >= total:
        return dict(counts)
    raw = {name: target_total * count / total for name, count in counts.items()}
    allocation = {name: int(value) for name, value in raw.items()}
    remainder = target_total - sum(allocation.values())
    ranked = sorted(raw, key=lambda name: (raw[name] - allocation[name], counts[name], name), reverse=True)
    for name in ranked[:remainder]:
        allocation[name] += 1
    return allocation


def selected_local_indices(counts: Counter[str], allocation: dict[str, int], seed: int) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = {}
    for subset, count in sorted(counts.items()):
        target = allocation.get(subset, 0)
        rng = random.Random(f"{seed}:{subset}:{count}:{target}")
        selected[subset] = set(rng.sample(range(count), min(target, count)))
    return selected


def write_stratified_train(source: Path, target: Path, selected: dict[str, set[int]]) -> Counter[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    seen: Counter[str] = Counter()
    written: Counter[str] = Counter()
    with target.open("w", encoding="utf-8") as out:
        for frame, comment in iter_extxyz_frames(source):
            subset = frame_subset(comment)
            local_idx = seen[subset]
            seen[subset] += 1
            if local_idx in selected.get(subset, set()):
                out.write(frame)
                written[subset] += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mad-root", type=Path, default=DEFAULT_MAD_ROOT)
    parser.add_argument("--output-root", type=Path, default=STAGING_RUNTIME_ROOT / "benchmark" / "profiles" / "mad10k")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mad_root = args.mad_root.expanduser()
    output_root = args.output_root.expanduser()
    train_src = mad_root / "mad-train.xyz"
    val_src = mad_root / "mad-val.xyz"
    test_src = mad_root / "mad-test.xyz"
    for path in (train_src, val_src, test_src):
        if not path.exists():
            raise SystemExit(f"missing MAD split file: {path}")

    train_dst = output_root / "train" / "mad10k-train.extxyz"
    val_dst = output_root / "val" / "mad-val.extxyz"
    test_dst = output_root / "test" / "mad-test.extxyz"
    manifest_path = output_root / "split_manifest.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path)
        return

    counts = count_subsets(train_src)
    allocation = allocate_stratified(counts, args.train_size)
    selected = selected_local_indices(counts, allocation, args.seed)
    written = write_stratified_train(train_src, train_dst, selected)
    val_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(val_src, val_dst)
    shutil.copy2(test_src, test_dst)

    val_counts = count_subsets(val_dst)
    test_counts = count_subsets(test_dst)
    manifest = {
        "schema": "mad10k_profile.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "External adaptation profile for MLIP architecture/train-code transfer. Not part of main RMD17/ISO17 Q_total.",
        "source_mad_root": str(mad_root),
        "seed": args.seed,
        "train_size": args.train_size,
        "splits": {
            "train": {"path": str(train_dst), "frames": sum(written.values()), "subset_counts": dict(sorted(written.items()))},
            "val": {"path": str(val_dst), "frames": sum(val_counts.values()), "subset_counts": dict(sorted(val_counts.items()))},
            "test": {"path": str(test_dst), "frames": sum(test_counts.values()), "subset_counts": dict(sorted(test_counts.items()))},
        },
        "selection": {
            "method": "deterministic_stratified_subset_from_mad_train",
            "source_train_counts": dict(sorted(counts.items())),
            "allocated_train_counts": dict(sorted(allocation.items())),
        },
        "metrics_policy": {
            "primary": ["test_force_mae_eV_per_A", "test_energy_mae_meV_per_atom", "failure_rate"],
            "secondary": ["val_force_mae_eV_per_A", "val_energy_mae_meV_per_atom", "subset_breakdown"],
            "stress": "record unsupported unless the unit explicitly implements stress support",
        },
    }
    save_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
