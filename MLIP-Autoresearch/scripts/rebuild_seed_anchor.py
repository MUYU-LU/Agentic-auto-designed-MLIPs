from __future__ import annotations

import argparse

from runtime_common import RUNTIME_ROOT, active_q_schema, load_json, save_json

BENCHMARK = RUNTIME_ROOT / "benchmark"
BASE = RUNTIME_ROOT / "base_unit"


def metrics_for(dataset: str) -> dict:
    path = BASE / "outputs" / dataset / "benchmark_metrics.json"
    metrics = load_json(path, {})
    required = ["mixed_force_mae", "mixed_energy_mae"]
    missing = [key for key in required if metrics.get(key) is None]
    if missing:
        raise SystemExit(f"base_unit is missing {missing} in {path}")
    return {
        "mixed_force_mae": float(metrics["mixed_force_mae"]),
        "mixed_energy_mae": float(metrics["mixed_energy_mae"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    anchors_path = BENCHMARK / "anchors.json"
    anchors = load_json(anchors_path, {})
    seed = anchors.setdefault("seed_anchor", {})

    desired = {dataset: metrics_for(dataset) for dataset in active_q_schema(anchors)["datasets"]}

    if not args.force:
        existing_ok = True
        for dataset in desired:
            cur = seed.get(dataset, {})
            if (
                float(cur.get("mixed_force_mae", 0.0) or 0.0) <= 0.0
                or float(cur.get("mixed_energy_mae", 0.0) or 0.0) <= 0.0
            ):
                existing_ok = False
        if existing_ok:
            print(anchors_path)
            return

    anchors["seed_anchor"] = desired
    save_json(anchors_path, anchors)
    print(anchors_path)


if __name__ == "__main__":
    main()
