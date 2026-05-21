from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_common import (
    FRONTIER,
    GENERATIONS,
    LEDGER,
    RUNTIME_ROOT,
    active_q_schema,
    load_json,
    now_utc,
    q_from_metrics,
    q_fields_for_unit,
    q_total,
    resolve_unit,
    save_json,
)


def parse_proposal_metadata(unit_meta: dict) -> dict:
    return {
        "family": unit_meta.get("family"),
        "phase": unit_meta.get("phase"),
        "jump_type": unit_meta.get("jump_type"),
        "budget_class": unit_meta.get("budget_class"),
        "expected_capability_gain": unit_meta.get("expected_capability_gain", []),
        "proposal_file": unit_meta.get("proposal_file"),
        "control_replicate": bool(unit_meta.get("control_replicate", False)),
    }


def summarize_history(history: list[dict]) -> dict:
    if not history:
        return {
            "epochs": 0,
            "last_epoch": None,
            "best_val_force_mae": None,
            "best_val_energy_mae": None,
            "force_trend": "unknown",
            "energy_trend": "unknown",
        }

    val_force = [row.get("val", {}).get("force_mae") for row in history if row.get("val", {}).get("force_mae") is not None]
    val_energy = [row.get("val", {}).get("energy_mae") for row in history if row.get("val", {}).get("energy_mae") is not None]

    def trend(seq: list[float]) -> str:
        if len(seq) < 2:
            return "flat_or_unknown"
        if seq[-1] < seq[0]:
            return "improving"
        if seq[-1] > seq[0]:
            return "worsening"
        return "flat_or_unknown"

    return {
        "epochs": len(history),
        "last_epoch": history[-1],
        "best_val_force_mae": min(val_force) if val_force else None,
        "best_val_energy_mae": min(val_energy) if val_energy else None,
        "force_trend": trend(val_force),
        "energy_trend": trend(val_energy),
    }


def dataset_payload(unit_root: Path, dataset: str, anchors: dict) -> dict:
    metrics_path = unit_root / "outputs" / dataset / "benchmark_metrics.json"
    history_path = unit_root / "outputs" / dataset / "train_history.json"

    metrics = load_json(metrics_path, {})
    history = load_json(history_path, [])
    q_ds = q_from_metrics(dataset, metrics, anchors)

    if metrics:
        metrics["Q_dataset"] = q_ds
        save_json(metrics_path, metrics)

    return {
        "metrics_path": str(metrics_path.relative_to(RUNTIME_ROOT)) if metrics_path.exists() else None,
        "history_path": str(history_path.relative_to(RUNTIME_ROOT)) if history_path.exists() else None,
        "metrics": metrics,
        "history_summary": summarize_history(history),
        "Q_dataset": q_ds,
    }


def maybe_initialize_seed_anchor() -> None:
    anchors_path = RUNTIME_ROOT / "benchmark" / "anchors.json"
    anchors = load_json(anchors_path, {})
    seed = anchors.setdefault("seed_anchor", {})
    changed = False

    for dataset in active_q_schema(anchors)["datasets"]:
        cur = seed.get(dataset, {})
        if (
            float(cur.get("mixed_force_mae", 0.0) or 0.0) > 0.0
            and float(cur.get("mixed_energy_mae", 0.0) or 0.0) > 0.0
        ):
            continue

        base_metrics = load_json(RUNTIME_ROOT / "base_unit" / "outputs" / dataset / "benchmark_metrics.json", {})
        if base_metrics.get("mixed_force_mae") is None or base_metrics.get("mixed_energy_mae") is None:
            continue

        seed[dataset] = {
            "mixed_force_mae": float(base_metrics["mixed_force_mae"]),
            "mixed_energy_mae": float(base_metrics["mixed_energy_mae"]),
        }
        changed = True

    if changed:
        save_json(anchors_path, anchors)


def write_frontier_record(record: dict) -> None:
    FRONTIER.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if FRONTIER.exists():
        for line in FRONTIER.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [row for row in rows if row.get("unit") != record.get("unit")]
    rows.append(record)
    FRONTIER.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def source_total_q(source_unit: str | None) -> float | None:
    if not source_unit:
        return None
    try:
        source_root = resolve_unit(source_unit, RUNTIME_ROOT)
    except SystemExit:
        return None
    anchors = load_json(RUNTIME_ROOT / "benchmark" / "anchors.json", {})
    return q_fields_for_unit(source_root, anchors).get("Q_total")


def unit_runtime_summary(unit_root: Path) -> dict:
    run_status = load_json(unit_root / "run_status.json", {})
    implementation_status = load_json(unit_root / "implementation_status.json", {})
    return {
        "run_state": run_status.get("run_state"),
        "failure_class": run_status.get("failure_class"),
        "launch_count": run_status.get("launch_count"),
        "retry_count": run_status.get("retry_count"),
        "implementation_state": implementation_status.get("implementation_state"),
        "remote_synced": implementation_status.get("remote_synced"),
        "remote_smoke_passed": implementation_status.get("remote_smoke_passed"),
        "repair_attempts": implementation_status.get("repair_attempts"),
        "same_failure_class_repairs": implementation_status.get("same_failure_class_repairs"),
        "last_failure_class": implementation_status.get("last_failure_class"),
    }


def build_generation_summary(generation_name: str) -> Path:
    generation_root = GENERATIONS / generation_name
    anchors = load_json(RUNTIME_ROOT / "benchmark" / "anchors.json", {})
    rows = []

    for unit_root in sorted(generation_root.glob("proposal_*")):
        unit_meta = load_json(unit_root / "unit_meta.json", {})
        row = {
            "unit": f"{generation_name}/{unit_root.name}",
            "proposal_file": unit_meta.get("proposal_file"),
            "control_replicate": bool(unit_meta.get("control_replicate", False)),
            **unit_runtime_summary(unit_root),
        }

        q_fields = q_fields_for_unit(unit_root, anchors)
        row.update(q_fields)

        for dataset in active_q_schema(anchors)["datasets"]:
            metrics = load_json(unit_root / "outputs" / dataset / "benchmark_metrics.json", {})
            history = load_json(unit_root / "outputs" / dataset / "train_history.json", [])
            row[f"{dataset}_metrics_present"] = bool(metrics)
            row[f"{dataset}_history_present"] = bool(history)
            for key, value in metrics.items():
                row[f"{dataset}_{key}"] = value
            hist = summarize_history(history)
            row[f"{dataset}_epochs"] = hist["epochs"]
            row[f"{dataset}_force_trend"] = hist["force_trend"]
            row[f"{dataset}_energy_trend"] = hist["energy_trend"]
            row[f"{dataset}_best_val_force_mae"] = hist["best_val_force_mae"]
            row[f"{dataset}_best_val_energy_mae"] = hist["best_val_energy_mae"]

        rows.append(row)

    control_rows = [r for r in rows if r.get("control_replicate")]
    best_control_q = None
    if control_rows:
        control_qs = [r.get("Q_total") for r in control_rows if r.get("Q_total") is not None]
        if control_qs:
            best_control_q = max(control_qs)

    payload = {
        "generation": generation_name,
        "created_at_utc": now_utc(),
        "units": rows,
        "best_control_Q_total": best_control_q,
    }

    out_path = LEDGER / f"{generation_name}_summary.json"
    save_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    maybe_initialize_seed_anchor()
    anchors = load_json(RUNTIME_ROOT / "benchmark" / "anchors.json", {})

    unit_root = resolve_unit(args.unit, RUNTIME_ROOT)
    unit_meta = load_json(unit_root / "unit_meta.json", {})
    proposal_meta = parse_proposal_metadata(unit_meta)

    active_schema = active_q_schema(anchors)
    datasets = {dataset: dataset_payload(unit_root, dataset, anchors) for dataset in active_schema["datasets"]}

    q_rmd17 = datasets.get("rmd17", {}).get("Q_dataset")
    q_iso17 = datasets.get("iso17", {}).get("Q_dataset")
    q_mad10k = datasets.get("mad10k", {}).get("Q_dataset")
    q_total_value = q_total(q_rmd17, q_iso17, q_mad10k, anchors)

    parent_q_total = source_total_q(unit_meta.get("source_unit"))
    g_delta = None if q_total_value is None or parent_q_total is None else float(q_total_value) - float(parent_q_total)

    summary = {
        "unit": args.unit,
        "unit_meta": unit_meta,
        "proposal_metadata": proposal_meta,
        "runtime_summary": unit_runtime_summary(unit_root),
        "datasets": datasets,
        "Q_rmd17": q_rmd17,
        "Q_iso17": q_iso17,
        "Q_mad10k": q_mad10k,
        "Q_total": q_total_value,
        "G_delta": g_delta,
        "benchmark_version": active_schema["version"],
    }

    save_json(unit_root / "outputs" / "summary.json", summary)

    generation_name = unit_meta.get("generation_round")
    generation_summary_path = None
    if generation_name:
        generation_summary_path = build_generation_summary(generation_name)
        summary["generation_summary"] = str(generation_summary_path.relative_to(RUNTIME_ROOT))
        save_json(unit_root / "outputs" / "summary.json", summary)

    if q_total_value is not None and not args.no_ledger:
        record = {
            "unit": args.unit,
            "source_unit": unit_meta.get("source_unit"),
            "family": proposal_meta.get("family"),
            "phase": proposal_meta.get("phase"),
            "jump_type": proposal_meta.get("jump_type"),
            "budget_class": proposal_meta.get("budget_class"),
            "Q_rmd17": q_rmd17,
            "Q_iso17": q_iso17,
            "Q_mad10k": q_mad10k,
            "Q_total": q_total_value,
            "G_delta": g_delta,
            "benchmark_version": active_schema["version"],
            "runtime_sec": unit_meta.get("runtime_sec"),
            "status": summary["runtime_summary"].get("run_state"),
            "capabilities": proposal_meta.get("expected_capability_gain", []),
        }
        write_frontier_record(record)
        summary["frontier_record"] = str(FRONTIER.relative_to(RUNTIME_ROOT))
        save_json(unit_root / "outputs" / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
