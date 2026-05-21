from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    active_q_schema,
    classify_failure_from_text,
    load_config,
    load_json,
    remote_unit_path,
    resolve_unit,
    rsync_from_remote,
    rsync_to_remote,
    sync_status_files_to_remote,
    update_implementation_status,
    update_run_status,
)


def non_active_benchmark_failures(unit_root: Path, launch_text: str) -> list[dict]:
    failures: list[dict] = []
    config_path = unit_root / "config.json"
    expected_datasets = []
    if config_path.exists():
        try:
            import json

            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(cfg.get("benchmark_datasets"), list):
                expected_datasets = [str(v) for v in cfg["benchmark_datasets"]]
        except Exception:
            expected_datasets = []

    def history_epochs(dataset: str) -> int:
        history_path = unit_root / "outputs" / dataset / "train_history.json"
        if not history_path.exists():
            return 0
        try:
            import json

            history = json.loads(history_path.read_text(encoding="utf-8"))
            return len(history) if isinstance(history, list) else 0
        except Exception:
            return 0

    mad_metrics = unit_root / "outputs" / "mad10k" / "benchmark_metrics.json"
    mad_epochs = history_epochs("mad10k")
    active_epochs = max(history_epochs("rmd17"), history_epochs("iso17"))

    if "mad10k" in expected_datasets and "Traceback" in launch_text and mad_epochs < active_epochs:
        failures.append(
            {
                "dataset": "mad10k",
                "failure_class": classify_failure_from_text(launch_text),
                "metrics_present": mad_metrics.exists(),
                "history_epochs": mad_epochs,
                "note": "MAD10K is an auxiliary benchmark for this active Q version; rmd17/iso17 may still be collectable.",
            }
        )
    return failures


def active_metric_failures(unit_root: Path, active_datasets: list[str]) -> list[dict]:
    failures: list[dict] = []
    for dataset in active_datasets:
        metrics_path = unit_root / "outputs" / dataset / "benchmark_metrics.json"
        if not metrics_path.exists():
            failures.append(
                {
                    "dataset": dataset,
                    "failure_class": "missing_active_benchmark_metrics",
                    "metrics_path": str(metrics_path),
                }
            )
            continue
        metrics = load_json(metrics_path, {})
        q_dataset = metrics.get("Q_dataset")
        try:
            q_value = float(q_dataset)
        except (TypeError, ValueError):
            q_value = math.nan
        if not math.isfinite(q_value):
            failures.append(
                {
                    "dataset": dataset,
                    "failure_class": "nonfinite_active_benchmark_metric",
                    "metrics_path": str(metrics_path),
                    "Q_dataset": q_dataset,
                    "mixed_force_mae": metrics.get("mixed_force_mae"),
                    "mixed_energy_mae": metrics.get("mixed_energy_mae"),
                    "gap_penalty": metrics.get("gap_penalty"),
                }
            )
    return failures


def mark_collect_failure(unit_root: Path, unit_name: str, *, config: dict, failure_class: str, launch_log: Path | None = None, **updates) -> None:
    update_run_status(
        unit_root,
        run_state="terminal_failure",
        failure_class=failure_class,
        launch_log_local=str(launch_log) if launch_log and launch_log.exists() else None,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        last_actor="remote_collect_unit.py",
        **updates,
    )
    update_implementation_status(unit_root, last_failure_class=failure_class, last_actor="remote_collect_unit.py")
    sync_status_files_to_remote(unit_root, unit_name, config=config, include_meta=True)


def sync_collected_metric_files_to_remote(unit_root: Path, unit_name: str, *, config: dict) -> None:
    """Copy only collect-generated metric JSONs back to the remote source tree."""
    local_outputs = unit_root / "outputs"
    if not local_outputs.exists():
        return

    files: list[Path] = []
    summary = local_outputs / "summary.json"
    if summary.exists():
        files.append(summary)
    files.extend(sorted(local_outputs.glob("*/benchmark_metrics.json")))

    if not files:
        return

    with tempfile.TemporaryDirectory(prefix="MLIP-Autoresearch_collect_sync_") as tmpdir:
        staged_outputs = Path(tmpdir) / "outputs"
        for src in files:
            rel = src.relative_to(local_outputs)
            dst = staged_outputs / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        result = rsync_to_remote(staged_outputs, f"{remote_unit_path(unit_name, config)}/outputs", config=config)
        if result.returncode != 0:
            raise SystemExit(f"Failed to sync collected metric JSONs to remote:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()

    config = load_config()
    unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    remote_path = remote_unit_path(args.unit, config)

    local_outputs = unit_root / "outputs"
    local_outputs.mkdir(parents=True, exist_ok=True)
    result = rsync_from_remote(f"{remote_path}/outputs/", local_outputs, config=config)
    if result.returncode != 0:
        mark_collect_failure(unit_root, args.unit, config=config, failure_class=f"remote_collect_exit_{result.returncode}")
        raise SystemExit(result.returncode)

    anchors = load_json(STAGING_RUNTIME_ROOT / "benchmark" / "anchors.json", {})
    active_datasets = active_q_schema(anchors)["datasets"]
    launch_log = unit_root / "outputs" / "launch.log"
    launch_text = launch_log.read_text(encoding="utf-8", errors="replace") if launch_log.exists() else ""
    auxiliary_failures = non_active_benchmark_failures(unit_root, launch_text)

    missing_active = [
        dataset
        for dataset in active_datasets
        if not (unit_root / "outputs" / dataset / "benchmark_metrics.json").exists()
    ]
    if missing_active:
        failure_class = classify_failure_from_text(launch_text) if launch_text else "missing_benchmark_metrics"
        mark_collect_failure(
            unit_root,
            args.unit,
            config=config,
            failure_class=failure_class,
            launch_log=launch_log,
            active_benchmark_failures=[
                {"dataset": dataset, "failure_class": "missing_active_benchmark_metrics"}
                for dataset in missing_active
            ],
        )
        raise SystemExit(1)

    collect_script = Path(__file__).resolve().parent / "collect_results.py"
    collect = subprocess.run([sys.executable, str(collect_script), "--unit", args.unit], text=True)
    if collect.returncode != 0:
        mark_collect_failure(unit_root, args.unit, config=config, failure_class=f"ledger_collect_exit_{collect.returncode}", launch_log=launch_log)
        raise SystemExit(collect.returncode)
    sync_collected_metric_files_to_remote(unit_root, args.unit, config=config)

    metric_failures = active_metric_failures(unit_root, active_datasets)
    summary = load_json(unit_root / "outputs" / "summary.json", {})
    q_total = summary.get("Q_total")
    try:
        q_total_value = float(q_total)
    except (TypeError, ValueError):
        q_total_value = math.nan
    if metric_failures or not math.isfinite(q_total_value):
        failure_class = "nonfinite_active_benchmark_metric"
        mark_collect_failure(
            unit_root,
            args.unit,
            config=config,
            failure_class=failure_class,
            launch_log=launch_log,
            active_benchmark_failures=metric_failures,
            auxiliary_benchmark_failures=auxiliary_failures,
        )
        raise SystemExit(1)

    update_run_status(
        unit_root,
        run_state="terminal_success",
        failure_class=None,
        active_benchmark_failures=[],
        auxiliary_benchmark_failures=auxiliary_failures,
        launch_log_local=str(launch_log) if launch_log.exists() else None,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        last_actor="remote_collect_unit.py",
    )
    sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)

    print(unit_root)


if __name__ == "__main__":
    main()
