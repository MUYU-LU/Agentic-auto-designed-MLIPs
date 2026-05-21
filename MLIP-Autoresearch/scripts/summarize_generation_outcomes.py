from __future__ import annotations

import argparse
import difflib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from runtime_common import (
    LEDGER,
    RUNTIME_ROOT,
    STAGING_RUNTIME_ROOT,
    active_q_schema,
    load_json,
    now_utc,
    q_fields_for_unit,
    remote_config,
    remote_runtime_root,
    remote_target,
    run_remote_bash,
    resolve_unit,
    rsync_ssh_arg,
    save_json,
)

GENERATION_SUMMARIES = LEDGER / "generation_summaries"
GENERATION_REPORTS = LEDGER / "generation_reports"
UNIT_CARDS = LEDGER / "unit_cards"
ALL_ATTEMPTS = LEDGER / "all_attempts.jsonl"
MECHANISM_OUTCOMES = LEDGER / "mechanism_outcomes.jsonl"
NEGATIVE_PATTERNS = LEDGER / "negative_patterns.jsonl"
PARTIAL_POSITIVE_PATTERNS = LEDGER / "partial_positive_patterns.jsonl"
DIFF_FILES = ["model/model.py", "model/train.py"]


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def unit_q(unit_root: Path) -> dict:
    return q_fields_for_unit(unit_root)


def dataset_metrics(unit_root: Path, dataset: str) -> dict:
    return load_json(unit_root / "outputs" / dataset / "benchmark_metrics.json", {})


def history_summary(unit_root: Path, dataset: str) -> dict:
    history = load_json(unit_root / "outputs" / dataset / "train_history.json", [])
    if not history:
        return {"epochs": 0, "force_trend": "unknown", "energy_trend": "unknown"}
    val_force = [r.get("val", {}).get("force_mae") for r in history if r.get("val", {}).get("force_mae") is not None]
    val_energy = [r.get("val", {}).get("energy_mae") for r in history if r.get("val", {}).get("energy_mae") is not None]

    def trend(seq: list[float]) -> str:
        if len(seq) < 2:
            return "flat_or_unknown"
        return "improving" if seq[-1] < seq[0] else "worsening" if seq[-1] > seq[0] else "flat_or_unknown"

    return {
        "epochs": len(history),
        "force_trend": trend(val_force),
        "energy_trend": trend(val_energy),
        "best_val_force_mae": min(val_force) if val_force else None,
        "best_val_energy_mae": min(val_energy) if val_energy else None,
    }


def proposal_text(unit_meta: dict) -> str:
    proposal_file = unit_meta.get("proposal_file")
    if not proposal_file:
        return ""
    path = Path(str(proposal_file))
    if not path.is_absolute():
        path = RUNTIME_ROOT / path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def list_field_from_proposal(text: str, field: str) -> list[str]:
    rows: list[str] = []
    capture = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.lower().startswith(f"{field.lower()}:"):
            tail = stripped.split(":", 1)[1].strip()
            if tail:
                rows.append(tail)
            capture = True
            continue
        if capture:
            if stripped.startswith("- "):
                rows.append(stripped[2:].strip())
                continue
            if stripped and not raw.startswith((" ", "\t")):
                break
    return rows


def proposal_metadata(unit_meta: dict) -> dict:
    text = proposal_text(unit_meta)
    return {
        "proposal_file": unit_meta.get("proposal_file"),
        "family": unit_meta.get("family"),
        "phase": unit_meta.get("phase"),
        "jump_type": unit_meta.get("jump_type"),
        "budget_class": unit_meta.get("budget_class"),
        "control_replicate": bool(unit_meta.get("control_replicate", False)),
        "mechanism_refs": list_field_from_proposal(text, "mechanism_refs"),
        "negative_pattern_refs": list_field_from_proposal(text, "negative_pattern_refs"),
        "historical_relation": next(iter(list_field_from_proposal(text, "historical_relation")), None),
    }


def infer_source_unit(rows: list[dict], override: str | None) -> str | None:
    if override:
        return override
    sources = [row.get("source_unit") for row in rows if row.get("source_unit")]
    return Counter(sources).most_common(1)[0][0] if sources else None


def parent_payload(source_unit: str | None) -> dict:
    if not source_unit:
        return {}
    try:
        root = resolve_unit(source_unit, STAGING_RUNTIME_ROOT)
    except SystemExit:
        return {"unit": source_unit, "missing": True}
    q = unit_q(root)
    payload = {
        "unit": source_unit,
        "Q_rmd17": q.get("Q_rmd17"),
        "Q_iso17": q.get("Q_iso17"),
        "Q_mad10k": q.get("Q_mad10k"),
        "Q_total": q.get("Q_total"),
        "benchmark_version": q.get("benchmark_version"),
    }
    for dataset in active_q_schema()["datasets"]:
        payload[dataset] = dataset_metrics(root, dataset)
    return payload


def read_unit_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def source_root_for(unit_meta: dict) -> Path | None:
    source_unit = unit_meta.get("source_unit")
    if not source_unit:
        return None
    if source_unit == "base_unit":
        return STAGING_RUNTIME_ROOT / "base_unit"
    try:
        return resolve_unit(str(source_unit), STAGING_RUNTIME_ROOT)
    except SystemExit:
        return None


def file_delta(source_root: Path | None, unit_root: Path, relative_path: str, *, max_diff_lines: int = 120) -> dict:
    child_path = unit_root / relative_path
    source_path = source_root / relative_path if source_root else None
    payload: dict[str, Any] = {
        "path": relative_path,
        "child_exists": child_path.exists(),
        "source_exists": bool(source_path and source_path.exists()),
        "same_as_source": None,
        "added_lines": 0,
        "removed_lines": 0,
        "diff_truncated": False,
        "unified_diff_excerpt": [],
    }
    if not child_path.exists() or not source_path or not source_path.exists():
        return payload

    source_text = read_unit_text(source_path)
    child_text = read_unit_text(child_path)
    diff = list(
        difflib.unified_diff(
            source_text.splitlines(),
            child_text.splitlines(),
            fromfile=f"source/{relative_path}",
            tofile=f"child/{relative_path}",
            lineterm="",
        )
    )
    changed = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    payload["same_as_source"] = not changed
    payload["added_lines"] = sum(1 for line in changed if line.startswith("+"))
    payload["removed_lines"] = sum(1 for line in changed if line.startswith("-"))
    payload["diff_truncated"] = len(diff) > max_diff_lines
    payload["unified_diff_excerpt"] = diff[:max_diff_lines]
    return payload


def code_delta(unit_meta: dict, unit_root: Path) -> dict:
    source_root = source_root_for(unit_meta)
    files = [file_delta(source_root, unit_root, relative_path) for relative_path in DIFF_FILES]
    return {
        "source_unit": unit_meta.get("source_unit"),
        "source_root": str(source_root) if source_root else None,
        "files": files,
        "changed_files": [row["path"] for row in files if row.get("same_as_source") is False],
        "diff_note": "Diff is child runnable unit against its declared source unit, limited to model/model.py and model/train.py.",
    }


def metric_improvements(child: dict, parent: dict) -> list[dict]:
    out: list[dict] = []
    for dataset in active_q_schema()["datasets"]:
        child_metrics = child.get(dataset, {}).get("metrics", {})
        parent_metrics = parent.get(dataset, {})
        for key in ["mixed_force_mae", "mixed_energy_mae", "gap_penalty"]:
            cv = finite_float(child_metrics.get(key))
            pv = finite_float(parent_metrics.get(key))
            if cv is None or pv is None or pv <= 0:
                continue
            rel = (pv - cv) / pv
            if rel > 0.02:
                out.append({"dataset": dataset, "metric": key, "parent": pv, "child": cv, "relative_improvement": rel})
    return out


def metric_regressions(child: dict, parent: dict) -> list[dict]:
    out: list[dict] = []
    for dataset in active_q_schema()["datasets"]:
        child_metrics = child.get(dataset, {}).get("metrics", {})
        parent_metrics = parent.get(dataset, {})
        for key in ["mixed_force_mae", "mixed_energy_mae", "gap_penalty"]:
            cv = finite_float(child_metrics.get(key))
            pv = finite_float(parent_metrics.get(key))
            if cv is None or pv is None or pv <= 0:
                continue
            rel = (cv - pv) / pv
            if rel > 0.10:
                out.append({"dataset": dataset, "metric": key, "parent": pv, "child": cv, "relative_regression": rel})
    return out


def classify_outcome(row: dict, parent: dict, margin: float) -> tuple[str, list[str]]:
    if row.get("control_replicate"):
        return "control_replicate", ["unit is marked as a control replicate"]
    run_state = row.get("run_state")
    if run_state != "terminal_success":
        failure_class = str(row.get("failure_class") or "")
        if "timeout" in failure_class or run_state == "terminal_timeout":
            return "timeout_or_stalled", [f"run_state={run_state}", f"failure_class={failure_class}"]
        if run_state == "terminal_abandoned":
            return "abandoned", [f"run_state={run_state}", f"failure_class={failure_class}"]
        return "implementation_failure", [f"run_state={run_state}", f"failure_class={failure_class}"]
    child_q = finite_float(row.get("Q_total"))
    parent_q = finite_float(parent.get("Q_total"))
    if child_q is None or parent_q is None:
        return "implementation_failure", ["missing Q_total for child or parent"]
    delta_q = child_q - parent_q
    row["delta_Q_vs_parent"] = delta_q
    if delta_q > margin:
        return "frontier_win", [f"delta_Q={delta_q:.6g} exceeds margin={margin:.6g}"]
    if abs(delta_q) <= margin:
        return "neutral_variance", [f"abs(delta_Q)={abs(delta_q):.6g} within margin={margin:.6g}"]
    improvements = metric_improvements(row, parent)
    regressions = metric_regressions(row, parent)
    row["component_improvements_vs_parent"] = improvements
    row["component_regressions_vs_parent"] = regressions
    if improvements and regressions:
        return "benchmark_tradeoff", ["some component metrics improved but other tracked metrics regressed strongly"]
    if improvements:
        return "partial_positive", ["one or more component metrics improved relative to parent while Q_total lost"]
    return "negative_method", ["Q_total lost and no tracked component metric improved relative to parent"]


def jsonl_upsert(path: Path, rows: list[dict], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {row.get(key): row for row in existing if row.get(key) is not None}
    for row in rows:
        if row.get(key) is not None:
            by_key[row[key]] = row
    ordered = sorted(by_key.values(), key=lambda row: str(row.get(key)))
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n", encoding="utf-8")


def jsonl_replace_generation(path: Path, generation: str, rows: list[dict], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = []
    for row in existing:
        unit = str(row.get("unit", ""))
        row_generation = row.get("generation")
        if row_generation == generation or unit.startswith(f"{generation}/"):
            continue
        kept.append(row)
    by_key = {row.get(key): row for row in kept if row.get(key) is not None}
    for row in rows:
        if row.get(key) is not None:
            by_key[row[key]] = row
    ordered = sorted(by_key.values(), key=lambda row: str(row.get(key)))
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + ("\n" if ordered else ""), encoding="utf-8")


def unit_card(unit: str, unit_root: Path) -> dict:
    unit_meta = load_json(unit_root / "unit_meta.json", {})
    run_status = load_json(unit_root / "run_status.json", {})
    impl = load_json(unit_root / "implementation_status.json", {})
    q = unit_q(unit_root)
    active_datasets = active_q_schema()["datasets"]
    dataset_sections = {
        dataset: {"metrics": dataset_metrics(unit_root, dataset), "history_summary": history_summary(unit_root, dataset)}
        for dataset in active_datasets
    }
    return {
        "unit": unit,
        "source_unit": unit_meta.get("source_unit"),
        "proposal": proposal_metadata(unit_meta),
        "run_state": run_status.get("run_state"),
        "failure_class": run_status.get("failure_class"),
        "auxiliary_benchmark_failures": run_status.get("auxiliary_benchmark_failures", []),
        "implementation_state": impl.get("implementation_state"),
        "remote_synced": impl.get("remote_synced"),
        "remote_smoke_passed": impl.get("remote_smoke_passed"),
        "code_delta": code_delta(unit_meta, unit_root),
        "implementation_notes": impl.get("notes"),
        "Q_rmd17": q.get("Q_rmd17"),
        "Q_iso17": q.get("Q_iso17"),
        "Q_mad10k": q.get("Q_mad10k"),
        "Q_total": q.get("Q_total"),
        "benchmark_version": q.get("benchmark_version"),
        **dataset_sections,
        "created_at_utc": now_utc(),
    }


def write_report(generation: str, payload: dict) -> None:
    lines = [
        f"# {generation} outcome report", "",
        f"- source_unit: `{payload.get('source_unit')}`",
        f"- parent_Q_total: `{payload.get('parent', {}).get('Q_total')}`",
        f"- margin: `{payload.get('margin')}`",
        f"- did_any_child_beat_parent: `{payload.get('did_any_child_beat_parent')}`", "",
        "## Best child", "", "```json", json.dumps(payload.get("best_child"), indent=2, ensure_ascii=False), "```", "",
        "## Outcome counts", "", "```json", json.dumps(payload.get("outcome_counts"), indent=2, ensure_ascii=False), "```", "",
        "## Lessons", "",
    ]
    lines.extend(f"- {lesson}" for lesson in payload.get("lessons", []))
    lines.extend(["", "## Units", "", "| unit | outcome_class | run_state | Q_total | delta_Q_vs_parent | code_delta | notes |", "|---|---|---|---:|---:|---|---|"])
    for row in payload.get("units", []):
        notes = "; ".join(row.get("outcome_reasons", [])).replace("|", "/")
        delta_files = []
        for file_row in row.get("code_delta", {}).get("files", []):
            if file_row.get("same_as_source") is False:
                delta_files.append(f"{file_row.get('path')} +{file_row.get('added_lines')}/-{file_row.get('removed_lines')}")
        code_delta_text = "; ".join(delta_files) if delta_files else "no tracked code diff"
        lines.append(f"| `{row.get('unit')}` | {row.get('outcome_class')} | {row.get('run_state')} | {row.get('Q_total')} | {row.get('delta_Q_vs_parent')} | {code_delta_text} | {notes} |")
    (GENERATION_REPORTS / f"{generation}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_generation(generation: str, *, source_unit: str | None = None) -> dict:
    generation_root = RUNTIME_ROOT / "generations" / generation
    if not generation_root.exists():
        raise SystemExit(f"Missing generation directory: {generation_root}")
    cards = [unit_card(f"{generation}/{p.name}", p) for p in sorted(generation_root.glob("proposal_*")) if p.is_dir()]
    if not cards:
        raise SystemExit(f"No proposal units found for {generation}")
    source_unit = infer_source_unit(cards, source_unit)
    parent = parent_payload(source_unit)
    control_qs = [finite_float(row.get("Q_total")) for row in cards if row.get("proposal", {}).get("control_replicate")]
    control_qs = [value for value in control_qs if value is not None]
    control_sigma = statistics.pstdev(control_qs) if len(control_qs) >= 2 else None
    margin = max(0.03, 2.0 * control_sigma) if control_sigma is not None else 0.03
    for row in cards:
        row["control_replicate"] = bool(row.get("proposal", {}).get("control_replicate"))
        row["outcome_class"], row["outcome_reasons"] = classify_outcome(row, parent, margin)
    valid = [row for row in cards if row.get("run_state") == "terminal_success" and finite_float(row.get("Q_total")) is not None]
    best_child = max(valid, key=lambda row: (float(row["Q_total"]), row["unit"])) if valid else None
    parent_q = finite_float(parent.get("Q_total"))
    did_beat_parent = bool(best_child and parent_q is not None and float(best_child["Q_total"]) > parent_q + margin)
    outcome_counts = Counter(row["outcome_class"] for row in cards)
    lessons = []
    if not did_beat_parent:
        lessons.append(f"No child beat parent {source_unit}; keep parent unless a reviewed override is chosen.")
    if any(row["outcome_class"] == "partial_positive" for row in cards):
        lessons.append("Some units improved tracked component metrics but lost Q_total; salvage only with explicit guardrails.")
    if any(row["outcome_class"] == "negative_method" for row in cards):
        lessons.append("Negative methods should not be repeated unless the proposal addresses the recorded failure pattern.")
    partial_patterns = []
    negative_patterns = []
    mechanism_outcomes = []
    for row in cards:
        for ref in row.get("proposal", {}).get("mechanism_refs", []):
            mechanism_outcomes.append({"mechanism_ref": ref, "unit": row["unit"], "generation": generation, "outcome_class": row["outcome_class"], "Q_total": row.get("Q_total"), "delta_Q_vs_parent": row.get("delta_Q_vs_parent"), "updated_at_utc": now_utc()})
        if row["outcome_class"] == "partial_positive":
            partial_patterns.append({"pattern_id": f"PP-{row['unit'].replace('/', '__')}", "unit": row["unit"], "source_unit": source_unit, "positive_signal": row.get("component_improvements_vs_parent", []), "negative_signal": f"Q_total below parent by {abs(row.get('delta_Q_vs_parent', 0.0)):.6g}", "retry_policy": "may retry only with explicit guard for degraded metrics and a non-duplicate mechanism relation", "updated_at_utc": now_utc()})
        if row["outcome_class"] == "negative_method":
            negative_patterns.append({"pattern_id": f"NEG-{row['unit'].replace('/', '__')}", "unit": row["unit"], "source_unit": source_unit, "reason": "; ".join(row.get("outcome_reasons", [])), "do_not_repeat_unless": ["proposal declares historical_relation as retry, salvage, or ablation", "proposal explains why_not_duplicate", "proposal adds an explicit guard for the failed metric pattern"], "updated_at_utc": now_utc()})
    payload = {"generation": generation, "created_at_utc": now_utc(), "source_unit": source_unit, "benchmark_version": active_q_schema()["version"], "parent": {"unit": parent.get("unit"), "Q_rmd17": parent.get("Q_rmd17"), "Q_iso17": parent.get("Q_iso17"), "Q_mad10k": parent.get("Q_mad10k"), "Q_total": parent.get("Q_total")}, "control_sigma_Q_total": control_sigma, "margin": margin, "did_any_child_beat_parent": did_beat_parent, "best_child": {"unit": best_child.get("unit"), "Q_rmd17": best_child.get("Q_rmd17"), "Q_iso17": best_child.get("Q_iso17"), "Q_mad10k": best_child.get("Q_mad10k"), "Q_total": best_child.get("Q_total"), "outcome_class": best_child.get("outcome_class")} if best_child else None, "outcome_counts": dict(sorted(outcome_counts.items())), "lessons": lessons, "units": cards, "partial_positive_patterns": partial_patterns, "negative_patterns": negative_patterns}
    GENERATION_SUMMARIES.mkdir(parents=True, exist_ok=True)
    GENERATION_REPORTS.mkdir(parents=True, exist_ok=True)
    UNIT_CARDS.mkdir(parents=True, exist_ok=True)
    save_json(GENERATION_SUMMARIES / f"{generation}.json", payload)
    for row in cards:
        save_json(UNIT_CARDS / f"{row['unit'].replace('/', '__')}.json", row)
    jsonl_upsert(ALL_ATTEMPTS, cards, "unit")
    jsonl_replace_generation(MECHANISM_OUTCOMES, generation, mechanism_outcomes, "mechanism_ref")
    jsonl_replace_generation(PARTIAL_POSITIVE_PATTERNS, generation, partial_patterns, "pattern_id")
    jsonl_replace_generation(NEGATIVE_PATTERNS, generation, negative_patterns, "pattern_id")
    write_report(generation, payload)
    rebuild = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "rebuild_tree_state.py")],
        text=True,
        capture_output=True,
    )
    if rebuild.returncode != 0:
        raise SystemExit(f"Failed to rebuild tree state:\n{rebuild.stderr or rebuild.stdout}")
    return payload


def sync_to_remote(paths: list[Path]) -> None:
    remote_root = remote_runtime_root()
    remote = remote_config()
    prefix: list[str] = []
    if remote.get("password"):
        prefix.extend(["sshpass", "-p", str(remote["password"])])
    for path in paths:
        remote_path = f"{remote_root}/{path.relative_to(RUNTIME_ROOT)}"
        mkdir = run_remote_bash(f"mkdir -p {Path(remote_path).parent}")
        if mkdir.returncode != 0:
            raise SystemExit(f"Failed to create remote directory for {remote_path}:\n{mkdir.stderr}")
        cmd = [*prefix, "rsync", "-az", "-e", rsync_ssh_arg(), str(path), f"{remote_target()}:{remote_path}"]
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            raise SystemExit(f"Failed to sync {path} to remote:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed generation into long-term outcome memory.")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--source-unit", help="Optional parent/source override.")
    parser.add_argument("--sync-remote", action="store_true")
    args = parser.parse_args()
    payload = summarize_generation(args.generation, source_unit=args.source_unit)
    paths = [GENERATION_SUMMARIES / f"{args.generation}.json", GENERATION_REPORTS / f"{args.generation}.md", ALL_ATTEMPTS, MECHANISM_OUTCOMES, PARTIAL_POSITIVE_PATTERNS, NEGATIVE_PATTERNS, LEDGER / "tree_state.json", LEDGER / "lineage_stats.json"]
    paths.extend(UNIT_CARDS / f"{row['unit'].replace('/', '__')}.json" for row in payload.get("units", []))
    if args.sync_remote:
        sync_to_remote([path for path in paths if path.exists()])
    print(json.dumps({"generation": args.generation, "summary": str((GENERATION_SUMMARIES / f"{args.generation}.json").relative_to(RUNTIME_ROOT)), "report": str((GENERATION_REPORTS / f"{args.generation}.md").relative_to(RUNTIME_ROOT)), "did_any_child_beat_parent": payload.get("did_any_child_beat_parent"), "best_child": payload.get("best_child"), "synced_remote": bool(args.sync_remote)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
