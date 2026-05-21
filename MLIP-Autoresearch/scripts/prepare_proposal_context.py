from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

from runtime_common import (
    FRONTIER,
    PROPOSALS,
    RUNTIME_ROOT,
    BASE_UNIT,
    active_q_schema,
    load_config,
    load_json,
    resolve_unit,
    save_json,
    sync_round_state_to_remote,
    unit_label,
)

BRIEFS = RUNTIME_ROOT / "knowledge" / "briefs"
LEDGER = RUNTIME_ROOT / "ledger"
OUTCOME_SUMMARIES = LEDGER / "generation_summaries"
OUTCOME_REPORTS = LEDGER / "generation_reports"
ALL_ATTEMPTS = LEDGER / "all_attempts.jsonl"
NEGATIVE_PATTERNS = LEDGER / "negative_patterns.jsonl"
PARTIAL_POSITIVE_PATTERNS = LEDGER / "partial_positive_patterns.jsonl"
TREE_STATE = LEDGER / "tree_state.json"
LINEAGE_STATS = LEDGER / "lineage_stats.json"
EVIDENCE_RUNS = RUNTIME_ROOT / "knowledge" / "evidence_runs"
EVIDENCE_PACKAGE_FILES = [
    "evidence_quality.json",
    "evidence_provenance.json",
    "mechanism_cards.json",
    "patch_blueprints.json",
    "proposal_constraints.json",
    "current_code_profile.json",
    "benchmark_diagnosis.json",
    "generation_memory.json",
    "audit_report.json",
]


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def evidence_package_dir_from_brief(brief_path: Path, brief_text: str) -> Path | None:
    """Resolve the evidence package referenced by a rendered brief, if present."""
    candidates: list[Path] = []
    for match in re.findall(r"(/[^\s`\"']*/knowledge/evidence_runs/[^\s`\"']+)", brief_text):
        path = Path(match.rstrip(".,)"))
        if path.name.endswith(".json") or path.name.endswith(".md"):
            path = path.parent
        candidates.append(path)

    sibling = EVIDENCE_RUNS / brief_path.stem.replace("evidence_brief_", "evidence_run_")
    candidates.append(sibling)

    for path in candidates:
        if path.exists() and path.is_dir() and (path / "evidence_run.json").exists():
            return path
    return None


def load_evidence_package(package_dir: Path | None) -> dict:
    if package_dir is None:
        return {
            "status": "missing",
            "warning": "Active evidence brief does not reference a structured evidence package. Treat it as legacy brief-only evidence.",
        }
    package = {"status": "available", "package_dir": str(package_dir), "files": {}}
    for name in EVIDENCE_PACKAGE_FILES:
        path = package_dir / name
        package["files"][name] = {
            "path": str(path),
            "exists": path.exists(),
            "payload": load_json(path, {}) if path.exists() else None,
        }
    return package


def summarize_diff(old_text: str, new_text: str, max_lines: int = 20) -> list[str]:
    diff = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm=""))
    kept = []
    for line in diff:
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            kept.append(line)
        if len(kept) >= max_lines:
            break
    return kept


def extract_code_defaults(model_text: str, train_text: str) -> dict:
    """Extract visible code-level MLIP capacity/training knobs for proposal writers."""

    def first(pattern: str, text: str) -> str | None:
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else None

    model_defaults = {
        "hidden_dim": first(r"hidden_dim:\s*int\s*=\s*([^,\n)]+)", model_text),
        "num_rbf": first(r"num_rbf:\s*int\s*=\s*([^,\n)]+)", model_text),
        "cutoff": first(r"cutoff:\s*float\s*=\s*([^,\n)]+)", model_text),
        "num_interactions": first(r"num_interactions:\s*int\s*=\s*([^,\n)]+)", model_text),
        "body_radial_dim": first(r"radial_dim:\s*int\s*=\s*([^,\n)]+)", model_text),
        "body_type_dim": first(r"type_dim:\s*int\s*=\s*([^,\n)]+)", model_text),
    }
    train_defaults = {}
    for name in [
        "MODEL_HIDDEN_DIM",
        "MODEL_NUM_RBF",
        "MODEL_CUTOFF",
        "TRAIN_LEARNING_RATE",
        "TRAIN_WEIGHT_DECAY",
        "TRAIN_ENERGY_WEIGHT",
        "TRAIN_FORCE_WEIGHT",
    ]:
        train_defaults[name] = first(rf"^{name}\s*=\s*([^\n#]+)", train_text)

    return {
        "model_defaults": {k: v for k, v in model_defaults.items() if v is not None},
        "train_defaults": {k: v for k, v in train_defaults.items() if v is not None},
        "scaling_axis_policy": {
            "status": "allowed_but_bounded",
            "purpose": "Test whether the current bottleneck is capacity/resolution/depth/training budget rather than a missing mechanism.",
            "examples": [
                "hidden_dim small increase",
                "num_interactions depth increase",
                "num_rbf radial-resolution increase",
                "body/global branch rank or direction count increase",
                "bounded training-budget or optimizer-schedule adjustment",
            ],
            "limits": [
                "Do not edit config.json for MLIP-quality changes.",
                "Do not make the whole proposal set capacity-only.",
                "Pure scaling proposals must state compute/runtime risk and a falsifiable capacity-bottleneck hypothesis.",
            ],
        },
    }


def round_state_path() -> Path:
    return LEDGER / "round_state.json"


def active_evidence_for_source(source_unit: str) -> tuple[Path, dict]:
    round_state = load_json(round_state_path(), {})
    brief_raw = round_state.get("active_evidence_brief")
    evidence_source = round_state.get("evidence_for_source_unit")
    if evidence_source != source_unit:
        raise SystemExit(
            "Active evidence does not match source unit: "
            f"evidence_for_source_unit={evidence_source!r}, source_unit={source_unit!r}"
        )
    if not brief_raw:
        raise SystemExit("round_state.active_evidence_brief is required before preparing proposal context")
    brief_path = Path(brief_raw)
    if not brief_path.exists():
        raise SystemExit(f"Active evidence brief does not exist: {brief_path}")
    return brief_path, round_state


def load_frontier_tail(n: int = 12) -> list[dict]:
    if not FRONTIER.exists():
        return []
    rows = []
    for line in FRONTIER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[-n:]


def source_unit_path(unit_root: Path) -> Path:
    meta = load_json(unit_root / "unit_meta.json", {})
    source = meta.get("source_unit")
    if not source or source == "base_unit":
        return BASE_UNIT
    return resolve_unit(source, RUNTIME_ROOT)


def dataset_dossier(unit_root: Path, dataset: str) -> dict:
    metrics = load_json(unit_root / "outputs" / dataset / "benchmark_metrics.json", {})
    history = load_json(unit_root / "outputs" / dataset / "train_history.json", [])
    return {
        "metrics": metrics,
        "history": history[-3:] if history else [],
        "history_summary": {
            "epochs": len(history),
            "last_epoch": history[-1] if history else None,
            "best_val_force_mae": min([row.get("val", {}).get("force_mae") for row in history if row.get("val", {}).get("force_mae") is not None], default=None),
            "best_val_energy_mae": min([row.get("val", {}).get("energy_mae") for row in history if row.get("val", {}).get("energy_mae") is not None], default=None),
        },
    }


def unit_runtime_summary(unit_root: Path) -> dict:
    return {
        "implementation_status": load_json(unit_root / "implementation_status.json", {}),
        "run_status": load_json(unit_root / "run_status.json", {}),
        "unit_summary": load_json(unit_root / "outputs" / "summary.json", {}),
    }


def profile_dossiers(unit_root: Path) -> dict:
    profiles_root = unit_root / "outputs" / "profiles"
    if not profiles_root.exists():
        return {}
    profiles = {}
    for summary_path in sorted(profiles_root.glob("*/summary.json")):
        profiles[summary_path.parent.name] = load_json(summary_path, {})
    return profiles


def generation_summary_for(unit_root: Path) -> dict | None:
    meta = load_json(unit_root / "unit_meta.json", {})
    generation_name = meta.get("generation_round")
    if not generation_name:
        return None
    path = LEDGER / f"{generation_name}_summary.json"
    if not path.exists():
        return None
    return load_json(path, {})


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compact_metric_changes(changes: list[dict], limit: int = 8) -> list[dict]:
    compact = []
    for row in changes[:limit]:
        compact.append({
            "dataset": row.get("dataset"),
            "metric": row.get("metric"),
            "relative_improvement": row.get("relative_improvement"),
            "relative_regression": row.get("relative_regression"),
        })
    return compact


def compact_code_delta(delta: dict, *, max_files: int = 2, max_diff_lines: int = 36) -> dict:
    files = []
    for file_row in delta.get("files", [])[:max_files]:
        files.append({
            "path": file_row.get("path"),
            "same_as_source": file_row.get("same_as_source"),
            "added_lines": file_row.get("added_lines"),
            "removed_lines": file_row.get("removed_lines"),
            "diff_truncated": file_row.get("diff_truncated"),
            "unified_diff_excerpt": file_row.get("unified_diff_excerpt", [])[:max_diff_lines],
        })
    return {
        "source_unit": delta.get("source_unit"),
        "changed_files": delta.get("changed_files", []),
        "files": files,
    }


def compact_outcome_unit(row: dict) -> dict:
    return {
        "unit": row.get("unit"),
        "source_unit": row.get("source_unit"),
        "outcome_class": row.get("outcome_class"),
        "run_state": row.get("run_state"),
        "Q_rmd17": row.get("Q_rmd17"),
        "Q_iso17": row.get("Q_iso17"),
        "Q_mad10k": row.get("Q_mad10k"),
        "Q_total": row.get("Q_total"),
        "benchmark_version": row.get("benchmark_version"),
        "delta_Q_vs_parent": row.get("delta_Q_vs_parent"),
        "mechanism_refs": row.get("proposal", {}).get("mechanism_refs", []),
        "historical_relation": row.get("proposal", {}).get("historical_relation"),
        "code_delta": compact_code_delta(row.get("code_delta", {})),
        "implementation_notes": row.get("implementation_notes"),
        "improvements": compact_metric_changes(row.get("component_improvements_vs_parent", [])),
        "regressions": compact_metric_changes(row.get("component_regressions_vs_parent", [])),
        "outcome_reasons": row.get("outcome_reasons", []),
    }


def compact_generation_outcome(summary: dict) -> dict:
    units = [compact_outcome_unit(row) for row in summary.get("units", [])]
    units = sorted(units, key=lambda row: row.get("Q_total") if isinstance(row.get("Q_total"), (int, float)) else -999, reverse=True)
    return {
        "generation": summary.get("generation"),
        "source_unit": summary.get("source_unit"),
        "parent": summary.get("parent"),
        "margin": summary.get("margin"),
        "did_any_child_beat_parent": summary.get("did_any_child_beat_parent"),
        "best_child": summary.get("best_child"),
        "outcome_counts": summary.get("outcome_counts", {}),
        "lessons": summary.get("lessons", []),
        "units_by_Q_total": units,
    }


def latest_outcome_summary_for(source_unit: str, round_state: dict) -> tuple[Path | None, dict | None]:
    preferred_generation = round_state.get("last_completed_generation")
    if preferred_generation:
        preferred_path = OUTCOME_SUMMARIES / f"{preferred_generation}.json"
        if preferred_path.exists():
            payload = load_json(preferred_path, {})
            if payload.get("source_unit") == source_unit:
                return preferred_path, payload

    candidates = []
    for path in sorted(OUTCOME_SUMMARIES.glob("generation_*.json")):
        payload = load_json(path, {})
        if payload.get("source_unit") == source_unit:
            candidates.append((path, payload))
    return candidates[-1] if candidates else (None, None)


def outcome_memory_for(source_unit: str, round_state: dict) -> dict:
    summary_path, summary = latest_outcome_summary_for(source_unit, round_state)
    attempts = [row for row in load_jsonl(ALL_ATTEMPTS) if row.get("source_unit") == source_unit]
    attempts = sorted(attempts, key=lambda row: row.get("unit", ""))[-24:]
    negative_patterns = [row for row in load_jsonl(NEGATIVE_PATTERNS) if row.get("source_unit") == source_unit][-20:]
    partial_patterns = [row for row in load_jsonl(PARTIAL_POSITIVE_PATTERNS) if row.get("source_unit") == source_unit][-20:]

    report_path = None
    report_text = ""
    if summary_path is not None:
        report_path = OUTCOME_REPORTS / f"{summary_path.stem}.md"
        report_text = load_text(report_path) if report_path.exists() else ""

    return {
        "status": "available" if summary else "missing",
        "warning": None if summary else "No generation outcome memory found. Run summarize_generation_outcomes.py for the last completed generation.",
        "summary_path": str(summary_path) if summary_path else None,
        "report_path": str(report_path) if report_path else None,
        "latest_completed_generation_summary": compact_generation_outcome(summary) if summary else None,
        "recent_attempts_for_source": [compact_outcome_unit(row) for row in attempts],
        "negative_patterns_for_source": negative_patterns,
        "partial_positive_patterns_for_source": partial_patterns,
        "report_text": report_text,
    }


def tree_context_for(source_unit: str) -> dict:
    tree = load_json(TREE_STATE, {})
    lineage = load_json(LINEAGE_STATS, {})
    nodes = tree.get("nodes", {}) if isinstance(tree.get("nodes"), dict) else {}
    source_node = nodes.get(source_unit, {})
    children = [nodes[unit] for unit in source_node.get("children", []) if unit in nodes]
    children = sorted(children, key=lambda row: row.get("Q_total") if isinstance(row.get("Q_total"), (int, float)) else -999, reverse=True)
    return {
        "status": "available" if tree else "missing",
        "tree_state_path": str(TREE_STATE),
        "lineage_stats_path": str(LINEAGE_STATS),
        "best_known_unit": tree.get("best_known_unit"),
        "best_known_Q_total": tree.get("best_known_Q_total"),
        "selection_count": tree.get("selection_count"),
        "current_source_node": {
            key: source_node.get(key)
            for key in [
                "unit",
                "source_unit",
                "Q_total",
                "G_delta",
                "family",
                "phase",
                "jump_type",
                "child_count",
                "selected_as_continuation_count",
            ]
        } if source_node else None,
        "best_children_of_current_source": [
            {
                "unit": row.get("unit"),
                "Q_total": row.get("Q_total"),
                "G_delta": row.get("G_delta"),
                "family": row.get("family"),
                "phase": row.get("phase"),
                "jump_type": row.get("jump_type"),
                "outcome_class": row.get("outcome_class"),
            }
            for row in children[:8]
        ],
        "recent_selected_sources": lineage.get("recent_selected_sources", []),
        "family_stats": lineage.get("family_stats", {}),
        "jump_type_stats": lineage.get("jump_type_stats", {}),
        "phase_stats": lineage.get("phase_stats", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()

    current_root = resolve_unit(args.unit, RUNTIME_ROOT)
    current_label = unit_label(current_root, RUNTIME_ROOT)
    source_root = source_unit_path(current_root)
    source_label = unit_label(source_root, RUNTIME_ROOT)
    brief_path, round_state = active_evidence_for_source(current_label)
    brief_text = load_text(brief_path)
    evidence_package = load_evidence_package(evidence_package_dir_from_brief(brief_path, brief_text))

    current_model = load_text(current_root / "model" / "model.py")
    current_train = load_text(current_root / "model" / "train.py")
    source_model = load_text(source_root / "model" / "model.py")
    source_train = load_text(source_root / "model" / "train.py")

    model_diff = summarize_diff(source_model, current_model)
    train_diff = summarize_diff(source_train, current_train)

    frontier_tail = load_frontier_tail()
    current_runtime = unit_runtime_summary(current_root)
    source_runtime = unit_runtime_summary(source_root)
    generation_summary = generation_summary_for(current_root)
    outcome_memory = outcome_memory_for(current_label, round_state)
    tree_context = tree_context_for(current_label)
    code_defaults = extract_code_defaults(current_model, current_train)

    lines = []
    lines.append(f"# Proposal context for {args.unit}")
    lines.append("")
    lines.append("## Current unit")
    lines.append(f"- unit: {current_label}")
    lines.append(f"- source unit: {source_label}")
    lines.append("")
    lines.append("## Active evidence")
    lines.append(f"- path: {brief_path}")
    lines.append(f"- source unit: {round_state.get('evidence_for_source_unit')}")
    lines.append(f"- mode: {round_state.get('evidence_mode') or 'unspecified'}")
    lines.append(f"- package status: {evidence_package.get('status')}")
    if evidence_package.get("package_dir"):
        lines.append(f"- package dir: {evidence_package.get('package_dir')}")
    lines.append("")
    lines.append("Rendered evidence brief is available at the path above. Do not treat the rendered brief body as the proposal source of truth when it duplicates older context snapshots.")
    lines.append("Use the structured evidence package and fresh outcome memory sections below as the authoritative handoff for proposal writing.")
    lines.append("")
    lines.append("## Evidence package for proposal writer")
    lines.append("Use these structured package artifacts as the proposal evidence source of truth. The rendered brief is only the compact human-readable index.")
    lines.append("```json")
    lines.append(json.dumps(evidence_package, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Current runtime summary")
    lines.append("```json")
    lines.append(json.dumps(current_runtime, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Source runtime summary")
    lines.append("```json")
    lines.append(json.dumps(source_runtime, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Outcome memory")
    lines.append("This section is generated from completed-generation memory and should be used to avoid repeating losing or tradeoff patterns.")
    lines.append("```json")
    lines.append(json.dumps({k: v for k, v in outcome_memory.items() if k != "report_text"}, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    if outcome_memory.get("report_text"):
        lines.append("## Latest outcome report")
        lines.append("```markdown")
        lines.append(outcome_memory["report_text"])
        lines.append("```")
        lines.append("")

    lines.append("## Tree / lineage context")
    lines.append("This section is rebuilt from prior completed generations. Use it to avoid repeatedly expanding saturated families and to identify under-explored high-prior branches.")
    lines.append("```json")
    lines.append(json.dumps(tree_context, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Current capacity / objective / training knobs")
    lines.append("These are visible code-level MLIP knobs in the current unit. Treat capacity scaling and training-objective changes as legitimate but bounded evolution axes, not as substitutes for mechanism search.")
    lines.append("```json")
    lines.append(json.dumps(code_defaults, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    active_datasets = active_q_schema()["datasets"]
    for dataset in active_datasets:
        lines.append(f"## {dataset} benchmark dossier")
        lines.append("```json")
        lines.append(json.dumps(dataset_dossier(current_root, dataset), indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    profiles = profile_dossiers(current_root)
    if profiles:
        lines.append("## Adaptation profile dossiers")
        lines.append("These are auxiliary diagnostics. If a profile is promoted into the active q_version, it must also appear in the benchmark dossier above.")
        lines.append("```json")
        lines.append(json.dumps(profiles, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    if generation_summary:
        lines.append("## Generation summary")
        lines.append("```json")
        lines.append(json.dumps(generation_summary, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    lines.append("## Diff vs source: model.py")
    if model_diff:
        lines.extend([f"- {line}" for line in model_diff])
    else:
        lines.append("- no concise diff")
    lines.append("")
    lines.append("## Diff vs source: train.py")
    if train_diff:
        lines.extend([f"- {line}" for line in train_diff])
    else:
        lines.append("- no concise diff")
    lines.append("")

    lines.append("## Frontier tail")
    if frontier_tail:
        for row in frontier_tail:
            lines.append(
                f"- {row.get('unit')} | family={row.get('family')} | phase={row.get('phase')} | "
                f"Q_rmd17={row.get('Q_rmd17')} | Q_iso17={row.get('Q_iso17')} | Q_mad10k={row.get('Q_mad10k')} | "
                f"Q_total={row.get('Q_total')} | G_delta={row.get('G_delta')} | status={row.get('status')}"
            )
    else:
        lines.append("- no frontier records yet")
    lines.append("")

    lines.append("## Evidence brief path")
    lines.append(f"- {brief_path}")
    lines.append("")

    lines.append("## Proposal writing rule")
    lines.append("- proposal decisions must be benchmark-centric, not force-only")
    lines.append("- discuss energy, force, gap_penalty, Q fields, train trends, runtime/failure, and control comparison when relevant")
    lines.append("- if evidence package mechanism_cards or patch_blueprints exist, proposals must cite mechanism_refs and include files_to_edit, code_insertion_points, minimal_edit_plan, and implementation_checklist")
    lines.append("- do not use vague model-family analogies as strong evidence unless they are backed by mechanism_cards/proposal_constraints")
    lines.append("- treat `cutoff`, `num_rbf`, `hidden_dim`, `learning_rate`, `weight_decay`, `energy_weight`, and `force_weight` as MLIP code-level knobs in `model.py` / `train.py`, not as `config.json` edits")
    lines.append("- capacity scaling is a valid proposal axis when bounded and justified as a capacity/resolution/depth/training-budget bottleneck test; do not let scaling proposals replace mechanism coverage")
    lines.append("- `config.json` is benchmark/runtime configuration; do not propose changing it to improve model quality")
    lines.append("")
    lines.append("## Required proposal skeleton")
    lines.append("Every proposal must be long enough to serve as an implementation handoff. Use this skeleton, keeping the exact field names checked by transition_round_state.py.")
    lines.append("```markdown")
    lines.append("# Proposal NNN: <short concrete name>")
    lines.append("")
    lines.append("- family: <short_family_name>")
    lines.append("- phase: <0-6>")
    lines.append("- jump_type: exploit | jump | backward-simplify | control | wildcard")
    lines.append("- budget_class: tiny | small | medium | large")
    lines.append("- expected_capability_gain: <one line>")
    lines.append("")
    lines.append("## one_sentence_hypothesis")
    lines.append("<One sentence linking the code edit to the expected benchmark effect.>")
    lines.append("")
    lines.append("## mechanism_refs")
    lines.append("- <mechanism_id from mechanism_cards.json, or [] for control / diagnosis-only>")
    lines.append("")
    lines.append("## evidence_refs")
    lines.append("- <evidence source id, paper/repo/source id, or benchmark memory path>")
    lines.append("")
    lines.append("## historical_relation")
    lines.append("- source_unit: <generation_xxx/proposal_yyy>")
    lines.append("- relation_to_source: exploit | jump | simplify | control | ablation")
    lines.append("- not_a_duplicate_of: <recent units and why>")
    lines.append("- lesson_used: <negative or partial-positive pattern from outcome memory>")
    lines.append("")
    lines.append("## benchmark_rationale")
    lines.append("- capacity/scaling hypothesis, if any:")
    for dataset in active_datasets:
        lines.append(f"- {dataset} energy:")
        lines.append(f"- {dataset} force:")
        lines.append(f"- {dataset} gap / Q:")
    lines.append("- training stability / runtime risk:")
    lines.append("- control comparison expectation:")
    lines.append("")
    lines.append("## files_to_edit")
    lines.append("- `model/model.py`")
    lines.append("- `model/train.py` if needed, otherwise `none`")
    lines.append("- never `config.json` for MLIP-quality changes")
    lines.append("")
    lines.append("## code_insertion_points")
    lines.append("- `model/model.py::<ClassOrFunction>`: <what to change>")
    lines.append("- `model/train.py::<ClassOrFunction>`: <what to change or none>")
    lines.append("- code-level MLIP knobs may include `EvolutionMLIP.__init__` defaults, `MODEL_*` constants, or `TRAIN_*` constants when present")
    lines.append("")
    lines.append("## minimal_edit_plan")
    lines.append("1. <smallest edit step>")
    lines.append("2. <smallest edit step>")
    lines.append("3. <smallest edit step>")
    lines.append("")
    lines.append("## implementation_checklist")
    lines.append("- [ ] Preserve `E = sum_i E_i` and force-from-energy contract.")
    lines.append("- [ ] Preserve benchmark metric field names and runnable entrypoint contract.")
    lines.append("- [ ] Modify only the allowed target unit files.")
    lines.append("- [ ] Do not edit `config.json`; change MLIP knobs in `model/model.py` or `model/train.py`.")
    lines.append("- [ ] Keep tensor shapes compatible with current dataloader and model forward.")
    lines.append("- [ ] Add no unbounded cubic neighbor/triplet loops unless explicitly justified by budget.")
    lines.append("- [ ] Call `mark_unit_implemented.py --unit <UNIT> --actor implementation_subagent` after edits.")
    lines.append("")
    lines.append("## expected_benchmark_effect")
    lines.append("- primary expected gain:")
    lines.append("- expected tradeoff:")
    lines.append("- failure signal that would falsify this proposal:")
    lines.append("")
    lines.append("## ablation_or_control")
    lines.append("- required control or comparison:")
    lines.append("- optional zero-gate / source-fallback / readout-only ablation:")
    lines.append("")
    lines.append("## implementation_notes_for_subagent")
    lines.append("<Concrete instructions that survive materialization into `research_context/proposal.md`.>")
    lines.append("```")
    lines.append("")

    lines.append("## Current model.py")
    lines.append("```python")
    lines.append(current_model)
    lines.append("```")
    lines.append("")

    lines.append("## Current train.py")
    lines.append("```python")
    lines.append(current_train)
    lines.append("```")

    if round_state.get("continuation_source_unit") == current_label and round_state.get("active_proposal_directory"):
        target_dir = Path(round_state["active_proposal_directory"])
    else:
        target_dir = PROPOSALS / args.unit
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "context.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    round_state["workflow_state"] = "proposal-writing"
    round_state["proposal_context_file"] = str(out_path)
    round_state["proposal_directory_ready"] = False
    round_state["proposal_source"] = None
    round_state["proposal_writer_session"] = None
    save_json(round_state_path(), round_state)
    sync_round_state_to_remote(config=load_config())
    print(out_path)


if __name__ == "__main__":
    main()
