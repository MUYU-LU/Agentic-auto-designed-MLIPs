from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from runtime_common import (
    GENERATIONS,
    LEDGER,
    RUNTIME_ROOT,
    active_q_schema,
    load_json,
    load_run_status,
    now_utc,
    q_fields_for_unit,
    resolve_unit,
    save_json,
)
from family_taxonomy import canonical_jump_type, family_record


TREE_STATE = LEDGER / "tree_state.json"
LINEAGE_STATS = LEDGER / "lineage_stats.json"


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def generation_number(name: str) -> int:
    if not name.startswith("generation_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return -1


def parse_proposal_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for raw_line in text.splitlines()[:120]:
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in {"family", "phase", "jump_type", "budget_class", "expected_capability_gain"}:
            continue
        metadata[key] = value
    if "phase" in metadata:
        try:
            metadata["phase"] = int(str(metadata["phase"]).strip())
        except Exception:
            pass
    return metadata


def proposal_metadata_for_unit(unit_root: Path) -> dict[str, Any]:
    candidates = [unit_root / "research_context" / "proposal.md"]
    impl = load_json(unit_root / "implementation_status.json", {})
    proposal_file = impl.get("proposal_file")
    if proposal_file:
        candidates.insert(0, Path(str(proposal_file)))
    for path in candidates:
        if not path.exists():
            continue
        try:
            return parse_proposal_metadata(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    return {}


def read_text_if_exists(path: Path, max_chars: int = 200_000) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def unit_q(unit_root: Path) -> dict[str, float | str | None]:
    summary = load_json(unit_root / "outputs" / "summary.json", {})
    active_version = active_q_schema()["version"]
    qt = finite_float(summary.get("Q_total"))
    if qt is not None and summary.get("benchmark_version") == active_version:
        return {
            "Q_rmd17": finite_float(summary.get("Q_rmd17")),
            "Q_iso17": finite_float(summary.get("Q_iso17")),
            "Q_mad10k": finite_float(summary.get("Q_mad10k")),
            "Q_total": qt,
            "benchmark_version": summary.get("benchmark_version"),
        }
    return q_fields_for_unit(unit_root)


def source_q(nodes: dict[str, dict[str, Any]], source_unit: str | None) -> float | None:
    if not source_unit:
        return None
    node = nodes.get(source_unit)
    if node:
        return finite_float(node.get("Q_total"))
    if source_unit == "base_unit":
        try:
            root = resolve_unit(source_unit, RUNTIME_ROOT)
        except Exception:
            return None
        return finite_float(unit_q(root).get("Q_total"))
    return None


def selected_history() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(LEDGER.glob("generation_*_continuation_decision.json"), key=lambda p: generation_number(p.stem.replace("_continuation_decision", ""))):
        payload = load_json(path, {})
        unit = payload.get("selected_continuation_source")
        if not isinstance(unit, str) or "/" not in unit:
            continue
        rows.append(
            {
                "decision_file": str(path.relative_to(RUNTIME_ROOT)),
                "completed_generation": payload.get("completed_generation"),
                "target_generation": payload.get("target_generation") or path.stem.replace("_continuation_decision", ""),
                "selected_continuation_source": unit,
                "selected_Q_total": payload.get("selected_Q_total"),
                "best_known_unit": payload.get("best_known_unit"),
                "best_known_Q_total": payload.get("best_known_Q_total"),
            }
        )
    return rows


def empty_stats() -> dict[str, Any]:
    return {
        "tried_count": 0,
        "terminal_success_count": 0,
        "selected_count": 0,
        "frontier_win_count": 0,
        "best_Q_total": None,
        "best_unit": None,
        "mean_G_delta": None,
        "positive_G_delta_count": 0,
        "negative_G_delta_count": 0,
    }


def add_stat(stats: dict[str, Any], node: dict[str, Any]) -> None:
    stats["tried_count"] += 1
    if node.get("run_state") == "terminal_success":
        stats["terminal_success_count"] += 1
    selected_count = int(node.get("selected_as_continuation_count") or 0)
    stats["selected_count"] += selected_count
    q = finite_float(node.get("Q_total"))
    if q is not None and (stats["best_Q_total"] is None or q > float(stats["best_Q_total"])):
        stats["best_Q_total"] = q
        stats["best_unit"] = node.get("unit")
    if node.get("outcome_class") == "frontier_win":
        stats["frontier_win_count"] += 1
    g = finite_float(node.get("G_delta"))
    if g is not None:
        stats.setdefault("_g_values", []).append(g)
        if g > 0:
            stats["positive_G_delta_count"] += 1
        elif g < 0:
            stats["negative_G_delta_count"] += 1


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    values = stats.pop("_g_values", [])
    if values:
        stats["mean_G_delta"] = sum(values) / len(values)
    return stats


def build_tree_state() -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}

    for generation_root in sorted(GENERATIONS.glob("generation_*"), key=lambda p: generation_number(p.name)):
        if not generation_root.is_dir():
            continue
        for unit_root in sorted(generation_root.glob("proposal_*")):
            if not unit_root.is_dir():
                continue
            unit = f"{generation_root.name}/{unit_root.name}"
            meta = load_json(unit_root / "unit_meta.json", {})
            run_status = load_run_status(unit_root)
            summary = load_json(unit_root / "outputs" / "summary.json", {})
            proposal_meta = proposal_metadata_for_unit(unit_root)
            implementation_status = load_json(unit_root / "implementation_status.json", {})
            family = proposal_meta.get("family")
            jump_type = proposal_meta.get("jump_type")
            family_info = family_record(
                family,
                implementation_status=implementation_status,
                model_text=read_text_if_exists(unit_root / "model" / "model.py"),
                train_text=read_text_if_exists(unit_root / "model" / "train.py"),
                proposal_file=implementation_status.get("proposal_file"),
            )
            q = unit_q(unit_root)
            nodes[unit] = {
                "unit": unit,
                "generation": generation_root.name,
                "proposal": unit_root.name,
                "source_unit": meta.get("source_unit"),
                "run_state": run_status.get("run_state"),
                "Q_rmd17": q.get("Q_rmd17"),
                "Q_iso17": q.get("Q_iso17"),
                "Q_mad10k": q.get("Q_mad10k"),
                "Q_total": q.get("Q_total"),
                "benchmark_version": q.get("benchmark_version"),
                "G_delta": finite_float(summary.get("G_delta")),
                "outcome_class": None,
                "family": family,
                "canonical_family": family_info["canonical_family"],
                "mechanism_tags": family_info["mechanism_tags"],
                "family_source": family_info["family_source"],
                "phase": proposal_meta.get("phase"),
                "jump_type": jump_type,
                "canonical_jump_type": canonical_jump_type(jump_type),
                "budget_class": proposal_meta.get("budget_class"),
                "expected_capability_gain": proposal_meta.get("expected_capability_gain"),
                "children": [],
                "child_count": 0,
                "selected_as_continuation_count": 0,
                "selected_for_targets": [],
            }

    for node in nodes.values():
        if node["G_delta"] is None:
            parent_q = source_q(nodes, node.get("source_unit"))
            q = finite_float(node.get("Q_total"))
            if parent_q is not None and q is not None:
                node["G_delta"] = q - parent_q
        source = node.get("source_unit")
        if source in nodes:
            nodes[source]["children"].append(node["unit"])

    summaries = {}
    for path in sorted((LEDGER / "generation_summaries").glob("generation_*.json")):
        payload = load_json(path, {})
        for row in payload.get("units", []) or []:
            if row.get("unit") in nodes:
                summaries[row["unit"]] = row
    for unit, row in summaries.items():
        node = nodes[unit]
        node["outcome_class"] = row.get("outcome_class")
        if node["G_delta"] is None:
            node["G_delta"] = finite_float(row.get("delta_Q_vs_parent"))

    selection_history = selected_history()
    for row in selection_history:
        unit = row["selected_continuation_source"]
        if unit in nodes:
            nodes[unit]["selected_as_continuation_count"] += 1
            nodes[unit]["selected_for_targets"].append(row.get("target_generation"))

    for node in nodes.values():
        node["children"] = sorted(node["children"])
        node["child_count"] = len(node["children"])

    best_known = max(
        (node for node in nodes.values() if finite_float(node.get("Q_total")) is not None and node.get("run_state") == "terminal_success"),
        key=lambda node: (float(node["Q_total"]), node["unit"]),
        default=None,
    )

    family_stats = defaultdict(empty_stats)
    raw_family_stats = defaultdict(empty_stats)
    jump_type_stats = defaultdict(empty_stats)
    raw_jump_type_stats = defaultdict(empty_stats)
    phase_stats = defaultdict(empty_stats)
    for node in nodes.values():
        add_stat(family_stats[str(node.get("canonical_family") or "unknown")], node)
        add_stat(raw_family_stats[str(node.get("family") or "unknown")], node)
        add_stat(jump_type_stats[str(node.get("canonical_jump_type") or "unknown")], node)
        add_stat(raw_jump_type_stats[str(node.get("jump_type") or "unknown")], node)
        add_stat(phase_stats[str(node.get("phase") if node.get("phase") is not None else "unknown")], node)

    payload = {
        "created_at_utc": now_utc(),
        "runtime_root": str(RUNTIME_ROOT),
        "active_benchmark_version": active_q_schema()["version"],
        "node_count": len(nodes),
        "selection_count": len(selection_history),
        "best_known_unit": best_known.get("unit") if best_known else None,
        "best_known_Q_total": best_known.get("Q_total") if best_known else None,
        "nodes": dict(sorted(nodes.items())),
        "selection_history": selection_history,
        "family_stats": {key: finalize_stats(value) for key, value in sorted(family_stats.items())},
        "raw_family_stats": {key: finalize_stats(value) for key, value in sorted(raw_family_stats.items())},
        "jump_type_stats": {key: finalize_stats(value) for key, value in sorted(jump_type_stats.items())},
        "raw_jump_type_stats": {key: finalize_stats(value) for key, value in sorted(raw_jump_type_stats.items())},
        "phase_stats": {key: finalize_stats(value) for key, value in sorted(phase_stats.items())},
    }
    return payload


def compact_lineage_stats(tree: dict[str, Any]) -> dict[str, Any]:
    recent = tree.get("selection_history", [])[-8:]
    return {
        "created_at_utc": tree.get("created_at_utc"),
        "best_known_unit": tree.get("best_known_unit"),
        "best_known_Q_total": tree.get("best_known_Q_total"),
        "node_count": tree.get("node_count"),
        "selection_count": tree.get("selection_count"),
        "recent_selected_sources": recent,
        "family_stats": tree.get("family_stats", {}),
        "jump_type_stats": tree.get("jump_type_stats", {}),
        "phase_stats": tree.get("phase_stats", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild PUCT-style tree and lineage statistics from research_runtime history.")
    parser.add_argument("--no-lineage-stats", action="store_true")
    args = parser.parse_args()

    tree = build_tree_state()
    save_json(TREE_STATE, tree)
    if not args.no_lineage_stats:
        save_json(LINEAGE_STATS, compact_lineage_stats(tree))
    print(json.dumps({"tree_state": str(TREE_STATE), "lineage_stats": str(LINEAGE_STATS), "node_count": tree["node_count"], "selection_count": tree["selection_count"], "best_known_unit": tree["best_known_unit"], "best_known_Q_total": tree["best_known_Q_total"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
