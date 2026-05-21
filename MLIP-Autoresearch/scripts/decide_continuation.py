from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable

from runtime_common import (
    GENERATIONS,
    LEDGER,
    PROPOSALS,
    STAGING_RUNTIME_ROOT,
    load_config,
    load_json,
    load_run_status,
    resolve_unit,
    now_utc,
    q_fields_for_unit,
    remote_config,
    remote_runtime_root,
    remote_target,
    rsync_ssh_arg,
    save_json,
    sync_round_state_to_remote,
    update_round_state,
)
from family_taxonomy import canonical_family, canonical_jump_type, family_record


REQUIRED_HANDOFF_FILES = (
    "proposal",
    "context",
    "mechanism_cards",
    "patch_blueprints",
    "proposal_constraints",
)

JUMP_PRIORITY = {
    "jump": 3,
    "backward-simplify": 2,
    "wildcard": 1,
    "exploit": 0,
    "control": -1,
}

TRAIN_ONLY_BREAKTHROUGH_MARGIN = 0.10
TREE_STATE = LEDGER / "tree_state.json"


DEFAULT_SELECTION = {
    "strategy": "puct_lite",
    "c_puct": 0.45,
    "penalty_weight": 0.08,
    "pattern_penalty_weight": 0.04,
    "max_pattern_penalty": 0.12,
    "negative_pattern_penalty": 0.08,
    "recent_pattern_window": 6,
    "pattern_overlap_threshold": 4,
    "novel_family_prior_bonus": 0.20,
    "positive_family_delta_scale": 1.00,
    "max_family_delta_prior": 0.08,
    "role_priors": {
        "jump": 1.08,
        "wildcard": 1.04,
        "backward-simplify": 1.02,
        "exploit": 1.00,
        "control": 0.20,
    },
}



def generation_number(name: str) -> int:
    if not name.startswith("generation_"):
        raise SystemExit(f"Invalid generation name: {name!r}")
    return int(name.split("_", 1)[1])



def next_generation_name(name: str) -> str:
    return f"generation_{generation_number(name) + 1:03d}"



def proposal_dir_for_source(source_unit: str, target_generation: str) -> Path:
    source_slug = source_unit.replace("/", "_")
    return PROPOSALS / f"{target_generation}_from_{source_slug}_continuation"



def _decision_files() -> Iterable[Path]:
    return sorted(
        LEDGER.glob("generation_*_continuation_decision.json"),
        key=lambda p: generation_number(p.stem.replace("_continuation_decision", "")),
    )



def continuation_source_streak(parent_source: str | None) -> int:
    if not parent_source:
        return 0
    streak = 0
    for path in reversed(list(_decision_files())):
        payload = load_json(path, {})
        if payload.get("selected_continuation_source") == parent_source:
            streak += 1
            continue
        break
    return streak


def recent_selected_units(window: int = 4) -> list[str]:
    units: list[str] = []
    for path in reversed(list(_decision_files())):
        payload = load_json(path, {})
        unit = payload.get("selected_continuation_source")
        if unit:
            units.append(unit)
        if len(units) >= window:
            break
    return units


def recent_selection_counters(window: int = 4) -> dict:
    family = Counter()
    phase = Counter()
    jump_type = Counter()
    for unit in recent_selected_units(window=window):
        meta = candidate_runtime_metadata(unit)
        if meta.get("canonical_family"):
            family[meta["canonical_family"]] += 1
        if meta.get("phase") is not None:
            phase[str(meta["phase"])] += 1
        if meta.get("canonical_jump_type"):
            jump_type[meta["canonical_jump_type"]] += 1
    return {
        "family": family,
        "phase": phase,
        "jump_type": jump_type,
    }


def load_negative_patterns() -> list[dict]:
    path = LEDGER / 'negative_patterns.jsonl'
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def finite_float(value) -> float | None:
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def selection_config(config: dict, args: argparse.Namespace | None = None) -> dict:
    configured = config.get("selection", {}) if isinstance(config.get("selection"), dict) else {}
    merged = dict(DEFAULT_SELECTION)
    merged.update({k: v for k, v in configured.items() if k != "role_priors"})
    role_priors = dict(DEFAULT_SELECTION["role_priors"])
    if isinstance(configured.get("role_priors"), dict):
        role_priors.update(configured["role_priors"])
    merged["role_priors"] = role_priors
    if args is not None:
        if args.selection_strategy:
            merged["strategy"] = args.selection_strategy
        if args.c_puct is not None:
            merged["c_puct"] = args.c_puct
    return merged


def load_tree_state() -> dict:
    return load_json(TREE_STATE, {})


def rank_scores(rows: list[dict]) -> dict[str, float]:
    valid = [row for row in rows if finite_float(row.get("Q_total")) is not None]
    valid = sorted(valid, key=lambda row: (float(row["Q_total"]), row.get("unit", "")))
    if not valid:
        return {}
    if len(valid) == 1:
        return {valid[0]["unit"]: 1.0}
    denom = len(valid) - 1
    return {row["unit"]: index / denom for index, row in enumerate(valid)}


def family_stats_for(row: dict, tree_state: dict) -> dict:
    family = str(row.get("canonical_family") or canonical_family(row.get("family")))
    return (tree_state.get("family_stats") or {}).get(family, {})


def node_stats_for(row: dict, tree_state: dict) -> dict:
    return (tree_state.get("nodes") or {}).get(row.get("unit"), {})


def puct_prior(row: dict, tree_state: dict, config: dict) -> float:
    role = str(row.get("canonical_jump_type") or canonical_jump_type(row.get("jump_type")))
    prior = float((config.get("role_priors") or {}).get(role, 1.0))
    family_stats = family_stats_for(row, tree_state)
    tried = int(family_stats.get("tried_count") or 0)
    if tried == 0:
        prior += float(config.get("novel_family_prior_bonus", 0.0) or 0.0)
    mean_g = finite_float(family_stats.get("mean_G_delta"))
    if mean_g is not None:
        scale = float(config.get("positive_family_delta_scale", 0.0) or 0.0)
        cap = float(config.get("max_family_delta_prior", 0.0) or 0.0)
        prior += max(-cap, min(cap, mean_g * scale))
    if row.get("recent_negative_pattern_match"):
        prior -= min(0.05, float(config.get("negative_pattern_penalty", 0.08) or 0.0))
    return max(0.05, prior)


def puct_expansion_count(row: dict, tree_state: dict) -> int:
    node = node_stats_for(row, tree_state)
    return int(node.get("selected_as_continuation_count") or 0)


def puct_select_child(candidates: list[dict], tree_state: dict, config: dict) -> tuple[dict | None, dict]:
    valid = [
        row
        for row in candidates
        if row.get("candidate_origin") == "completed_generation"
        and row.get("run_state") == "terminal_success"
        and isinstance(row.get("Q_total"), (int, float))
        and row.get("research_worthy_child") is True
    ]
    if not valid:
        return None, {
            "strategy": config.get("strategy"),
            "tree_state_path": str(TREE_STATE),
            "tree_state_available": bool(tree_state),
            "selected_unit": None,
            "reason": "no_research_worthy_terminal_success_child",
            "candidate_scores": [],
        }

    ranks = rank_scores(valid)
    total_expansions = int(tree_state.get("selection_count") or 0)
    exploration_root = math.sqrt(math.log(1.0 + max(total_expansions, 1)))
    c_puct = float(config.get("c_puct", 0.0) or 0.0)
    penalty_weight = float(config.get("penalty_weight", 0.0) or 0.0)
    pattern_penalty_weight = float(config.get("pattern_penalty_weight", 0.04) or 0.0)
    max_pattern_penalty = float(config.get("max_pattern_penalty", 0.12) or 0.0)
    negative_pattern_penalty_value = float(config.get("negative_pattern_penalty", 0.08) or 0.0)
    scored = []
    for row in valid:
        prior = puct_prior(row, tree_state, config)
        expansions = puct_expansion_count(row, tree_state)
        rank_score = ranks.get(row["unit"], 0.0)
        exploration = c_puct * prior * exploration_root / (1.0 + expansions)
        pattern_penalty = min(max_pattern_penalty, pattern_penalty_weight * float(row.get("pattern_repetition_count") or 0.0))
        negative_pattern_penalty = negative_pattern_penalty_value if row.get("recent_negative_pattern_match") else 0.0
        penalty = pattern_penalty + negative_pattern_penalty + penalty_weight * max(0.0, float(row.get("legacy_repetition_penalty_score") or 0.0))
        score = rank_score + exploration - penalty
        scored.append(
            {
                "unit": row["unit"],
                "Q_total": row.get("Q_total"),
                "rank_score": rank_score,
                "prior": prior,
                "node_expansion_count": expansions,
                "family_selected_count": family_stats_for(row, tree_state).get("selected_count"),
                "exploration_bonus": exploration,
                "pattern_repetition_count": row.get("pattern_repetition_count"),
                "pattern_repetition_penalty": pattern_penalty,
                "negative_pattern_penalty": negative_pattern_penalty,
                "penalty": penalty,
                "puct_score": score,
                "family": row.get("family"),
                "canonical_family": row.get("canonical_family"),
                "phase": row.get("phase"),
                "jump_type": row.get("jump_type"),
                "canonical_jump_type": row.get("canonical_jump_type"),
                "mechanism_refs": row.get("mechanism_refs"),
                "files_changed_kind": row.get("files_changed_kind"),
                "model_change_kind": row.get("model_change_kind"),
                "train_change_kind": row.get("train_change_kind"),
                "long_memory_penalty_score": row.get("long_memory_penalty_score"),
            }
        )
    scored = sorted(scored, key=lambda item: (item["puct_score"], item.get("Q_total") or -1.0, item["unit"]), reverse=True)
    selected_unit = scored[0]["unit"]
    selected = next(row for row in valid if row["unit"] == selected_unit)
    return selected, {
        "strategy": config.get("strategy"),
        "tree_state_path": str(TREE_STATE),
        "tree_state_available": bool(tree_state),
        "tree_selection_count": tree_state.get("selection_count"),
        "tree_best_known_unit": tree_state.get("best_known_unit"),
        "tree_best_known_Q_total": tree_state.get("best_known_Q_total"),
        "selected_unit": selected_unit,
        "candidate_scores": scored,
    }



def best_known_candidate(candidates: list[dict]) -> dict | None:
    valid = [row for row in candidates if row.get("run_state") == "terminal_success" and isinstance(row.get("Q_total"), (int, float))]
    if not valid:
        return None
    return max(valid, key=lambda row: (float(row["Q_total"]), row["unit"]))



def best_completed_generation_child(candidates: list[dict]) -> dict | None:
    valid = [
        row
        for row in candidates
        if row.get("candidate_origin") == "completed_generation"
        and row.get("run_state") == "terminal_success"
        and isinstance(row.get("Q_total"), (int, float))
    ]
    if not valid:
        return None
    return max(valid, key=lambda row: (float(row["Q_total"]), row["unit"]))



def _research_priority(row: dict) -> tuple:
    return (
        int(bool(row.get("evolution_relevant_change"))),
        -int(row.get("long_memory_penalty_score") or 0),
        JUMP_PRIORITY.get(row.get("jump_type"), 0),
        int(row.get("phase") if isinstance(row.get("phase"), int) else -1),
        float(row.get("Q_total") or -1),
        row.get("unit"),
    )


def best_research_worthy_child(candidates: list[dict], *, research_tie_margin: float) -> tuple[dict | None, dict]:
    valid = [
        row
        for row in candidates
        if row.get("candidate_origin") == "completed_generation"
        and row.get("run_state") == "terminal_success"
        and isinstance(row.get("Q_total"), (int, float))
        and row.get("research_worthy_child") is True
    ]
    if not valid:
        return None, {
            "research_tiebreak_applied": False,
            "evolution_priority_reason": None,
            "best_jump_candidate": None,
            "best_exploit_candidate": None,
            "best_q_research_child": None,
        }
    best_q_child = max(valid, key=lambda row: (float(row["Q_total"]), row["unit"]))
    best_q = float(best_q_child["Q_total"])
    close = [row for row in valid if best_q - float(row["Q_total"]) <= research_tie_margin]
    selected = max(close, key=_research_priority)
    best_jump = max((row for row in valid if row.get("jump_type") == "jump"), key=lambda row: (float(row["Q_total"]), row["unit"]), default=None)
    best_exploit = max((row for row in valid if row.get("jump_type") == "exploit"), key=lambda row: (float(row["Q_total"]), row["unit"]), default=None)
    reason = None
    applied = len(close) > 1 and selected.get("unit") != best_q_child.get("unit")
    if applied:
        selected_penalty = int(selected.get("long_memory_penalty_score") or 0)
        best_penalty = int(best_q_child.get("long_memory_penalty_score") or 0)
        if selected_penalty < best_penalty:
            reason = "research_tiebreak_avoided_recently_penalized_branch_within_q_margin"
        elif selected.get("jump_type") in {"jump", "backward-simplify"}:
            reason = "research_tiebreak_preferred_nontrivial_branch_within_q_margin"
        elif selected.get("evolution_relevant_change"):
            reason = "research_tiebreak_preferred_evolution_relevant_change_within_q_margin"
        else:
            reason = "research_tiebreak_selected_highest_priority_close_candidate"
    return selected, {
        "research_tiebreak_applied": applied,
        "evolution_priority_reason": reason,
        "best_jump_candidate": best_jump.get("unit") if best_jump else None,
        "best_exploit_candidate": best_exploit.get("unit") if best_exploit else None,
        "best_q_research_child": best_q_child.get("unit"),
    }



def _find_changed_sha(implementation_status: dict, rel_path: str) -> str | None:
    for row in implementation_status.get("changed_files", []) or []:
        if row.get("path") == rel_path:
            return row.get("sha256")
    return None



def _handoff_present(implementation_report: dict) -> bool:
    handoff = implementation_report.get("handoff_files_present", {}) or {}
    return all(bool(handoff.get(key)) for key in REQUIRED_HANDOFF_FILES)


def _parse_proposal_metadata_lines(text: str) -> dict:
    metadata: dict = {}
    lines = text.splitlines()
    in_mechanism_refs = False
    mechanism_refs: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("## "):
            heading = line.lstrip("#").strip().lower().replace(" ", "_")
            in_mechanism_refs = heading == "mechanism_refs"
            continue
        if in_mechanism_refs:
            if line.startswith("- "):
                ref = line[2:].strip().strip("`")
                if ref:
                    mechanism_refs.append(ref)
            elif line:
                in_mechanism_refs = False
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"family", "phase", "jump_type", "budget_class", "expected_capability_gain"}:
            metadata[key] = value
    if "phase" in metadata:
        try:
            metadata["phase"] = int(str(metadata["phase"]).strip())
        except Exception:
            pass
    metadata["mechanism_refs"] = mechanism_refs
    return metadata


def proposal_metadata_for_unit(unit_root: Path, implementation_status: dict) -> dict:
    candidates = []
    proposal_file = implementation_status.get("proposal_file")
    if proposal_file:
        candidates.append(Path(str(proposal_file)))
    candidates.append(unit_root / "research_context" / "proposal.md")
    for candidate in candidates:
        try:
            if candidate.exists():
                return _parse_proposal_metadata_lines(candidate.read_text())
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


def _slug_token(value: object) -> str:
    text = str(value or "").strip().lower()
    out = []
    prev_underscore = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        elif not prev_underscore:
            out.append("_")
            prev_underscore = True
    return "".join(out).strip("_") or "none"


def files_changed_kind(implementation_status: dict) -> str:
    paths = {str(row.get("path")) for row in implementation_status.get("changed_files", []) or []}
    model_changed = "model/model.py" in paths
    train_changed = "model/train.py" in paths
    if model_changed and train_changed:
        return "model_and_train"
    if model_changed:
        return "model_only"
    if train_changed:
        return "train_only"
    return "no_code_change"


def _diff_text(implementation_report: dict, rel_path: str) -> str:
    file_diffs = implementation_report.get("file_diffs_vs_source", {}) or {}
    payload = file_diffs.get(rel_path, {}) or {}
    lines = payload.get("unified_diff_excerpt", []) or []
    return "\n".join(str(line) for line in lines).lower()


def infer_model_change_kind(proposal_metadata: dict, implementation_report: dict) -> str:
    text = " ".join(
        [
            str(proposal_metadata.get("family") or ""),
            str(proposal_metadata.get("expected_capability_gain") or ""),
            " ".join(proposal_metadata.get("mechanism_refs") or []),
            _diff_text(implementation_report, "model/model.py"),
        ]
    ).lower()
    if not text.strip():
        return "none"
    if any(token in text for token in ("readout", "vector_norm", "energy_mlp", "energy_head", "layernorm", "per_atom_energy")):
        return "readout_path"
    if any(token in text for token in ("triplet", "angle", "manybody", "body_order", "higher_order", "three_body")):
        return "manybody_path"
    if any(token in text for token in ("message", "interaction", "edge_filter", "balancedinteraction", "vector_state", "equivariant", "radial_proj")):
        return "message_passing_path"
    if any(token in text for token in ("scalar_mix", "scalar_reentry", "scalar_residual")):
        return "scalar_mixing_path"
    if any(token in text for token in ("atomref", "offset", "baseline", "calibration")):
        return "energy_baseline_path"
    return "other_model_change"


def infer_train_change_kind(proposal_metadata: dict, implementation_report: dict) -> str:
    text = " ".join(
        [
            str(proposal_metadata.get("family") or ""),
            str(proposal_metadata.get("expected_capability_gain") or ""),
            " ".join(proposal_metadata.get("mechanism_refs") or []),
            _diff_text(implementation_report, "model/train.py"),
        ]
    ).lower()
    if not text.strip():
        return "none"
    if any(token in text for token in ("atomref", "lstsq", "offset", "freeze", "requires_grad")):
        return "atomref_training"
    if any(token in text for token in ("loss", "energy_weight", "force_weight", "gap", "regularization")):
        return "loss_weighting"
    if any(token in text for token in ("scheduler", "warmup", "learning_rate", "lr", "optimizer", "adam")):
        return "optimizer_schedule"
    if any(token in text for token in ("batch", "sample", "split", "dataloader")):
        return "data_sampling"
    return "other_train_change"


def normalize_mechanism_refs(refs: object) -> list[str]:
    if not isinstance(refs, list):
        return []
    return sorted({_slug_token(ref) for ref in refs if str(ref or "").strip()})


def pattern_overlap_score(row: dict, other: dict) -> int:
    score = 0
    if row.get("canonical_family") and row.get("canonical_family") == other.get("canonical_family"):
        score += 1
    if row.get("canonical_jump_type") and row.get("canonical_jump_type") == other.get("canonical_jump_type"):
        score += 1
    if row.get("phase") is not None and row.get("phase") == other.get("phase"):
        score += 1
    if set(row.get("mechanism_refs_norm") or []).intersection(other.get("mechanism_refs_norm") or []):
        score += 2
    if row.get("files_changed_kind") not in {None, "none", "no_code_change"} and row.get("files_changed_kind") == other.get("files_changed_kind"):
        score += 1
    if row.get("model_change_kind") != "none" and row.get("model_change_kind") == other.get("model_change_kind"):
        score += 1
    if row.get("train_change_kind") != "none" and row.get("train_change_kind") == other.get("train_change_kind"):
        score += 1
    return score


def recent_pattern_repetition(row: dict, *, window: int, threshold: int) -> dict:
    matches = []
    for unit in recent_selected_units(window=window):
        other = candidate_runtime_metadata(unit)
        if not other:
            continue
        score = pattern_overlap_score(row, other)
        if score >= threshold:
            matches.append(
                {
                    "unit": unit,
                    "overlap_score": score,
                    "canonical_family": other.get("canonical_family"),
                    "canonical_jump_type": other.get("canonical_jump_type"),
                    "phase": other.get("phase"),
                    "mechanism_refs": other.get("mechanism_refs"),
                    "model_change_kind": other.get("model_change_kind"),
                    "train_change_kind": other.get("train_change_kind"),
                }
            )
    return {"count": len(matches), "matches": matches}



def candidate_runtime_metadata(unit: str) -> dict:
    try:
        unit_root = resolve_unit(unit, STAGING_RUNTIME_ROOT)
    except Exception:
        return {}
    implementation_status = load_json(unit_root / "implementation_status.json", {})
    implementation_report = load_json(unit_root / "research_context" / "implementation_report.json", {})
    notes = implementation_report.get("notes") or implementation_status.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    proposal_metadata = proposal_metadata_for_unit(unit_root, implementation_status)
    family = proposal_metadata.get("family")
    jump_type = proposal_metadata.get("jump_type")
    mechanism_refs = proposal_metadata.get("mechanism_refs") or []
    family_info = family_record(
        family,
        implementation_status=implementation_status,
        model_text=read_text_if_exists(unit_root / "model" / "model.py"),
        train_text=read_text_if_exists(unit_root / "model" / "train.py"),
        proposal_file=implementation_status.get("proposal_file"),
    )
    return {
        "control_replicate": bool(implementation_status.get("control_replicate")),
        "model_sha256": _find_changed_sha(implementation_status, "model/model.py"),
        "train_sha256": _find_changed_sha(implementation_status, "model/train.py"),
        "implementation_status_file": str(unit_root / "implementation_status.json"),
        "implementation_report_file": str(unit_root / "research_context" / "implementation_report.json"),
        "handoff_files_present": _handoff_present(implementation_report),
        "implementation_notes": notes,
        "family": family,
        "canonical_family": family_info["canonical_family"],
        "mechanism_tags": family_info["mechanism_tags"],
        "family_source": family_info["family_source"],
        "phase": proposal_metadata.get("phase"),
        "jump_type": jump_type,
        "canonical_jump_type": canonical_jump_type(jump_type),
        "budget_class": proposal_metadata.get("budget_class"),
        "expected_capability_gain": proposal_metadata.get("expected_capability_gain"),
        "mechanism_refs": mechanism_refs,
        "mechanism_refs_norm": normalize_mechanism_refs(mechanism_refs),
        "files_changed_kind": files_changed_kind(implementation_status),
        "model_change_kind": infer_model_change_kind(proposal_metadata, implementation_report),
        "train_change_kind": infer_train_change_kind(proposal_metadata, implementation_report),
    }



def annotate_candidate_relationships(
    candidates: list[dict],
    parent_candidate: dict | None,
    *,
    select_config: dict | None = None,
) -> None:
    parent_model = parent_candidate.get("model_sha256") if parent_candidate else None
    parent_train = parent_candidate.get("train_sha256") if parent_candidate else None
    recent_counters = recent_selection_counters()
    negative_patterns = load_negative_patterns()
    config = select_config or DEFAULT_SELECTION
    pattern_window = int(config.get("recent_pattern_window") or DEFAULT_SELECTION["recent_pattern_window"])
    pattern_threshold = int(config.get("pattern_overlap_threshold") or DEFAULT_SELECTION["pattern_overlap_threshold"])
    for row in candidates:
        row["is_parent_source"] = bool(parent_candidate and row.get("unit") == parent_candidate.get("unit"))
        row["parent_model_sha256"] = parent_model
        row["parent_train_sha256"] = parent_train
        same_model = bool(parent_model and row.get("model_sha256") and row.get("model_sha256") == parent_model)
        same_train = bool(parent_train and row.get("train_sha256") and row.get("train_sha256") == parent_train)
        row["same_model_as_parent"] = same_model
        row["same_train_as_parent"] = same_train
        row["near_duplicate_to_parent"] = bool(
            parent_candidate
            and row.get("candidate_origin") == "completed_generation"
            and same_model
            and same_train
        )
        row["only_train_diff_vs_parent"] = bool(
            parent_candidate
            and row.get("candidate_origin") == "completed_generation"
            and same_model
            and not same_train
        )
        row["train_only_breakthrough"] = bool(
            row.get("only_train_diff_vs_parent")
            and isinstance(row.get("Q_total"), (int, float))
            and isinstance(parent_candidate.get("Q_total") if parent_candidate else None, (int, float))
            and float(row["Q_total"]) >= float(parent_candidate["Q_total"]) + TRAIN_ONLY_BREAKTHROUGH_MARGIN
        )
        row["same_family_as_parent"] = bool(
            parent_candidate
            and row.get("candidate_origin") == "completed_generation"
            and row.get("canonical_family")
            and row.get("canonical_family") == parent_candidate.get("canonical_family")
        )
        row["same_phase_as_parent"] = bool(
            parent_candidate
            and row.get("candidate_origin") == "completed_generation"
            and row.get("phase") is not None
            and row.get("phase") == parent_candidate.get("phase")
        )
        row["same_jump_type_as_parent"] = bool(
            parent_candidate
            and row.get("candidate_origin") == "completed_generation"
            and row.get("canonical_jump_type")
            and row.get("canonical_jump_type") == parent_candidate.get("canonical_jump_type")
        )
        row["evolution_relevant_change"] = bool(
            row.get("candidate_origin") == "completed_generation"
            and row.get("canonical_jump_type") not in {None, "control"}
            and (
                not row.get("same_family_as_parent")
                or not row.get("same_phase_as_parent")
                or not row.get("same_jump_type_as_parent")
                or not row.get("same_model_as_parent")
            )
        )
        row["recent_selected_family_count"] = recent_counters['family'].get(row.get('canonical_family'), 0) if row.get('canonical_family') else 0
        row["recent_selected_phase_count"] = recent_counters['phase'].get(str(row.get('phase')), 0) if row.get('phase') is not None else 0
        row["recent_selected_jump_type_count"] = recent_counters['jump_type'].get(row.get('canonical_jump_type'), 0) if row.get('canonical_jump_type') else 0
        row["recent_negative_pattern_match"] = any(
            patt.get('source_unit') == (parent_candidate.get('unit') if parent_candidate else None)
            and (
                (row.get('canonical_family') and candidate_runtime_metadata(patt.get('unit', '')).get('canonical_family') == row.get('canonical_family'))
                or (row.get('canonical_jump_type') and candidate_runtime_metadata(patt.get('unit', '')).get('canonical_jump_type') == row.get('canonical_jump_type'))
            )
            for patt in negative_patterns
        )
        pattern = recent_pattern_repetition(row, window=pattern_window, threshold=pattern_threshold)
        row["pattern_repetition_count"] = pattern["count"]
        row["pattern_repetition_matches"] = pattern["matches"]
        row["legacy_repetition_penalty_score"] = 0
        row["long_memory_penalty_score"] = (
            int(bool(row.get('recent_negative_pattern_match'))) * 2
            + int(row.get("pattern_repetition_count") or 0)
        )
        row["train_only_research_worthy"] = bool(
            row.get("only_train_diff_vs_parent")
            and row.get("canonical_family") == "training_objective"
            and row.get("mechanism_refs")
            and isinstance(row.get("Q_total"), (int, float))
            and isinstance(parent_candidate.get("Q_total") if parent_candidate else None, (int, float))
            and float(row["Q_total"]) >= float(parent_candidate["Q_total"]) - 0.05
        )
        row["research_worthy_child"] = bool(
            row.get("candidate_origin") == "completed_generation"
            and row.get("run_state") == "terminal_success"
            and not row.get("control_replicate")
            and not row.get("near_duplicate_to_parent")
            and (
                not row.get("only_train_diff_vs_parent")
                or row.get("train_only_breakthrough")
                or row.get("train_only_research_worthy")
            )
            and row.get("handoff_files_present")
        )



def choose_continuation(
    candidates: list[dict],
    requested: str | None,
    parent_candidate: dict | None,
    *,
    parent_margin: float,
    max_parent_streak: int,
    research_tie_margin: float,
    selection_strategy: str = "current",
    puct_config: dict | None = None,
    tree_state: dict | None = None,
) -> tuple[dict, dict]:
    puct_config = puct_config or DEFAULT_SELECTION
    tree_state = tree_state or {}

    def decision_fields(
        *,
        parent: dict | None,
        selected_child: dict | None,
        anchor_child: dict | None,
        streak: int,
        duplicate_guard_triggered: bool,
        fallback_to_parent_reason: str | None,
        selection_mode: str,
        research_tiebreak: dict,
        puct_diagnostics: dict | None = None,
    ) -> dict:
        return {
            'selection_mode': selection_mode,
            'parent_source_unit': parent.get('unit') if parent else None,
            'parent_source_streak': streak,
            'parent_family': parent.get('family') if parent else None,
            'parent_phase': parent.get('phase') if parent else None,
            'parent_jump_type': parent.get('jump_type') if parent else None,
            'parent_budget_class': parent.get('budget_class') if parent else None,
            'selected_child_unit': selected_child.get('unit') if selected_child else None,
            'selected_child_Q_total': selected_child.get('Q_total') if selected_child else None,
            'selected_child_model_hash': selected_child.get('model_sha256') if selected_child else None,
            'selected_child_train_hash': selected_child.get('train_sha256') if selected_child else None,
            'selected_child_family': selected_child.get('family') if selected_child else None,
            'selected_child_canonical_family': selected_child.get('canonical_family') if selected_child else None,
            'selected_child_phase': selected_child.get('phase') if selected_child else None,
            'selected_child_jump_type': selected_child.get('jump_type') if selected_child else None,
            'selected_child_canonical_jump_type': selected_child.get('canonical_jump_type') if selected_child else None,
            'selected_child_budget_class': selected_child.get('budget_class') if selected_child else None,
            'selected_child_mechanism_refs': selected_child.get('mechanism_refs') if selected_child else None,
            'selected_child_files_changed_kind': selected_child.get('files_changed_kind') if selected_child else None,
            'selected_child_model_change_kind': selected_child.get('model_change_kind') if selected_child else None,
            'selected_child_train_change_kind': selected_child.get('train_change_kind') if selected_child else None,
            'selected_child_pattern_repetition_count': selected_child.get('pattern_repetition_count') if selected_child else None,
            'selected_child_pattern_repetition_matches': selected_child.get('pattern_repetition_matches') if selected_child else None,
            'best_child_unit': anchor_child.get('unit') if anchor_child else None,
            'best_child_Q_total': anchor_child.get('Q_total') if anchor_child else None,
            'best_child_gap_to_parent': (
                (float(parent.get('Q_total')) - float(anchor_child.get('Q_total')))
                if parent and anchor_child and isinstance(parent.get('Q_total'), (int, float)) and isinstance(anchor_child.get('Q_total'), (int, float))
                else None
            ),
            'best_child_model_hash': anchor_child.get('model_sha256') if anchor_child else None,
            'best_child_train_hash': anchor_child.get('train_sha256') if anchor_child else None,
            'best_child_family': anchor_child.get('family') if anchor_child else None,
            'best_child_canonical_family': anchor_child.get('canonical_family') if anchor_child else None,
            'best_child_phase': anchor_child.get('phase') if anchor_child else None,
            'best_child_jump_type': anchor_child.get('jump_type') if anchor_child else None,
            'best_child_canonical_jump_type': anchor_child.get('canonical_jump_type') if anchor_child else None,
            'best_child_budget_class': anchor_child.get('budget_class') if anchor_child else None,
            'best_child_mechanism_refs': anchor_child.get('mechanism_refs') if anchor_child else None,
            'best_child_files_changed_kind': anchor_child.get('files_changed_kind') if anchor_child else None,
            'best_child_model_change_kind': anchor_child.get('model_change_kind') if anchor_child else None,
            'best_child_train_change_kind': anchor_child.get('train_change_kind') if anchor_child else None,
            'best_child_pattern_repetition_count': anchor_child.get('pattern_repetition_count') if anchor_child else None,
            'best_child_pattern_repetition_matches': anchor_child.get('pattern_repetition_matches') if anchor_child else None,
            'parent_model_hash': parent.get('model_sha256') if parent else None,
            'parent_train_hash': parent.get('train_sha256') if parent else None,
            'best_child_is_near_duplicate': anchor_child.get('near_duplicate_to_parent') if anchor_child else None,
            'research_worthy_child': anchor_child.get('research_worthy_child') if anchor_child else None,
            'duplicate_guard_triggered': duplicate_guard_triggered,
            'selected_child_recent_negative_pattern_match': selected_child.get('recent_negative_pattern_match') if selected_child else None,
            'selected_child_long_memory_penalty_score': selected_child.get('long_memory_penalty_score') if selected_child else None,
            'selected_child_recent_selected_family_count': selected_child.get('recent_selected_family_count') if selected_child else None,
            'selected_child_recent_selected_jump_type_count': selected_child.get('recent_selected_jump_type_count') if selected_child else None,
            'selected_child_only_train_diff_vs_parent': selected_child.get('only_train_diff_vs_parent') if selected_child else None,
            'selected_child_train_only_breakthrough': selected_child.get('train_only_breakthrough') if selected_child else None,
            'selected_child_train_only_research_worthy': selected_child.get('train_only_research_worthy') if selected_child else None,
            'best_child_recent_negative_pattern_match': anchor_child.get('recent_negative_pattern_match') if anchor_child else None,
            'best_child_long_memory_penalty_score': anchor_child.get('long_memory_penalty_score') if anchor_child else None,
            'best_child_recent_selected_family_count': anchor_child.get('recent_selected_family_count') if anchor_child else None,
            'best_child_recent_selected_jump_type_count': anchor_child.get('recent_selected_jump_type_count') if anchor_child else None,
            'best_child_only_train_diff_vs_parent': anchor_child.get('only_train_diff_vs_parent') if anchor_child else None,
            'best_child_train_only_breakthrough': anchor_child.get('train_only_breakthrough') if anchor_child else None,
            'best_child_train_only_research_worthy': anchor_child.get('train_only_research_worthy') if anchor_child else None,
            **research_tiebreak,
            'puct_lite': puct_diagnostics,
            'fallback_to_parent_reason': fallback_to_parent_reason,
        }

    if requested:
        selected = next((row for row in candidates if row['unit'] == requested), None)
        if not selected:
            raise SystemExit(f'Requested source unit is not in candidate set: {requested}')
        return selected, decision_fields(
            parent=parent_candidate,
            selected_child=None,
            anchor_child=None,
            streak=continuation_source_streak(parent_candidate.get('unit')) if parent_candidate else 0,
            duplicate_guard_triggered=False,
            fallback_to_parent_reason=None,
            selection_mode='requested_override',
            research_tiebreak={
                'research_tiebreak_applied': False,
                'evolution_priority_reason': None,
                'best_jump_candidate': None,
                'best_exploit_candidate': None,
                'best_q_research_child': None,
            },
            puct_diagnostics=None,
        )

    best_child = best_completed_generation_child(candidates)
    best_research_child, research_tiebreak = best_research_worthy_child(candidates, research_tie_margin=research_tie_margin)
    puct_child, puct_diagnostics = puct_select_child(candidates, tree_state, puct_config)
    use_puct = selection_strategy == "puct_lite"
    best_q_research_child = None
    if research_tiebreak.get('best_q_research_child'):
        best_q_research_child = next((row for row in candidates if row.get('unit') == research_tiebreak.get('best_q_research_child')), None)
    if not best_child and not parent_candidate:
        raise SystemExit('No terminal_success continuation candidates with Q_total.')
    if not parent_candidate:
        selected = (puct_child if use_puct and puct_child else None) or best_research_child or best_child
        return selected, decision_fields(
            parent=None,
            selected_child=selected,
            anchor_child=best_q_research_child or selected,
            streak=0,
            duplicate_guard_triggered=False,
            fallback_to_parent_reason=None,
            selection_mode='puct_lite_no_parent' if use_puct and puct_child else ('best_research_worthy_child' if best_research_child else 'best_completed_generation_child'),
            research_tiebreak=research_tiebreak,
            puct_diagnostics=puct_diagnostics,
        )

    parent_q = parent_candidate.get('Q_total')
    if parent_candidate.get('run_state') != 'terminal_success' or not isinstance(parent_q, (int, float)):
        selected = (puct_child if use_puct and puct_child else None) or best_research_child or best_child
        if selected:
            return selected, decision_fields(
                parent=parent_candidate,
                selected_child=selected,
                anchor_child=best_q_research_child or selected,
                streak=continuation_source_streak(parent_candidate.get('unit')),
                duplicate_guard_triggered=False,
                fallback_to_parent_reason='parent_source_missing_terminal_success_q_total',
                selection_mode='parent_invalid_use_puct_lite' if use_puct and puct_child else 'parent_invalid_use_best_child',
                research_tiebreak=research_tiebreak,
                puct_diagnostics=puct_diagnostics,
            )
        raise SystemExit(f"Continuation parent source is not terminal_success with Q_total: {parent_candidate.get('unit')}")

    streak = continuation_source_streak(parent_candidate.get('unit'))
    if not best_child:
        return parent_candidate, decision_fields(
            parent=parent_candidate,
            selected_child=None,
            anchor_child=None,
            streak=streak,
            duplicate_guard_triggered=False,
            fallback_to_parent_reason='no_terminal_success_child_with_q_total',
            selection_mode='parent_fallback_no_valid_child',
            research_tiebreak=research_tiebreak,
            puct_diagnostics=puct_diagnostics,
        )

    selected_child = (puct_child if use_puct and puct_child else None) or best_research_child or best_child
    anchor_child = best_q_research_child or selected_child
    child_q = float(anchor_child['Q_total'])
    parent_q = float(parent_q)
    duplicate_guard_triggered = bool(
        best_child
        and selected_child
        and best_child.get('unit') == selected_child.get('unit')
        and best_child.get('near_duplicate_to_parent')
    )

    if not best_research_child:
        return parent_candidate, decision_fields(
            parent=parent_candidate,
            selected_child=selected_child,
            anchor_child=anchor_child,
            streak=streak,
            duplicate_guard_triggered=duplicate_guard_triggered,
            fallback_to_parent_reason='best_child_failed_research_worthiness_gate',
            selection_mode='parent_fallback_non_research_worthy_child',
            research_tiebreak=research_tiebreak,
            puct_diagnostics=puct_diagnostics,
        )

    selected_q = float(selected_child['Q_total'])

    if streak >= max_parent_streak and selected_child['unit'] != parent_candidate.get('unit'):
        return selected_child, decision_fields(
            parent=parent_candidate,
            selected_child=selected_child,
            anchor_child=anchor_child,
            streak=streak,
            duplicate_guard_triggered=duplicate_guard_triggered,
            fallback_to_parent_reason=None,
            selection_mode='puct_lite_streak_forced_child' if use_puct and puct_child else 'streak_forced_child',
            research_tiebreak=research_tiebreak,
            puct_diagnostics=puct_diagnostics,
        )
    if selected_q >= parent_q - parent_margin:
        return selected_child, decision_fields(
            parent=parent_candidate,
            selected_child=selected_child,
            anchor_child=anchor_child,
            streak=streak,
            duplicate_guard_triggered=duplicate_guard_triggered,
            fallback_to_parent_reason=None,
            selection_mode='puct_lite_child_within_parent_margin' if use_puct and puct_child else 'child_within_parent_margin',
            research_tiebreak=research_tiebreak,
            puct_diagnostics=puct_diagnostics,
        )
    return parent_candidate, decision_fields(
        parent=parent_candidate,
        selected_child=selected_child,
        anchor_child=anchor_child,
        streak=streak,
        duplicate_guard_triggered=duplicate_guard_triggered,
        fallback_to_parent_reason='best_child_below_parent_margin',
        selection_mode='parent_fallback_higher_q_total',
        research_tiebreak=research_tiebreak,
        puct_diagnostics=puct_diagnostics,
    )


def unit_q(unit_root: Path) -> dict:
    return q_fields_for_unit(unit_root)



def candidate_for_unit(unit: str) -> dict | None:
    try:
        unit_root = resolve_unit(unit, STAGING_RUNTIME_ROOT)
    except Exception:
        return None
    run_status = load_run_status(unit_root)
    q = unit_q(unit_root)
    payload = {
        "unit": unit,
        "run_state": run_status.get("run_state"),
        "Q_rmd17": q.get("Q_rmd17"),
        "Q_iso17": q.get("Q_iso17"),
        "Q_mad10k": q.get("Q_mad10k"),
        "Q_total": q.get("Q_total"),
        "benchmark_version": q.get("benchmark_version"),
        "candidate_origin": "parent_source",
    }
    payload.update(candidate_runtime_metadata(unit))
    return payload



def candidates_for_generation(generation: str) -> list[dict]:
    generation_root = GENERATIONS / generation
    if not generation_root.exists():
        raise SystemExit(f"Missing generation directory: {generation_root}")
    rows = []
    for unit_root in sorted(generation_root.glob("proposal_*")):
        if not unit_root.is_dir():
            continue
        run_status = load_run_status(unit_root)
        q = unit_q(unit_root)
        payload = {
            "unit": f"{generation}/{unit_root.name}",
            "run_state": run_status.get("run_state"),
            "Q_rmd17": q.get("Q_rmd17"),
            "Q_iso17": q.get("Q_iso17"),
            "Q_mad10k": q.get("Q_mad10k"),
            "Q_total": q.get("Q_total"),
            "benchmark_version": q.get("benchmark_version"),
            "candidate_origin": "completed_generation",
        }
        payload.update(candidate_runtime_metadata(payload["unit"]))
        rows.append(payload)
    return rows



def sync_file_to_remote(local_path: Path, remote_path: str, *, config: dict) -> None:
    command = []
    password = remote_config(config).get("password")
    if password:
        command.extend(["sshpass", "-p", str(password)])
    command.extend(["rsync", "-az", "-e", rsync_ssh_arg(config), str(local_path), f"{remote_target(config)}:{remote_path}"])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"Failed to sync {local_path.name} to remote:\n{result.stderr}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Write continuation decision and advance round_state.")
    parser.add_argument("--completed-generation", required=True)
    parser.add_argument("--target-generation")
    parser.add_argument("--source-unit", help="Optional reviewed continuation source override.")
    parser.add_argument("--evidence-mode", default="balanced", choices=["balanced", "exploit", "jump"])
    parser.add_argument("--parent-margin", type=float, default=0.15)
    parser.add_argument("--max-parent-streak", type=int, default=3)
    parser.add_argument("--research-tie-margin", type=float, default=0.05)
    parser.add_argument("--selection-strategy", choices=["current", "puct_lite"], default=None)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--sync-remote", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    select_config = selection_config(config, args)
    tree_state = load_tree_state()
    completed_generation = args.completed_generation
    target_generation = args.target_generation or next_generation_name(completed_generation)
    candidates = candidates_for_generation(completed_generation)
    round_state_existing = load_json(LEDGER / "round_state.json", {})
    decision_path = LEDGER / f"{target_generation}_continuation_decision.json"
    existing_decision = load_json(decision_path, {}) if decision_path.exists() else {}
    prior_round_decision = load_json(LEDGER / f"{completed_generation}_continuation_decision.json", {})
    parent_source = (
        prior_round_decision.get("selected_continuation_source")
        or existing_decision.get("parent_source_unit")
        or round_state_existing.get("continuation_source_unit")
    )
    parent_candidate = None
    if parent_source:
        parent_candidate = candidate_for_unit(parent_source)
        if parent_candidate and not any(row.get("unit") == parent_candidate.get("unit") for row in candidates):
            candidates.append(parent_candidate)
    annotate_candidate_relationships(candidates, parent_candidate, select_config=select_config)
    selected, selection_diagnostics = choose_continuation(
        candidates,
        args.source_unit,
        parent_candidate,
        parent_margin=args.parent_margin,
        max_parent_streak=args.max_parent_streak,
        research_tie_margin=args.research_tie_margin,
        selection_strategy=str(select_config.get("strategy") or "current"),
        puct_config=select_config,
        tree_state=tree_state,
    )
    best_known = best_known_candidate(candidates)
    proposal_dir = proposal_dir_for_source(selected["unit"], target_generation)
    if decision_path.exists() and not args.force:
        existing = load_json(decision_path, {})
        if existing.get("selected_continuation_source") != selected["unit"]:
            raise SystemExit(f"Existing decision uses a different source. Review {decision_path} or pass --force.")

    decision = {
        "generated_at_utc": now_utc(),
        "completed_generation": completed_generation,
        "target_generation": target_generation,
        "selected_continuation_source": selected["unit"],
        "next_to_expand_unit": selected["unit"],
        "best_known_unit": best_known["unit"] if best_known else None,
        "best_known_Q_total": best_known.get("Q_total") if best_known else None,
        "selection_rule": "best-known unit is tracked separately from next-to-expand; continuation prefers research-worthy completed-generation children and keeps the old parent only as fallback under margin/streak guards unless --source-unit is provided",
        "selection_strategy": select_config.get("strategy"),
        "selection_config": select_config,
        "selected_Q_rmd17": selected["Q_rmd17"],
        "selected_Q_iso17": selected["Q_iso17"],
        "selected_Q_mad10k": selected.get("Q_mad10k"),
        "selected_Q_total": selected["Q_total"],
        "benchmark_version": selected.get("benchmark_version"),
        "parent_margin": args.parent_margin,
        "max_parent_streak": args.max_parent_streak,
        "research_tie_margin": args.research_tie_margin,
        **selection_diagnostics,
        "candidates": sorted(candidates, key=lambda row: row.get("Q_total") if row.get("Q_total") is not None else -1, reverse=True),
    }
    save_json(decision_path, decision)

    round_state = update_round_state(
        STAGING_RUNTIME_ROOT,
        workflow_state="evidence-needed",
        current_generation=completed_generation,
        active_writer_unit=None,
        active_evidence_task=None,
        blocked_reason=None,
        last_completed_generation=completed_generation,
        continuation_source_unit=selected["unit"],
        active_proposal_directory=str(proposal_dir),
        active_selection_file=None,
        materialized_units_root=None,
        active_evidence_brief=None,
        evidence_for_source_unit=None,
        evidence_mode=args.evidence_mode,
        proposal_context_file=None,
        proposal_directory_ready=False,
        proposal_writer_session=None,
        proposal_source=None,
        continuation_decision_file=str(decision_path),
        next_recommended_step=f"invoke_evidence_for_{target_generation}",
        source_of_truth="remote",
    )

    if args.sync_remote:
        sync_file_to_remote(decision_path, f"{remote_runtime_root(config)}/ledger/{decision_path.name}", config=config)
        sync_round_state_to_remote(config=config)

    print({
        "decision_file": str(decision_path),
        "selected_continuation_source": selected["unit"],
        "target_generation": target_generation,
        "round_state_workflow": round_state.get("workflow_state"),
        "synced_remote": bool(args.sync_remote),
    })


if __name__ == "__main__":
    main()
