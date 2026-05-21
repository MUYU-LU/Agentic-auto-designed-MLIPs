#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def compact_list(items: list[Any], limit: int = 5) -> list[Any]:
    return items[:limit] if isinstance(items, list) else []


def package_index(run: dict) -> dict:
    package_dir = run.get("evidence_package_dir")
    if not package_dir:
        return {}
    return {
        "evidence_quality": f"{package_dir}/evidence_quality.json",
        "evidence_provenance": f"{package_dir}/evidence_provenance.json",
        "current_code_profile": f"{package_dir}/current_code_profile.json",
        "benchmark_diagnosis": f"{package_dir}/benchmark_diagnosis.json",
        "generation_memory": f"{package_dir}/generation_memory.json",
        "mechanism_cards": f"{package_dir}/mechanism_cards.json",
        "patch_blueprints": f"{package_dir}/patch_blueprints.json",
        "proposal_constraints": f"{package_dir}/proposal_constraints.json",
        "audit_report": f"{package_dir}/audit_report.json",
        "source_plan": f"{package_dir}/source_plan.json",
        "source_artifacts_index": f"{package_dir}/source_artifacts_index.json",
        "source_novelty": f"{package_dir}/source_novelty.json",
        "source_analysis_requirements": f"{package_dir}/source_analysis_requirements.json",
        "paper_artifacts_dir": f"{package_dir}/paper_artifacts",
        "repo_artifacts_dir": f"{package_dir}/repo_artifacts",
        "evidence_run": run.get("evidence_run_path"),
    }


def source_attempt_summary(provenance: dict) -> dict:
    records = provenance.get("records", []) if isinstance(provenance, dict) else []
    by_type: dict[str, dict[str, int]] = {}
    strong_capable = []
    failed = []
    reused = []
    for row in records:
        kind = str(row.get("type", "unknown"))
        bucket = by_type.setdefault(kind, {"total": 0, "success": 0, "fresh": 0, "can_support_strong": 0})
        bucket["total"] += 1
        if row.get("success"):
            bucket["success"] += 1
        if row.get("freshness") == "fresh":
            bucket["fresh"] += 1
        if row.get("can_support_strong"):
            bucket["can_support_strong"] += 1
            strong_capable.append(row.get("source_id"))
        if row.get("success") is False:
            failed.append({"source_id": row.get("source_id"), "type": row.get("type"), "notes": row.get("notes")})
        if row.get("freshness") == "reused":
            reused.append(row.get("source_id"))
    return {
        "by_type": by_type,
        "strong_capable_sources": strong_capable,
        "failed_sources": failed[:8],
        "reused_sources": reused[:12],
    }


def mechanism_summary(mechanism_cards: dict) -> dict:
    cards = mechanism_cards.get("cards", []) if isinstance(mechanism_cards, dict) else []
    strong = mechanism_cards.get("strong_cards", []) if isinstance(mechanism_cards, dict) else []
    weak = mechanism_cards.get("weak_or_hypothesis_cards", []) if isinstance(mechanism_cards, dict) else []
    return {
        "strong_mechanism_ids": [card.get("mechanism_id") for card in strong],
        "weak_or_hypothesis_ids": [card.get("mechanism_id") for card in weak],
        "cards": [
            {
                "mechanism_id": card.get("mechanism_id"),
                "claim_strength": card.get("claim_strength"),
                "strong_ready": card.get("strong_ready"),
                "source_refs": card.get("source_refs", []),
                "concrete_mechanism": card.get("concrete_mechanism"),
                "current_code_insertion_point": card.get("current_code_insertion_point", []),
                "bounded_edit": card.get("bounded_edit", []),
                "downgrade_reasons": card.get("downgrade_reasons", []),
            }
            for card in compact_list(cards, 10)
        ],
    }


def blueprint_summary(patch_blueprints: dict) -> dict:
    blueprints = patch_blueprints.get("blueprints", []) if isinstance(patch_blueprints, dict) else []
    return {
        "implementation_ready_blueprints": [row.get("blueprint_id") for row in blueprints if row.get("implementation_ready")],
        "blueprints": [
            {
                "blueprint_id": row.get("blueprint_id"),
                "mechanism_id": row.get("mechanism_id"),
                "implementation_ready": row.get("implementation_ready"),
                "target_files": row.get("target_files", []),
                "target_insertion_points": row.get("target_insertion_points", []),
                "blocked_until": row.get("blocked_until", []),
            }
            for row in compact_list(blueprints, 10)
        ],
    }


def generation_memory_brief_summary(memory: dict) -> dict:
    latest = memory.get("latest_completed_generation_summary") or {}
    latest_summary = latest.get("summary", {}) if isinstance(latest, dict) else {}
    return {
        "version": memory.get("version"),
        "source_unit": memory.get("source_unit"),
        "last_completed_generation": memory.get("last_completed_generation"),
        "ignored_legacy_source_count": memory.get("ignored_legacy_source_count"),
        "completed_generation_count_for_source": memory.get("completed_generation_count_for_source"),
        "recent_attempt_count_for_source": memory.get("recent_attempt_count_for_source"),
        "unit_card_count_for_source": memory.get("unit_card_count_for_source"),
        "negative_pattern_count_for_source": memory.get("negative_pattern_count_for_source"),
        "partial_positive_pattern_count_for_source": memory.get("partial_positive_pattern_count_for_source"),
        "latest_generation": latest_summary.get("generation"),
        "latest_best_child": latest_summary.get("best_child"),
        "latest_outcome_counts": latest_summary.get("outcome_counts", {}),
        "latest_lessons": latest_summary.get("lessons", []),
        "note": "Full generation memory, unit cards, code deltas, and attempts are in generation_memory.json, not expanded in this brief.",
    }


def context_reference_summary(run: dict) -> dict:
    local_context = run.get("local_context", {}) if isinstance(run.get("local_context"), dict) else {}
    proposal_context = local_context.get("proposal_context", {}) if isinstance(local_context.get("proposal_context"), dict) else {}
    benchmark = run.get("benchmark_diagnosis") or run.get("benchmark_dossier") or {}
    return {
        "source_unit": run.get("source_unit"),
        "context_path": proposal_context.get("context_path"),
        "unit_root": local_context.get("unit_root"),
        "current_code_profile_path": package_index(run).get("current_code_profile"),
        "benchmark_diagnosis_path": package_index(run).get("benchmark_diagnosis"),
        "generation_memory_path": package_index(run).get("generation_memory"),
        "generation_memory": generation_memory_brief_summary(run.get("generation_memory", {})),
        "benchmark_warnings": benchmark.get("warnings", []) if isinstance(benchmark, dict) else [],
        "note": "Full proposal context, current code profile, and benchmark dossier are referenced here, not repeated in the brief.",
    }


def what_is_new(run: dict) -> dict:
    quality = run.get("evidence_quality", {})
    cards = run.get("mechanism_cards", {})
    constraints = run.get("proposal_constraints", {})
    return {
        "usable_for_proposal": quality.get("usable_for_proposal"),
        "usable_for_implementation": quality.get("usable_for_implementation"),
        "diagnosis_only": quality.get("diagnosis_only"),
        "strong_mechanisms": [card.get("mechanism_id") for card in cards.get("strong_cards", [])],
        "weak_or_hypothesis_mechanisms": [card.get("mechanism_id") for card in cards.get("weak_or_hypothesis_cards", [])],
        "proposal_allowed_mechanisms": constraints.get("allowed_mechanisms", []),
        "proposal_blocked_mechanisms": constraints.get("blocked_mechanisms", []),
    }


def build_markdown(run: dict) -> str:
    metadata = {
        "version": run.get("version"),
        "generated_at_utc": run.get("generated_at_utc"),
        "mode": run.get("mode"),
        "source_unit": run.get("source_unit"),
        "evidence_package_dir": run.get("evidence_package_dir"),
        "evidence_run_path": run.get("evidence_run_path"),
    }
    sections = [
        ("run_metadata", metadata),
        ("evidence_package_index", package_index(run)),
        ("package_contract", [
            "This brief is an evidence-delta index, not a full proposal context dump.",
            "JSON files in evidence_package_index are the handoff source of truth.",
            "Full local_context/current_code_profile/benchmark_diagnosis are referenced by path and compact summary only.",
            "No provenance -> no strong evidence.",
            "No mechanism card -> no proposal mechanism.",
            "No formula derivation / algorithm / repo code path / code trace / current insertion point -> weak evidence only.",
            "Benchmark diagnosis is internal diagnosis, not external mechanism evidence.",
            "Reading artifacts is not enough: strong mechanisms require derivation, tensor/data-flow reasoning, code trace, and bounded edit mapping.",
            "Source novelty is tracked; repeated recent sources without new mechanism extraction downgrade proposal readiness.",
        ]),
        ("evidence_quality", run.get("evidence_quality", {})),
        ("context_references", context_reference_summary(run)),
        ("source_attempt_summary", source_attempt_summary(run.get("evidence_provenance", {}))),
        ("source_novelty", run.get("source_novelty", {})),
        ("source_artifacts", run.get("source_artifacts", {})),
        ("source_analysis_requirements", run.get("source_analysis_requirements", {})),
        ("what_is_new_for_proposal", what_is_new(run)),
        ("new_mechanism_cards_summary", mechanism_summary(run.get("mechanism_cards", {}))),
        ("new_patch_blueprints_summary", blueprint_summary(run.get("patch_blueprints", {}))),
        ("proposal_constraints", run.get("proposal_constraints", {})),
        ("audit_report", run.get("audit_report", {})),
        ("what_not_to_use_as_strong_evidence", {
            "weak_or_hypothesis_mechanisms": run.get("proposal_constraints", {}).get("weak_or_hypothesis_mechanisms", []),
            "blocked_mechanisms": run.get("proposal_constraints", {}).get("blocked_mechanisms", []),
            "diagnosis_only": run.get("evidence_quality", {}).get("diagnosis_only"),
        }),
        ("followup_queries", run.get("followup_queries", [])),
    ]
    lines = ["# MLIP-Evidence Delta Brief", ""]
    for title, payload in sections:
        lines.extend([f"## {title}", "", render_json(payload), ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown evidence delta brief from evidence-run JSON.")
    parser.add_argument("--evidence-run", required=True, help="Path to evidence-run JSON.")
    parser.add_argument("--output", help="Output markdown path. Defaults beside JSON.")
    args = parser.parse_args()

    run_path = Path(args.evidence_run).expanduser().resolve()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve() if args.output else run_path.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(run), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
