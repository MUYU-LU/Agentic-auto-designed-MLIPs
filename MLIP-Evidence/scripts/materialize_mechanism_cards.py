#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_run_evidence_module():
    script_path = Path(__file__).resolve().with_name("run_evidence.py")
    spec = importlib.util.spec_from_file_location("run_evidence", script_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_build_brief_module():
    script_path = Path(__file__).resolve().with_name("build_brief.py")
    spec = importlib.util.spec_from_file_location("build_brief", script_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def infer_runtime_root(package_dir: Path) -> Path | None:
    # Expected: <runtime>/knowledge/evidence_runs/<run_id>
    if package_dir.parent.name == "evidence_runs" and package_dir.parent.parent.name == "knowledge":
        return package_dir.parent.parent.parent
    return None


def strong_source_usage(cards: list[dict]) -> dict[str, list[str]]:
    usage: dict[str, list[str]] = {}
    for card in cards:
        if not card.get("strong_ready"):
            continue
        mechanism_id = str(card.get("mechanism_id") or "")
        if not mechanism_id:
            continue
        for ref in card.get("source_refs", []) or []:
            if not isinstance(ref, str):
                continue
            if not ref.startswith(("paper_artifact:", "repo_artifact:")):
                continue
            usage.setdefault(ref, [])
            if mechanism_id not in usage[ref]:
                usage[ref].append(mechanism_id)
    return usage


def backfill_source_history(package_dir: Path, run: dict, cards: list[dict]) -> dict:
    runtime_root = infer_runtime_root(package_dir)
    if runtime_root is None:
        return {"updated": False, "reason": "cannot infer runtime root from package dir"}

    history_path = runtime_root / "knowledge" / "evidence_source_history.jsonl"
    rows = load_jsonl(history_path)
    if not rows:
        return {"updated": False, "reason": "history file missing or empty", "history_path": str(history_path)}

    run_id = str(run.get("run_id") or package_dir.name)
    usage = strong_source_usage(cards)
    touched = 0
    for row in rows:
        if row.get("run_id") != run_id:
            continue
        artifact_ref = row.get("artifact_ref")
        mechanism_ids = usage.get(artifact_ref, [])
        if not mechanism_ids:
            continue
        merged = list(dict.fromkeys([*(row.get("mechanism_ids") or []), *mechanism_ids]))
        row["used_as_strong"] = True
        row["mechanism_ids"] = merged
        touched += 1

    if touched:
        write_jsonl(history_path, rows)

    return {
        "updated": bool(touched),
        "history_path": str(history_path),
        "run_id": run_id,
        "strong_source_ref_count": len(usage),
        "history_rows_updated": touched,
    }


def normalize_cards(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        cards = payload
    elif isinstance(payload, dict):
        cards = payload.get("cards") or payload.get("mechanism_cards") or []
    else:
        cards = []
    if not isinstance(cards, list):
        raise SystemExit("mechanism card draft must be a list or an object with a cards field")
    out = []
    for index, card in enumerate(cards, start=1):
        if not isinstance(card, dict):
            raise SystemExit(f"card {index} is not an object")
        card = dict(card)
        card.setdefault("mechanism_id", f"MATERIALIZED-{index:03d}")
        card.setdefault("claim_strength", "candidate_strong")
        out.append(card)
    return out


def rebuild_blueprints(run_evidence, cards: dict) -> dict:
    return run_evidence.build_patch_blueprints(cards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and materialize evidence-agent mechanism cards into an existing evidence package.")
    parser.add_argument("--package-dir", required=True, help="Path to research_runtime/knowledge/evidence_runs/<run_id>.")
    parser.add_argument("--mechanism-cards", required=True, help="JSON draft written after reading paper/repo artifacts.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    package_dir = Path(args.package_dir).expanduser().resolve()
    draft_path = Path(args.mechanism_cards).expanduser().resolve()
    if not package_dir.exists():
        raise SystemExit(f"package dir does not exist: {package_dir}")
    if not draft_path.exists():
        raise SystemExit(f"mechanism card draft does not exist: {draft_path}")

    run_evidence = load_run_evidence_module()
    build_brief = load_build_brief_module()

    run_path = package_dir / "evidence_run.json"
    run = load_json(run_path, {})
    provenance = load_json(package_dir / "evidence_provenance.json", {})
    benchmark = load_json(package_dir / "benchmark_diagnosis.json", {})
    source_novelty = load_json(package_dir / "source_novelty.json", {})
    draft_payload = load_json(draft_path, {})
    cards = normalize_cards(draft_payload)

    for card in cards:
        strong_ready, reasons = run_evidence.validate_mechanism_card(card, provenance)
        card["strong_ready"] = strong_ready
        card["downgrade_reasons"] = reasons
        if strong_ready:
            card["claim_strength"] = "strong"
        elif card.get("claim_strength") == "strong":
            card["claim_strength"] = "weak_hypothesis"

    mechanism_cards = {
        "version": "mechanism_cards.v1.materialized",
        "materialized_from": str(draft_path),
        "cards": cards,
        "strong_cards": [card for card in cards if card.get("strong_ready")],
        "weak_or_hypothesis_cards": [card for card in cards if not card.get("strong_ready")],
    }
    patch_blueprints = rebuild_blueprints(run_evidence, mechanism_cards)
    require_external = bool((run.get("evidence_quality") or {}).get("require_external_evidence"))
    evidence_quality = run_evidence.build_evidence_quality(provenance, mechanism_cards, patch_blueprints, benchmark, source_novelty, require_external)
    proposal_constraints = run_evidence.build_proposal_constraints(mechanism_cards, evidence_quality)
    audit_report = run_evidence.build_audit_report(provenance, mechanism_cards, evidence_quality)

    run.update({
        "mechanism_cards": mechanism_cards,
        "patch_blueprints": patch_blueprints,
        "evidence_quality": evidence_quality,
        "proposal_constraints": proposal_constraints,
        "audit_report": audit_report,
        "mechanism_ledger": mechanism_cards.get("cards", []),
        "mechanism_cards_materialized_from": str(draft_path),
    })
    history_backfill = backfill_source_history(package_dir, run, cards)
    run["source_history_backfill"] = history_backfill

    write_json(package_dir / "mechanism_cards.json", mechanism_cards)
    write_json(package_dir / "patch_blueprints.json", patch_blueprints)
    write_json(package_dir / "evidence_quality.json", evidence_quality)
    write_json(package_dir / "proposal_constraints.json", proposal_constraints)
    write_json(package_dir / "audit_report.json", audit_report)
    write_json(run_path, run)

    brief_text = build_brief.build_markdown(run)
    package_brief = Path(run.get("package_brief_path") or package_dir / "evidence_brief.md")
    package_brief.write_text(brief_text, encoding="utf-8")
    legacy_brief_raw = run.get("brief_path")
    if legacy_brief_raw:
        Path(legacy_brief_raw).write_text(brief_text, encoding="utf-8")

    print(json.dumps({
        "package_dir": str(package_dir),
        "mechanism_card_count": len(cards),
        "strong_mechanism_card_count": len(mechanism_cards["strong_cards"]),
        "usable_for_proposal": evidence_quality.get("usable_for_proposal"),
        "usable_for_implementation": evidence_quality.get("usable_for_implementation"),
        "diagnosis_only": evidence_quality.get("diagnosis_only"),
        "source_history_backfill": history_backfill,
        "evidence_quality": evidence_quality,
    }, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
