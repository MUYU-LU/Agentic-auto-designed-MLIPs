#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_KEYS = [
    "brief_path",
    "question",
    "mode",
    "source_unit",
    "current_bottleneck",
    "strongest_exploit_angle",
    "strongest_jump_angle",
    "local_context",
    "current_unit_profile",
    "benchmark_dossier",
    "strong_evidence",
    "weak_but_relevant",
    "background_context",
    "mathematical_forms",
    "physical_principles",
    "chemical_regime",
    "textual_evidence",
    "code_evidence",
    "relevant_papers",
    "relevant_repos",
    "capability_gap",
    "implementable_design_moves",
    "exploit_angles",
    "jump_angles",
    "risks_or_mismatches",
    "handoff_summary",
    "confidence",
]

SOURCE_HISTORY = "knowledge/evidence_source_history.jsonl"
DEFAULT_SOURCE_NOVELTY_POLICY = {
    "lookback_runs": 3,
    "min_attempted_external_sources": 2,
    "min_new_sources": 1,
    "max_recent_reused_sources": 1,
}

DEFAULT_SOURCE_DISCOVERY_BUDGET = {
    "min_successful_pdf_sources": 1,
    "min_successful_repo_sources": 1,
    "min_strong_mechanism_cards": 1,
    "max_pdf_attempts": 6,
    "max_repo_attempts": 4,
    "max_external_source_attempts": 10,
    "max_source_expansion_rounds": 2,
    "source_expansion_round": 1,
}


def load_local_module(script_name: str):
    script_path = Path(__file__).resolve().with_name(script_name)
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_runtime_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    cwd = Path.cwd().resolve()
    env_root = os.environ.get("MLIP_RUNTIME_ROOT")
    workspace = Path(os.environ.get("OPENCLAW_WORKSPACE", str(cwd))).expanduser()
    candidates = [cwd / "research_runtime", cwd.parent / "research_runtime", workspace / "research_runtime"]
    if env_root:
        candidates.insert(0, Path(env_root).expanduser())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (cwd / "research_runtime").resolve()


def resolve_unit(runtime_root: Path, source_unit: str) -> Path:
    candidate = Path(source_unit).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if source_unit == "base_unit":
        return runtime_root / "base_unit"
    if source_unit == "seed_unit":
        return runtime_root / "seed_unit"
    return runtime_root / "generations" / source_unit


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default



def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value

def read_jsonl_tail(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            rows.append(sanitize_json_value(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return rows


def list_recent_briefs(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    files = sorted(path.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    return [str(item) for item in files[:limit]]


def read_recent_notes(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    files = sorted(path.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    notes = []
    for item in files[:limit]:
        text = item.read_text(encoding="utf-8", errors="replace")
        notes.append({"path": str(item), "snippet": text[:1200]})
    return notes



def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(sanitize_json_value(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return rows


def compact_code_delta(delta: dict, max_diff_lines: int = 18) -> dict:
    if not isinstance(delta, dict):
        return {}
    files = []
    for row in delta.get("files", [])[:2]:
        files.append({
            "path": row.get("path"),
            "same_as_source": row.get("same_as_source"),
            "added_lines": row.get("added_lines"),
            "removed_lines": row.get("removed_lines"),
            "diff_truncated": row.get("diff_truncated"),
            "unified_diff_excerpt": row.get("unified_diff_excerpt", [])[:max_diff_lines],
        })
    return {
        "source_unit": delta.get("source_unit"),
        "changed_files": delta.get("changed_files", []),
        "files": files,
    }


def compact_attempt(row: dict) -> dict:
    return {
        "unit": row.get("unit"),
        "source_unit": row.get("source_unit"),
        "outcome_class": row.get("outcome_class"),
        "run_state": row.get("run_state"),
        "Q_rmd17": row.get("Q_rmd17"),
        "Q_iso17": row.get("Q_iso17"),
        "Q_total": row.get("Q_total"),
        "delta_Q_vs_parent": row.get("delta_Q_vs_parent"),
        "mechanism_refs": row.get("proposal", {}).get("mechanism_refs", []),
        "historical_relation": row.get("proposal", {}).get("historical_relation"),
        "code_delta": compact_code_delta(row.get("code_delta", {})),
        "outcome_reasons": row.get("outcome_reasons", []),
    }


def compact_generation_summary(summary: dict) -> dict:
    units = [compact_attempt(row) for row in summary.get("units", [])]
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
        "units_by_Q_total": units[:12],
    }


def generation_memory(runtime_root: Path, source_unit: str, round_state: dict) -> dict:
    ledger = runtime_root / "ledger"
    summaries_dir = ledger / "generation_summaries"
    reports_dir = ledger / "generation_reports"
    unit_cards_dir = ledger / "unit_cards"
    all_attempts = load_jsonl(ledger / "all_attempts.jsonl")
    source_attempts = [row for row in all_attempts if row.get("source_unit") == source_unit]
    source_attempts = sorted(source_attempts, key=lambda row: row.get("unit", ""))[-32:]

    summaries = []
    for path in sorted(summaries_dir.glob("generation_*.json")):
        payload = load_json(path, {})
        if payload.get("source_unit") == source_unit:
            summaries.append({
                "path": str(path),
                "report_path": str(reports_dir / f"{path.stem}.md") if (reports_dir / f"{path.stem}.md").exists() else None,
                "summary": compact_generation_summary(payload),
            })

    unit_card_paths = sorted(unit_cards_dir.glob("*.json"))[-48:] if unit_cards_dir.exists() else []
    unit_cards = []
    source_units = {row.get("unit") for row in source_attempts}
    for path in unit_card_paths:
        payload = load_json(path, {})
        if payload.get("source_unit") == source_unit or payload.get("unit") in source_units:
            unit_cards.append({"path": str(path), "unit": payload.get("unit"), "card": compact_attempt(payload)})

    old_flat_summaries = sorted(ledger.glob("generation_*_summary.json"))
    return {
        "version": "generation_memory.v1",
        "source_unit": source_unit,
        "last_completed_generation": round_state.get("last_completed_generation"),
        "canonical_sources": {
            "generation_summaries_dir": str(summaries_dir),
            "generation_reports_dir": str(reports_dir),
            "all_attempts": str(ledger / "all_attempts.jsonl"),
            "unit_cards_dir": str(unit_cards_dir),
            "mechanism_outcomes": str(ledger / "mechanism_outcomes.jsonl"),
            "negative_patterns": str(ledger / "negative_patterns.jsonl"),
            "partial_positive_patterns": str(ledger / "partial_positive_patterns.jsonl"),
        },
        "ignored_legacy_sources": [str(path) for path in old_flat_summaries],
        "reason_legacy_ignored": "Flat ledger/generation_XXX_summary.json files are legacy compatibility artifacts; canonical completed-generation memory is under ledger/generation_summaries/ plus all_attempts/unit_cards.",
        "completed_generation_summaries_for_source": summaries[-6:],
        "recent_attempts_for_source": [compact_attempt(row) for row in source_attempts],
        "unit_cards_for_source": unit_cards[-24:],
        "mechanism_outcomes_for_source": [row for row in load_jsonl(ledger / "mechanism_outcomes.jsonl") if row.get("unit") in source_units][-64:],
        "negative_patterns_for_source": [row for row in load_jsonl(ledger / "negative_patterns.jsonl") if row.get("source_unit") == source_unit][-32:],
        "partial_positive_patterns_for_source": [row for row in load_jsonl(ledger / "partial_positive_patterns.jsonl") if row.get("source_unit") == source_unit][-32:],
    }


def compact_generation_memory(memory: dict) -> dict:
    return {
        "version": memory.get("version"),
        "source_unit": memory.get("source_unit"),
        "last_completed_generation": memory.get("last_completed_generation"),
        "canonical_sources": memory.get("canonical_sources", {}),
        "ignored_legacy_source_count": len(memory.get("ignored_legacy_sources", [])),
        "completed_generation_count_for_source": len(memory.get("completed_generation_summaries_for_source", [])),
        "recent_attempt_count_for_source": len(memory.get("recent_attempts_for_source", [])),
        "unit_card_count_for_source": len(memory.get("unit_cards_for_source", [])),
        "negative_pattern_count_for_source": len(memory.get("negative_patterns_for_source", [])),
        "partial_positive_pattern_count_for_source": len(memory.get("partial_positive_patterns_for_source", [])),
        "latest_completed_generation_summary": (memory.get("completed_generation_summaries_for_source") or [None])[-1],
    }


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8", errors="replace")


def context_candidates(runtime_root: Path, source_unit: str, round_state: dict) -> list[Path]:
    candidates = [
        runtime_root / "proposals" / source_unit / "context.md",
        runtime_root / "proposals" / source_unit.replace("/", "_") / "context.md",
    ]
    active_dir = round_state.get("active_proposal_directory")
    if active_dir:
        candidates.append(Path(active_dir).expanduser() / "context.md")
    return candidates


def load_full_context(runtime_root: Path, source_unit: str, round_state: dict) -> dict:
    context_path = None
    context_body = ""
    for candidate in context_candidates(runtime_root, source_unit, round_state):
        if candidate.exists():
            context_path = candidate
            context_body = read_text(candidate)
            break

    unit_root = resolve_unit(runtime_root, source_unit)
    unit_summary = load_json(unit_root / "outputs" / "summary.json", {})
    unit_meta = load_json(unit_root / "unit_meta.json", {})
    generation_summary = {}
    generation_name = unit_meta.get("generation_round")
    if generation_name:
        canonical = runtime_root / "ledger" / "generation_summaries" / f"{generation_name}.json"
        legacy = runtime_root / "ledger" / f"{generation_name}_summary.json"
        generation_summary = load_json(canonical, {}) or load_json(legacy, {})

    return {
        "context_path": str(context_path) if context_path else None,
        "context_body": context_body,
        "unit_summary": unit_summary,
        "generation_summary": generation_summary,
    }


def benchmark_dossier_from_context(local_context: dict) -> dict:
    body = str(local_context.get("proposal_context", {}).get("context_body", ""))
    unit_summary = local_context.get("proposal_context", {}).get("unit_summary", {}) or {}
    generation_summary = local_context.get("proposal_context", {}).get("generation_summary", {}) or {}
    serialized = json.dumps(local_context, ensure_ascii=False)
    warnings = []
    required_benchmark_strings = ["mixed_force_mae", "mixed_energy_mae", "gap_penalty"]
    if not any(token in serialized for token in required_benchmark_strings):
        warnings.append("local_context appears benchmark-incomplete; evidence quality may be force-biased")
    if "Q_total" not in serialized and "Q_dataset" not in serialized:
        warnings.append("Q fields are absent from local_context")
    if "run_state" not in serialized:
        warnings.append("runtime/failure state is absent from local_context")
    return {
        "context_path": local_context.get("proposal_context", {}).get("context_path"),
        "has_full_context_body": bool(body),
        "unit_Q_rmd17": unit_summary.get("Q_rmd17"),
        "unit_Q_iso17": unit_summary.get("Q_iso17"),
        "unit_Q_total": unit_summary.get("Q_total"),
        "unit_G_delta": unit_summary.get("G_delta"),
        "unit_runtime_summary": unit_summary.get("runtime_summary"),
        "datasets": unit_summary.get("datasets", {}),
        "generation": generation_summary.get("generation"),
        "best_control_Q_total": generation_summary.get("best_control_Q_total"),
        "generation_units": generation_summary.get("units", []),
        "warnings": warnings,
    }

def selection_context(runtime_root: Path) -> list[dict]:
    selections = []
    for selection in sorted(runtime_root.glob("proposals/**/selection.json"))[-5:]:
        selections.append({"path": str(selection), "content": load_json(selection, {})})
    for selection in sorted(runtime_root.glob("generations/**/selection.json"))[-5:]:
        selections.append({"path": str(selection), "content": load_json(selection, {})})
    return selections


def angle_text(angle: dict | None) -> str | None:
    if not angle:
        return None
    return angle.get("angle") or angle.get("principle") or str(angle)


def default_angles(profile: dict) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    caps = profile.get("capabilities", {})
    missing = set(profile.get("missing", []))

    exploit = []
    jump = []
    fit = []
    followup = []

    if "cutoff-locality" in missing:
        exploit.append({
            "angle": "Add explicit cutoff and local pair masking to reduce nonphysical global coupling.",
            "why_now": "Current unit does not clearly expose locality in the profile.",
        })
        fit.append({"capability": "cutoff-locality", "fit": "high", "reason": "Small code change if pair distances already exist."})
    if "atomref" in missing:
        exploit.append({
            "angle": "Add element-wise reference energy / species bias before deeper architecture changes.",
            "why_now": "Improves energy scale handling without changing the training loop heavily.",
        })
        fit.append({"capability": "atomref", "fit": "high", "reason": "Usually a small model-head addition."})
    if caps.get("relative-geometry") and "message-passing" in missing:
        jump.append({
            "angle": "Move from pair pooling to local message passing over neighbor edges.",
            "why_now": "Relative geometry is already present, so edge graph features are the next structural phase.",
        })
        fit.append({"capability": "message-passing", "fit": "medium", "reason": "Requires graph batching and aggregation changes."})
    if "angular-triplets" in missing:
        jump.append({
            "angle": "Evaluate angular or three-body features for force accuracy on chemically distinct local environments.",
            "why_now": "Pair-only geometry can miss directional chemistry.",
        })
        followup.append("Find MLIP papers/repos with lightweight angular features that do not require full equivariance.")
    if "equivariance" in missing:
        jump.append({
            "angle": "Compare a minimal E(3)-equivariant candidate against the current invariant baseline.",
            "why_now": "Equivariance is a major phase jump and should be tested with implementation evidence first.",
        })
        followup.append("Search for compact equivariant MLIP repos with readable training loops.")

    if not exploit:
        exploit.append({"angle": "Use current profile to target loss, data mix, and evaluation gaps.", "why_now": "No obvious shallow capability gap found."})
    if not jump:
        jump.append({"angle": "Use repo evidence to identify the next missing physical prior.", "why_now": "No obvious jump gap found from pattern profile."})

    return exploit, jump, fit, followup


def build_handoff(profile: dict, exploit: list[dict], jump: list[dict]) -> dict:
    bottleneck = profile.get("current_bottleneck", "unknown")
    safest = "Do not launch from evidence alone; return to main agent for state-machine gated sync/smoke/launch."
    if bottleneck == "implementation":
        safest = "Spawn one implementation subagent for the target unit with bounded write scope."
    elif bottleneck == "sync":
        safest = "Use bundled remote_sync_unit.py for the next target unit."
    elif bottleneck == "smoke":
        safest = "Use bundled remote_smoke_unit.py and classify the smoke failure before repair."
    elif bottleneck == "launch":
        safest = "Launch only through bundled batch/unit launch scripts after launch_ready is confirmed."
    elif bottleneck == "repair":
        safest = "Collect logs, classify failure, and check repair budget before spawning a repair writer."
    return {
        "current_bottleneck": bottleneck,
        "safest_next_step": safest,
        "strongest_exploit_move": angle_text(exploit[0] if exploit else None),
        "strongest_jump_move": angle_text(jump[0] if jump else None),
        "what_not_to_do_next": "Do not edit runnable units, write selection.json, or launch runs from the evidence subagent.",
    }


def stable_source_key(prefix: str, value: str) -> str:
    normalized = value.strip()
    digest = hashlib.sha1(normalized.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}:{normalized}#{digest}"


def canonical_repo_ref(repo: str) -> str:
    value = str(repo).strip()
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1].removesuffix('.git')}"
    if value.endswith('.git'):
        value = value[:-4]
    return value


def ascii_slug(name: str) -> str:
    slug = []
    for ch in name:
        if ch.isascii() and (ch.isalnum() or ch in ('-', '_', '.')):
            slug.append(ch)
        elif ch in (' ', '/', '\\'):
            slug.append('_')
        else:
            slug.append(f"u{ord(ch):x}")
    compact = ''.join(slug).strip('._')
    return compact or 'source'


def stage_pdf_source(raw_path: str, source_inputs_dir: Path) -> Path:
    source = Path(raw_path).expanduser()
    source_inputs_dir.mkdir(parents=True, exist_ok=True)
    try:
        if source.exists() and source.resolve().is_relative_to(source_inputs_dir.resolve()):
            return source.resolve()
    except Exception:
        pass
    suffix = source.suffix or '.pdf'
    digest = hashlib.sha1(str(source).encode('utf-8', errors='replace')).hexdigest()[:10]
    staged_name = f"{ascii_slug(source.stem)}_{digest}{suffix}"
    staged = source_inputs_dir / staged_name
    if source.exists() and not staged.exists():
        shutil.copy2(source, staged)
    return staged


def normalize_source_plan(runtime_root: Path, source_plan: dict) -> dict:
    normalized = json.loads(json.dumps(source_plan, ensure_ascii=False))
    normalized.setdefault('version', 'source_plan.v1')
    source_inputs_dir = runtime_root / 'knowledge' / 'evidence_runs' / 'source_inputs'
    selected_pdfs = []
    for item in list(normalized.get('selected_pdfs', []) or []):
        row = dict(item) if isinstance(item, dict) else {'path': str(item)}
        raw_path = row.get('path')
        if raw_path:
            staged = stage_pdf_source(raw_path, source_inputs_dir)
            row['original_path'] = str(Path(raw_path).expanduser())
            row['path'] = str(staged)
        selected_pdfs.append(row)
    normalized['selected_pdfs'] = selected_pdfs

    selected_repos = []
    for item in list(normalized.get('selected_repos', []) or []):
        row = dict(item) if isinstance(item, dict) else {'repo': str(item)}
        repo = row.get('repo')
        if repo:
            row['original_repo'] = str(repo)
            row['repo'] = canonical_repo_ref(str(repo))
        selected_repos.append(row)
    normalized['selected_repos'] = selected_repos
    return normalized


def load_source_plan(path: Path | None, runtime_root: Path) -> dict:
    if path is None:
        return {
            "version": "source_plan.v1",
            "selected_pdfs": [],
            "selected_repos": [],
            "target_generation": None,
            "novelty_policy": DEFAULT_SOURCE_NOVELTY_POLICY,
            "source_discovery_budget": DEFAULT_SOURCE_DISCOVERY_BUDGET,
            "warning": "No source_plan was provided; external evidence collection is disabled.",
        }
    return normalize_source_plan(runtime_root, load_json(path.expanduser().resolve(), {}))


def history_path(runtime_root: Path) -> Path:
    return runtime_root / SOURCE_HISTORY


def recent_history(history: list[dict], lookback_runs: int) -> list[dict]:
    seen_runs = []
    selected = []
    for row in reversed(history):
        run_id = row.get("run_id")
        if run_id and run_id not in seen_runs:
            seen_runs.append(run_id)
        if len(seen_runs) > lookback_runs:
            break
        selected.append(row)
    return list(reversed(selected))


def run_helper_json(script_name: str, args: list[str]) -> tuple[bool, dict, str]:
    script_path = Path(__file__).resolve().with_name(script_name)
    workspace = Path(os.environ.get("OPENCLAW_WORKSPACE", str(Path.cwd()))).expanduser()
    proc = subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(workspace if workspace.exists() else Path.cwd()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )
    if proc.returncode != 0:
        return False, {"error": proc.stderr.strip() or proc.stdout.strip(), "returncode": proc.returncode}, proc.stderr.strip()
    try:
        return True, json.loads(proc.stdout), ""
    except json.JSONDecodeError as exc:
        return False, {"error": f"helper output was not JSON: {exc}", "stdout": proc.stdout[:4000]}, proc.stderr.strip()


def collect_external_sources(runtime_root: Path, package_dir: Path, source_plan: dict, run_id: str, allow_recent_source_reuse: bool, record_history: bool = True) -> dict:
    paper_dir = package_dir / "paper_artifacts"
    repo_dir = package_dir / "repo_artifacts"
    paper_dir.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)

    policy = {**DEFAULT_SOURCE_NOVELTY_POLICY, **(source_plan.get("novelty_policy") or {})}
    budget = {**DEFAULT_SOURCE_DISCOVERY_BUDGET, **(source_plan.get("source_discovery_budget") or {})}
    history = load_jsonl(history_path(runtime_root))
    recent = recent_history(history, int(policy.get("lookback_runs", 3)))
    recent_attempted_ids = {row.get("source_id") for row in recent if row.get("source_id")}
    recent_strong_ids = {
        row.get("source_id")
        for row in recent
        if row.get("source_id") and row.get("used_as_strong")
    }

    records = []
    history_rows = []
    artifacts = {"papers": [], "repos": []}
    attempted_ids = []
    successful_ids = []
    pdf_attempted = 0
    pdf_success = 0
    repo_attempted = 0
    repo_success = 0

    normalized_plan = normalize_source_plan(runtime_root, source_plan)
    selected_pdfs = list(normalized_plan.get("selected_pdfs", []) or [])[: int(budget.get("max_pdf_attempts", 6))]
    selected_repos = list(normalized_plan.get("selected_repos", []) or [])[: int(budget.get("max_repo_attempts", 4))]
    max_total = int(budget.get("max_external_source_attempts", 10))
    if len(selected_pdfs) + len(selected_repos) > max_total:
        overflow = len(selected_pdfs) + len(selected_repos) - max_total
        if overflow > 0:
            selected_repos = selected_repos[: max(0, len(selected_repos) - overflow)]
        if len(selected_pdfs) + len(selected_repos) > max_total:
            selected_pdfs = selected_pdfs[:max_total]
            selected_repos = []

    for idx, item in enumerate(selected_pdfs, start=1):
        raw_path = item.get("path") if isinstance(item, dict) else str(item)
        source_id = stable_source_key("local_pdf", str(Path(raw_path).expanduser()))
        artifact_ref = f"paper_artifact:paper_{idx:03d}"
        artifact_path = paper_dir / f"paper_{idx:03d}.json"
        attempted_ids.append(source_id)
        pdf_attempted += 1
        ok, payload, err = run_helper_json("extract_local_paper.py", [raw_path, "--pages", str(item.get("pages", 8) if isinstance(item, dict) else 8), "--max-chars", str(item.get("max_chars", 24000) if isinstance(item, dict) else 24000)])
        success = bool(ok and payload.get("readable"))
        if success:
            pdf_success += 1
        payload.update({"artifact_ref": artifact_ref, "source_id": source_id, "source_plan_entry": item})
        write_json(artifact_path, payload)
        if success:
            successful_ids.append(source_id)
        records.append(source_record(
            artifact_ref,
            "local_pdf",
            "mechanism_candidate",
            f"scripts/extract_local_paper.py {raw_path}",
            "fresh",
            success,
            str(artifact_path),
            success,
            "Fresh local PDF extraction artifact. It can support strong evidence only if a mechanism card cites this artifact and includes formula derivation / implementation mapping." if success else f"PDF extraction failed: {payload.get('error') or err}",
        ))
        artifacts["papers"].append({"artifact_ref": artifact_ref, "source_id": source_id, "path": str(artifact_path), "success": success})
        history_rows.append({"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_type": "local_pdf", "source_id": source_id, "artifact_ref": artifact_ref, "artifact": str(artifact_path), "success": success, "used_as_strong": False, "mechanism_ids": []})

    for idx, item in enumerate(selected_repos, start=1):
        repo = item.get("repo") if isinstance(item, dict) else str(item)
        source_id = stable_source_key("repo", repo)
        artifact_ref = f"repo_artifact:repo_{idx:03d}"
        artifact_path = repo_dir / f"repo_{idx:03d}.json"
        attempted_ids.append(source_id)
        repo_attempted += 1
        ok, payload, err = run_helper_json("read_github_repo.py", ["--repo", repo, "--max-files", str(item.get("max_files", 28) if isinstance(item, dict) else 28), "--max-file-chars", str(item.get("max_file_chars", 20000) if isinstance(item, dict) else 20000)])
        success = bool(ok and payload.get("key_files_content"))
        if success:
            repo_success += 1
        payload.update({"artifact_ref": artifact_ref, "source_id": source_id, "source_plan_entry": item})
        write_json(artifact_path, payload)
        if success:
            successful_ids.append(source_id)
        repo_name = payload.get("repo") or repo
        records.append(source_record(
            artifact_ref,
            "repo_code",
            "mechanism_candidate",
            f"scripts/read_github_repo.py --repo {repo}",
            "fresh",
            success,
            str(artifact_path),
            success,
            "Fresh repository deep-read artifact with README/tree/key files. It can support strong evidence only if a mechanism card cites exact files/classes/functions and maps them to current insertion points." if success else f"Repo deep-read failed: {payload.get('error') or err}",
        ))
        artifacts["repos"].append({"artifact_ref": artifact_ref, "source_id": source_id, "repo": repo_name, "path": str(artifact_path), "success": success})
        history_rows.append({"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_type": "repo_code", "source_id": source_id, "artifact_ref": artifact_ref, "artifact": str(artifact_path), "success": success, "used_as_strong": False, "mechanism_ids": []})

    # Novelty gates should prevent repeatedly depending on the same strong sources.
    # Merely attempted/read sources are tracked for diagnostics, but they do not
    # block reuse: a PDF/repo may have been readable yet not materialized into a
    # mechanism card in an earlier run.
    new_ids = [sid for sid in attempted_ids if sid not in recent_strong_ids]
    reused_ids = [sid for sid in attempted_ids if sid in recent_strong_ids]
    recent_attempted_reused_ids = [sid for sid in attempted_ids if sid in recent_attempted_ids]
    source_budget_exhausted = bool(
        pdf_attempted >= int(budget.get("max_pdf_attempts", 6))
        or repo_attempted >= int(budget.get("max_repo_attempts", 4))
        or len(attempted_ids) >= int(budget.get("max_external_source_attempts", 10))
        or int(budget.get("source_expansion_round", 1)) >= int(budget.get("max_source_expansion_rounds", 2))
    )
    source_targets_met = bool(
        pdf_success >= int(budget.get("min_successful_pdf_sources", 1))
        and repo_success >= int(budget.get("min_successful_repo_sources", 1))
    )
    source_novelty = {
        "version": "source_novelty.v1",
        "history_path": str(history_path(runtime_root)),
        "policy": policy,
        "source_discovery_budget": budget,
        "pdf_attempted_source_count": pdf_attempted,
        "pdf_successful_source_count": pdf_success,
        "repo_attempted_source_count": repo_attempted,
        "repo_successful_source_count": repo_success,
        "attempted_external_source_count": len(attempted_ids),
        "successful_external_source_count": len(successful_ids),
        "source_targets_met": source_targets_met,
        "source_budget_exhausted": source_budget_exhausted,
        "new_source_count": len(new_ids),
        "recent_reused_source_count": len(reused_ids),
        "new_source_ids": new_ids,
        "recent_reused_source_ids": reused_ids,
        "recent_attempted_reused_source_count": len(recent_attempted_reused_ids),
        "recent_attempted_reused_source_ids": recent_attempted_reused_ids,
        "reuse_semantics": "recent_reused_source_ids counts sources used as strong evidence in recent runs. recent_attempted_reused_source_ids is diagnostic only and does not fail novelty by itself.",
        "all_sources_recent_repeats": bool(attempted_ids) and not new_ids,
        "source_novelty_passed": bool(
            len(attempted_ids) >= int(policy.get("min_attempted_external_sources", 2))
            and len(new_ids) >= int(policy.get("min_new_sources", 1))
            and source_targets_met
            and (allow_recent_source_reuse or len(reused_ids) <= int(policy.get("max_recent_reused_sources", 1)))
        ),
        "allow_recent_source_reuse": allow_recent_source_reuse,
    }

    write_json(package_dir / "source_plan.json", normalized_plan)
    write_json(package_dir / "source_artifacts_index.json", artifacts)
    write_json(package_dir / "source_novelty.json", source_novelty)
    analysis_requirements = {
        "version": "source_analysis_requirements.v1",
        "instruction": "Reading artifacts is not enough. Promote a mechanism only after writing formula derivation / shape or data-flow reasoning / repo code trace / current insertion point / bounded edit. If no strong mechanism can be materialized and the source budget is not exhausted, expand the source plan with new PDFs/repos instead of writing a proposal-ready brief.",
        "required_for_strong_mechanism": [
            "mathematical_form or algorithmic update",
            "formula_derivation with reasoning steps from source expression to current-code implication",
            "tensor_shapes or data_flow for model-code mechanisms",
            "repo_code_trace with artifact, repo_file, class_or_function, implementation_pattern, mapped_current_insertion_point",
            "current_code_insertion_point",
            "bounded_edit",
            "benchmark expectation over energy/force/gap/Q/runtime/failure risk",
        ],
        "source_discovery_budget": budget,
        "source_budget_exhausted": source_budget_exhausted,
        "source_targets_met": source_targets_met,
        "artifact_index": artifacts,
    }
    write_json(package_dir / "source_analysis_requirements.json", analysis_requirements)

    if history_rows and record_history:
        hist = history_path(runtime_root)
        hist.parent.mkdir(parents=True, exist_ok=True)
        with hist.open("a", encoding="utf-8") as fh:
            for row in history_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"records": records, "artifacts": artifacts, "source_novelty": source_novelty, "analysis_requirements": analysis_requirements}


def source_record(source_id: str, source_type: str, claim_role: str, tool_or_command: str, freshness: str, success: bool, artifact: str | None, can_support_strong: bool, notes: str) -> dict:
    return {
        "source_id": source_id,
        "type": source_type,
        "claim_role": claim_role,
        "tool_or_command": tool_or_command,
        "freshness": freshness,
        "success": success,
        "artifact": artifact,
        "can_support_strong": can_support_strong,
        "notes": notes,
    }


def build_evidence_provenance(runtime_root: Path, source_unit: str, local_context: dict, profile: dict, external_collection: dict | None = None) -> dict:
    records = [
        source_record(
            "current_unit_profile",
            "local_code_profile",
            "current_code_profile",
            "scripts/profile_current_unit.py",
            "fresh",
            True,
            None,
            False,
            "Profiles the current/source runnable unit; it supports implementation fit and insertion-point reasoning but is not external mechanism evidence.",
        ),
        source_record(
            "proposal_context",
            "local_runtime_context",
            "benchmark_diagnosis",
            "read active proposal context from research_runtime/proposals",
            "fresh" if local_context.get("proposal_context", {}).get("context_path") else "missing",
            bool(local_context.get("proposal_context", {}).get("context_path")),
            local_context.get("proposal_context", {}).get("context_path"),
            False,
            "Benchmark and history context. Useful for diagnosis, not external mechanism support.",
        ),
        source_record(
            "frontier_tail",
            "local_ledger",
            "benchmark_diagnosis",
            "read research_runtime/ledger/frontier.jsonl tail",
            "fresh",
            bool(local_context.get("recent_frontier")),
            str(runtime_root / "ledger" / "frontier.jsonl"),
            False,
            "Internal outcome evidence only.",
        ),
        source_record(
            "generation_memory",
            "local_ledger",
            "long_term_outcome_memory",
            "read canonical ledger/generation_summaries, all_attempts, unit_cards, mechanism_outcomes, negative/partial patterns",
            "fresh",
            bool(local_context.get("generation_memory_ref", {}).get("completed_generation_count_for_source") or local_context.get("generation_memory_ref", {}).get("recent_attempt_count_for_source")),
            str(runtime_root / "ledger"),
            False,
            "Long-term generation memory supports diagnosis and avoids duplicate attempts; it is not external mechanism evidence.",
        ),
    ]
    for idx, path in enumerate(local_context.get("recent_briefs", []), start=1):
        records.append(source_record(
            f"prior_brief:{idx}",
            "prior_brief",
            "search_anchor",
            "read recent brief path list",
            "reused",
            True,
            path,
            False,
            "Prior briefs are anchors/hypotheses; they cannot support fresh strong mechanism claims by themselves.",
        ))
    for idx, note in enumerate(local_context.get("recent_notes", []), start=1):
        records.append(source_record(
            f"prior_note:{idx}",
            "prior_note",
            "search_anchor",
            "read research_runtime/knowledge/notes",
            "reused",
            True,
            note.get("path"),
            False,
            "Prior notes are reusable memory, not fresh verification.",
        ))
    if external_collection:
        records.extend(external_collection.get("records", []))
    return {
        "version": "evidence_provenance.v1",
        "source_unit": source_unit,
        "records": records,
        "strong_evidence_rule": "Only records with can_support_strong=true may support strong external mechanism evidence. Strong mechanism cards must cite artifact refs from this run, not just paper/repo names.",
    }


def mechanism_card_from_move(index: int, move: dict, provenance: dict) -> dict:
    mechanism_id = f"HYP-B{index:03d}"
    principle = move.get("principle") or move.get("angle") or "bootstrap hypothesis"
    return {
        "mechanism_id": mechanism_id,
        "claim_strength": "weak_hypothesis",
        "source_refs": ["current_unit_profile", "proposal_context"],
        "concrete_mechanism": principle,
        "mathematical_form": [move.get("math_form") or "Requires fresh paper/PDF/repo extraction before promotion to strong mechanism."],
        "formula_derivation": [],
        "tensor_shapes": {},
        "data_flow": [],
        "physical_principle": [],
        "chemical_regime": [],
        "repo_code_path": [],
        "repo_code_trace": [],
        "current_code_insertion_point": ["model/model.py", "model/train.py"],
        "bounded_edit": [move.get("code_pattern") or "No implementation-ready code pattern verified yet."],
        "expected_benchmark_effect": {"hypothesis": move.get("expected_effect"), "energy": None, "force": None, "gap": None, "Q": None, "runtime": None, "failure_risk": None},
        "ablation_or_control": [move.get("control") or "control replicate of current best runnable unit"],
        "promotion_requirements": [
            "fresh local PDF or paper extraction with equation/algorithm evidence",
            "formula derivation or reasoning from source expression to implementation implication",
            "fresh or verified repository deep-read with path/class/function evidence",
            "repo code trace from external implementation pattern to current unit insertion point",
            "specific current-code insertion point and bounded edit",
        ],
    }


def validate_mechanism_card(card: dict, provenance: dict) -> tuple[bool, list[str]]:
    reasons = []
    strong_sources = {row.get("source_id") for row in provenance.get("records", []) if row.get("can_support_strong") and row.get("success")}
    artifact_sources = {sid for sid in strong_sources if str(sid).startswith(("paper_artifact:", "repo_artifact:"))}
    refs = set(card.get("source_refs", []))
    if not any(ref in strong_sources for ref in refs):
        reasons.append("no strong provenance source")
    if artifact_sources and not any(ref in artifact_sources for ref in refs):
        reasons.append("no current-run artifact source_ref")
    if not card.get("mathematical_form"):
        reasons.append("missing mathematical_form")
    if not card.get("formula_derivation"):
        reasons.append("missing formula_derivation")
    if not card.get("tensor_shapes") and not card.get("data_flow"):
        reasons.append("missing tensor_shapes_or_data_flow")
    if not card.get("current_code_insertion_point"):
        reasons.append("missing current_code_insertion_point")
    if not card.get("bounded_edit"):
        reasons.append("missing bounded_edit")
    if not card.get("repo_code_path"):
        reasons.append("missing repo_code_path")
    if not card.get("repo_code_trace"):
        reasons.append("missing_repo_code_trace")
    return not reasons, reasons


def build_mechanism_cards(design_moves: list[dict], provenance: dict) -> dict:
    cards = [mechanism_card_from_move(i, move, provenance) for i, move in enumerate(design_moves, start=1)]
    for card in cards:
        strong_ready, reasons = validate_mechanism_card(card, provenance)
        card["strong_ready"] = strong_ready
        card["downgrade_reasons"] = reasons
    return {
        "version": "mechanism_cards.v1",
        "cards": cards,
        "strong_cards": [card for card in cards if card.get("strong_ready")],
        "weak_or_hypothesis_cards": [card for card in cards if not card.get("strong_ready")],
    }


def build_patch_blueprints(mechanism_cards: dict) -> dict:
    blueprints = []
    for card in mechanism_cards.get("cards", []):
        blueprints.append({
            "blueprint_id": card["mechanism_id"].replace("HYP", "PB"),
            "mechanism_id": card["mechanism_id"],
            "implementation_ready": bool(card.get("strong_ready")),
            "target_files": ["model/model.py", "model/train.py"],
            "target_insertion_points": card.get("current_code_insertion_point", []),
            "pseudo_diff": [],
            "bounded_edit": card.get("bounded_edit", []),
            "must_preserve": [
                "benchmark metric field names",
                "runnable unit entrypoint contract",
                "energy-to-force autograd consistency unless explicitly justified by strong evidence",
            ],
            "local_sanity_expectation": [
                "forward pass returns scalar energy with expected batch semantics",
                "force shape matches target force tensor when forces are requested",
            ],
            "blocked_until": card.get("downgrade_reasons", []),
        })
    return {"version": "patch_blueprints.v1", "blueprints": blueprints}


def build_evidence_quality(provenance: dict, mechanism_cards: dict, patch_blueprints: dict, benchmark_dossier: dict, source_novelty: dict | None = None, require_external_evidence: bool = False) -> dict:
    records = provenance.get("records", [])
    fresh_local_pdf_verified = any(row.get("type") == "local_pdf" and row.get("freshness") == "fresh" and row.get("success") for row in records)
    fresh_repo_code_verified = any(row.get("type") in {"repo_code", "github_repo"} and row.get("freshness") == "fresh" and row.get("success") for row in records)
    has_strong_mechanism_cards = bool(mechanism_cards.get("strong_cards"))
    has_implementation_blueprints = any(row.get("implementation_ready") for row in patch_blueprints.get("blueprints", []))
    source_novelty = source_novelty or {}
    external_attempted = int(source_novelty.get("attempted_external_source_count", 0) or 0)
    external_success = int(source_novelty.get("successful_external_source_count", 0) or 0)
    novelty_passed = bool(source_novelty.get("source_novelty_passed", not require_external_evidence))
    min_strong = int((source_novelty.get("source_discovery_budget") or {}).get("min_strong_mechanism_cards", 1) or 1)
    strong_card_count = len(mechanism_cards.get("strong_cards", []))
    source_budget_exhausted = bool(source_novelty.get("source_budget_exhausted", False))
    strong_card_target_met = strong_card_count >= min_strong
    usable_for_proposal = strong_card_target_met and (not require_external_evidence or (external_success > 0 and novelty_passed))
    usable_for_implementation = usable_for_proposal and has_implementation_blueprints
    if usable_for_implementation:
        grade = "A" if fresh_local_pdf_verified and fresh_repo_code_verified else "B"
    elif usable_for_proposal:
        grade = "B"
    elif benchmark_dossier.get("warnings"):
        grade = "D"
    else:
        grade = "C"
    return {
        "version": "evidence_quality.v1",
        "grade": grade,
        "fresh_local_pdf_verified": fresh_local_pdf_verified,
        "fresh_repo_code_verified": fresh_repo_code_verified,
        "has_mechanism_cards": bool(mechanism_cards.get("cards")),
        "has_strong_mechanism_cards": has_strong_mechanism_cards,
        "strong_mechanism_card_count": strong_card_count,
        "min_strong_mechanism_cards": min_strong,
        "strong_card_target_met": strong_card_target_met,
        "has_patch_blueprints": bool(patch_blueprints.get("blueprints")),
        "has_implementation_ready_blueprints": has_implementation_blueprints,
        "usable_for_proposal": usable_for_proposal,
        "usable_for_implementation": usable_for_implementation,
        "diagnosis_only": not usable_for_proposal,
        "require_external_evidence": require_external_evidence,
        "source_novelty": source_novelty,
        "external_attempted_source_count": external_attempted,
        "external_successful_source_count": external_success,
        "source_novelty_passed": novelty_passed,
        "source_budget_exhausted": source_budget_exhausted,
        "needs_source_expansion": bool(require_external_evidence and not usable_for_proposal and not source_budget_exhausted),
        "hard_rules": [
            "No provenance -> no strong evidence.",
            "No mechanism card -> no proposal mechanism.",
            "No formula derivation/algorithm/repo path/code trace/current insertion point -> weak evidence only.",
            "No patch blueprint -> not implementation-ready.",
            "Benchmark diagnosis is not external mechanism evidence.",
        ],
    }


def build_proposal_constraints(mechanism_cards: dict, evidence_quality: dict) -> dict:
    allowed = [card["mechanism_id"] for card in mechanism_cards.get("strong_cards", [])]
    weak = [card["mechanism_id"] for card in mechanism_cards.get("weak_or_hypothesis_cards", [])]
    return {
        "version": "proposal_constraints.v1",
        "allowed_mechanisms": allowed,
        "weak_or_hypothesis_mechanisms": weak,
        "blocked_mechanisms": [
            "unproven model-family label without formula derivation and code trace",
            "force-only proposal when energy/gap/Q context is available",
            "implementation rewrite without patch blueprint",
        ],
        "proposal_can_use_as_strong_evidence": evidence_quality.get("usable_for_proposal", False),
        "required_sections_in_proposal": [
            "mechanism_refs",
            "evidence_refs",
            "historical_relation",
            "why_not_duplicate",
            "files_to_edit",
            "code_insertion_points",
            "minimal_edit_plan",
            "implementation_checklist",
        ],
    }


def build_audit_report(provenance: dict, mechanism_cards: dict, evidence_quality: dict) -> dict:
    issues = []
    if not evidence_quality.get("fresh_local_pdf_verified"):
        issues.append("No fresh local PDF extraction was recorded in this package.")
    if not evidence_quality.get("fresh_repo_code_verified"):
        issues.append("No fresh repository code deep-read was recorded in this package.")
    if not evidence_quality.get("has_strong_mechanism_cards"):
        issues.append("No mechanism card passed strong-evidence validation.")
    if evidence_quality.get("require_external_evidence") and not evidence_quality.get("external_successful_source_count"):
        issues.append("External evidence was required, but no PDF/repo source was successfully read.")
    if evidence_quality.get("require_external_evidence") and not evidence_quality.get("source_novelty_passed"):
        issues.append("Source novelty policy failed; the run did not add enough new external sources or repeated recent sources too heavily.")
    if not evidence_quality.get("usable_for_proposal"):
        issues.append("Package is not proposal-ready; it lacks at least one mechanism with provenance, formula derivation, code trace, insertion point, and bounded edit.")
    return {
        "version": "audit_report.v1",
        "issues": issues,
        "mechanism_card_count": len(mechanism_cards.get("cards", [])),
        "strong_mechanism_card_count": len(mechanism_cards.get("strong_cards", [])),
        "provenance_record_count": len(provenance.get("records", [])),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_run(question: str, mode: str, source_unit: str, runtime_root: Path, package_dir: Path, run_id: str, source_plan: dict, require_external_evidence: bool, allow_recent_source_reuse: bool, record_history: bool = True) -> dict:
    profiler = load_local_module("profile_current_unit.py")
    unit_root = resolve_unit(runtime_root, source_unit)
    profile = profiler.profile_unit(unit_root)
    exploit, jump, implementation_fit, followup = default_angles(profile)
    handoff = build_handoff(profile, exploit, jump)
    round_state = load_json(runtime_root / "ledger" / "round_state.json", {})
    proposal_context = load_full_context(runtime_root, source_unit, round_state)
    local_context = {
        "source_unit": source_unit,
        "unit_root": str(unit_root),
        "runtime_root": str(runtime_root),
        "round_state": round_state,
        "selection_files": selection_context(runtime_root),
        "recent_frontier": read_jsonl_tail(runtime_root / "ledger" / "frontier.jsonl", 12),
        "recent_briefs": list_recent_briefs(runtime_root / "knowledge" / "briefs", 5),
        "recent_notes": read_recent_notes(runtime_root / "knowledge" / "notes", 3),
        "proposal_context": proposal_context,
    }
    gen_memory = generation_memory(runtime_root, source_unit, round_state)
    local_context["generation_memory_ref"] = compact_generation_memory(gen_memory)

    required_benchmark_strings = ["mixed_force_mae", "mixed_energy_mae", "gap_penalty"]
    serialized_context = json.dumps(local_context, ensure_ascii=False)
    if not any(token in serialized_context for token in required_benchmark_strings):
        print("[warning] local_context appears benchmark-incomplete; evidence quality may be force-biased.")
    benchmark_dossier = benchmark_dossier_from_context(local_context)
    external_collection = collect_external_sources(runtime_root, package_dir, source_plan, run_id, allow_recent_source_reuse, record_history=record_history)
    capability_gap = {
        "current_can_do": profile.get("present", []),
        "current_missing": profile.get("missing", []),
        "external_requirements": [],
        "gap_level": "unknown until paper/repo evidence is attached",
        "process_bottleneck": profile.get("current_bottleneck"),
    }
    design_moves = [
        {
            "principle": angle.get("angle"),
            "math_form": "Requires evidence fill-in from local PDFs/papers/repos.",
            "code_pattern": "Requires key-file repo evidence before implementation.",
            "expected_effect": angle.get("why_now"),
            "control": "control replicate of current best runnable unit",
        }
        for angle in (exploit + jump)
    ]

    run = {
        "version": "evidence-run.v7-package-deepread",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "brief_path": None,
        "evidence_run_path": None,
        "question": question,
        "mode": mode,
        "source_unit": source_unit,
        "runtime_root": str(runtime_root),
        "current_bottleneck": profile.get("current_bottleneck", "unknown"),
        "strongest_exploit_angle": angle_text(exploit[0] if exploit else None),
        "strongest_jump_angle": angle_text(jump[0] if jump else None),
        "local_context": local_context,
        "current_unit_profile": {"see": "current_code_profile.json", "summary": {"present": profile.get("present", []), "missing": profile.get("missing", []), "current_bottleneck": profile.get("current_bottleneck")}},
        "benchmark_dossier": {"see": "benchmark_diagnosis.json", "warnings": benchmark_dossier.get("warnings", [])},
        "generation_memory": compact_generation_memory(gen_memory),
        "strong_evidence": [],
        "weak_but_relevant": [],
        "background_context": {"see": "generation_memory.json and evidence_run.local_context", "note": "Full local context is retained for audit in evidence_run.json; the rendered brief references it but does not repeat it."},
        "mathematical_forms": [
            "Bootstrap profile only. A proposal-ready package must include equation/algorithm extraction plus formula derivation into implementation consequences.",
        ],
        "physical_principles": [
            "Bootstrap profile only. Fill locality, symmetry, conservation, force consistency, and long-range implications from evidence.",
        ],
        "chemical_regime": [
            "Bootstrap profile only. Fill molecular/materials regime, angular chemistry, many-body effects, and composition baseline evidence.",
        ],
        "textual_evidence": [],
        "code_evidence": [
            {"type": "current_unit_profile", "profile": profile},
        ],
        "relevant_papers": [],
        "relevant_repos": [],
        "capability_gap": capability_gap,
        "implementable_design_moves": design_moves,
        "exploit_angles": exploit if mode in ("exploit", "balanced") else [],
        "jump_angles": jump if mode in ("jump", "balanced") else [],
        "implementation_fit": implementation_fit,
        "risks_or_mismatches": [
            "This bootstrap run profiles local code and process state only; a complete dossier must add verified local PDF/paper and repo/code evidence.",
        "Do not justify proposal direction from force-only context; inspect benchmark_dossier for energy/gap/Q/runtime/control fields.",
            "If local PDF extraction fails, do not count that PDF as locally verified evidence.",
            "If repo deep-read fails, record it as unreadable instead of treating README guesses as code evidence.",
        ],
        "handoff_summary": handoff,
        "followup_queries": followup,
        "confidence": {
            "grade": "D",
            "reason": "Bootstrap profile only; paper and repo evidence still need to be attached by the evidence subagent.",
            "current_code_profile": "medium",
            "paper_evidence": "none",
            "repo_evidence": "none",
            "local_context_completeness": "low" if benchmark_dossier.get("warnings") else "benchmark-centric",
        },
    }
    evidence_provenance = build_evidence_provenance(runtime_root, source_unit, local_context, profile, external_collection)
    mechanism_cards = build_mechanism_cards(design_moves, evidence_provenance)
    patch_blueprints = build_patch_blueprints(mechanism_cards)
    evidence_quality = build_evidence_quality(evidence_provenance, mechanism_cards, patch_blueprints, benchmark_dossier, external_collection.get("source_novelty"), require_external_evidence)
    proposal_constraints = build_proposal_constraints(mechanism_cards, evidence_quality)
    audit_report = build_audit_report(evidence_provenance, mechanism_cards, evidence_quality)

    run["evidence_quality"] = evidence_quality
    run["evidence_provenance"] = evidence_provenance
    run["mechanism_cards"] = mechanism_cards
    run["patch_blueprints"] = patch_blueprints
    run["proposal_constraints"] = proposal_constraints
    run["audit_report"] = audit_report
    run["source_plan"] = source_plan
    run["source_artifacts"] = external_collection.get("artifacts", {})
    run["source_novelty"] = external_collection.get("source_novelty", {})
    run["source_analysis_requirements"] = external_collection.get("analysis_requirements", {})
    run["current_code_profile"] = profile
    run["benchmark_diagnosis"] = benchmark_dossier
    run["generation_memory_full"] = gen_memory
    run["mechanism_ledger"] = mechanism_cards.get("cards", [])

    for key in REQUIRED_KEYS:
        run.setdefault(key, [] if key.endswith("s") or key.endswith("angles") or key.endswith("evidence") else None)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a unified MLIP-Evidence run and markdown dossier.")
    parser.add_argument("--question", required=True, help="Evidence question.")
    parser.add_argument("--mode", choices=["exploit", "jump", "balanced"], required=True)
    parser.add_argument("--source-unit", required=True, help="base_unit, generation_###/proposal_###, or unit path.")
    parser.add_argument("--runtime-root", required=True, help="Path to research_runtime.")
    parser.add_argument("--source-plan", default=None, help="JSON plan listing selected PDFs/repos and novelty policy.")
    parser.add_argument("--require-external-evidence", action="store_true", help="Require successful, novel external PDF/repo evidence for proposal-ready quality.")
    parser.add_argument("--allow-recent-source-reuse", action="store_true", help="Allow source novelty policy to pass despite recent source reuse.")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON without writing files.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    runtime_root = resolve_runtime_root(args.runtime_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"evidence_run_{timestamp}"
    run_root = runtime_root / "knowledge" / "evidence_runs"
    package_dir = (Path("/tmp/MLIP-Evidence_dry_runs") / run_id) if args.dry_run else (run_root / run_id)
    package_dir.mkdir(parents=True, exist_ok=True)
    source_plan = load_source_plan(Path(args.source_plan) if args.source_plan else None, runtime_root)
    run = build_run(args.question, args.mode, args.source_unit, runtime_root, package_dir, run_id, source_plan, args.require_external_evidence, args.allow_recent_source_reuse, record_history=not args.dry_run)

    if args.dry_run:
        print(json.dumps(run, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    brief_dir = runtime_root / "knowledge" / "briefs"
    package_dir.mkdir(parents=True, exist_ok=True)
    brief_dir.mkdir(parents=True, exist_ok=True)
    (runtime_root / "knowledge" / "paper_cards.jsonl").touch(exist_ok=True)
    (runtime_root / "knowledge" / "repo_cards.jsonl").touch(exist_ok=True)

    package_brief_path = package_dir / "evidence_brief.md"
    legacy_brief_path = brief_dir / f"evidence_brief_{timestamp}.md"
    run_path = package_dir / "evidence_run.json"
    legacy_run_path = run_root / f"{run_id}.json"
    run["run_id"] = run_id
    run["brief_path"] = str(legacy_brief_path)
    run["package_brief_path"] = str(package_brief_path)
    run["evidence_run_path"] = str(run_path)
    run["evidence_package_dir"] = str(package_dir)

    write_json(package_dir / "source_plan.json", run["source_plan"])
    write_json(package_dir / "source_novelty.json", run["source_novelty"])
    write_json(package_dir / "source_analysis_requirements.json", run["source_analysis_requirements"])
    write_json(package_dir / "evidence_quality.json", run["evidence_quality"])
    write_json(package_dir / "evidence_provenance.json", run["evidence_provenance"])
    write_json(package_dir / "current_code_profile.json", run["current_code_profile"])
    write_json(package_dir / "benchmark_diagnosis.json", run["benchmark_diagnosis"])
    write_json(package_dir / "generation_memory.json", run["generation_memory_full"])
    write_json(package_dir / "mechanism_cards.json", run["mechanism_cards"])
    write_json(package_dir / "patch_blueprints.json", run["patch_blueprints"])
    write_json(package_dir / "proposal_constraints.json", run["proposal_constraints"])
    write_json(package_dir / "audit_report.json", run["audit_report"])
    write_json(run_path, run)
    write_json(legacy_run_path, {
        "run_id": run_id,
        "evidence_package_dir": str(package_dir),
        "evidence_run_path": str(run_path),
        "brief_path": str(legacy_brief_path),
        "evidence_quality": run.get("evidence_quality"),
        "source_novelty": run.get("source_novelty"),
    })

    builder = load_local_module("build_brief.py")
    brief_text = builder.build_markdown(run)
    package_brief_path.write_text(brief_text, encoding="utf-8")
    legacy_brief_path.write_text(brief_text, encoding="utf-8")

    print(json.dumps({
        "brief_path": str(legacy_brief_path),
        "package_brief_path": str(package_brief_path),
        "evidence_package_dir": str(package_dir),
        "evidence_run": str(run_path),
        "question": args.question,
        "mode": args.mode,
        "source_unit": args.source_unit,
        "current_bottleneck": run.get("current_bottleneck"),
        "strongest_exploit_angle": run.get("strongest_exploit_angle"),
        "strongest_jump_angle": run.get("strongest_jump_angle"),
        "evidence_quality": run.get("evidence_quality"),
        "source_novelty": run.get("source_novelty"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
