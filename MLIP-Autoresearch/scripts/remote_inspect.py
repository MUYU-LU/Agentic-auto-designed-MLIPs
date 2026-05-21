from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from runtime_common import (
    GENERATIONS,
    LEDGER,
    PROPOSALS,
    STAGING_RUNTIME_ROOT,
    load_config,
    load_implementation_status,
    load_json,
    load_round_state,
    load_run_status,
    remote_runtime_root,
    remote_status_paths,
    fetch_remote_json,
    q_fields_for_unit,
    run_remote_bash,
)

UNIT_RE = re.compile(r"generation_\d+/proposal_\d+")
REQUIRED_PROPOSAL_SECTIONS = (
    "## mechanism_refs",
    "## evidence_refs",
    "## files_to_edit",
    "## code_insertion_points",
    "## minimal_edit_plan",
    "## implementation_checklist",
)
REQUIRED_PROPOSAL_METADATA = (
    "- family:",
    "- phase:",
    "- jump_type:",
    "- budget_class:",
    "- expected_capability_gain:",
)
ROUND_STATE_AUTHORITY_FIELDS = (
    "current_generation",
    "workflow_state",
    "continuation_source_unit",
    "active_evidence_brief",
    "evidence_for_source_unit",
    "proposal_context_file",
    "active_proposal_directory",
    "active_selection_file",
    "materialized_units_root",
    "proposal_directory_ready",
    "proposal_source",
    "proposal_writer_session",
    "active_writer_unit",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generation_sort_key(name: str | None) -> int:
    if not name or not name.startswith("generation_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return -1


def next_generation_name(generation: str | None) -> str | None:
    number = generation_sort_key(generation)
    if number < 0:
        return None
    return f"generation_{number + 1:03d}"


def infer_current_generation(round_state_local: dict, round_state_remote: dict) -> str | None:
    local_gen = round_state_local.get("current_generation")
    remote_gen = round_state_remote.get("current_generation")
    candidates = []
    if local_gen:
        candidates.append(local_gen)
    if remote_gen and remote_gen not in candidates:
        candidates.append(remote_gen)
    for generation_root in GENERATIONS.glob("generation_*"):
        if generation_root.is_dir() and list(generation_root.glob("proposal_*")):
            candidates.append(generation_root.name)
    if candidates:
        return max(candidates, key=generation_sort_key)
    return None


def active_openclaw_writer_unit() -> str | None:
    runs_path = Path.home() / ".openclaw" / "subagents" / "runs.json"
    runs = load_json(runs_path, {}).get("runs", {})
    for run in runs.values():
        status = str(run.get("status") or run.get("state") or "").lower()
        if status in {"completed", "failed", "cancelled", "terminated", "done", "success"}:
            continue
        role_blob = json.dumps(
            {
                "task": run.get("task"),
                "kind": run.get("kind"),
                "role": run.get("role"),
                "agent_role": run.get("agent_role"),
                "title": run.get("title"),
            },
            ensure_ascii=False,
        )
        if not any(
            marker in role_blob.lower()
            for marker in [
                "implementation subagent",
                "repair subagent",
                "edit exactly one target runnable unit",
                "implementation / repair subagent",
            ]
        ):
            continue
        match = UNIT_RE.search(role_blob)
        if match:
            return match.group(0)
    return None


def proposal_dir_for_source(source_unit: str | None, target_generation: str | None = None) -> Path | None:
    if not source_unit:
        return None
    source_slug = source_unit.replace("/", "_")
    if target_generation:
        return PROPOSALS / f"{target_generation}_from_{source_slug}_continuation"
    return PROPOSALS / f"{source_slug}_continuation"


def reusable_evidence_brief(source_unit: str | None, target_generation: str | None) -> str | None:
    if not source_unit:
        return None
    knowledge_root = STAGING_RUNTIME_ROOT / "knowledge"
    runs_root = knowledge_root / "evidence_runs"
    briefs_root = knowledge_root / "briefs"
    if not runs_root.exists():
        return None

    def run_mentions_target(run_dir: Path) -> bool:
        if not target_generation:
            return True
        for candidate_name in ("evidence_run.json", "source_plan.json"):
            candidate = run_dir / candidate_name
            if not candidate.exists():
                continue
            payload = load_json(candidate, {})
            if payload.get("target_generation") == target_generation:
                return True
            if target_generation in json.dumps(payload, ensure_ascii=False):
                return True
        return False

    run_dirs = sorted((path for path in runs_root.glob("evidence_run_*") if path.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        evidence_run = load_json(run_dir / "evidence_run.json", {})
        if evidence_run.get("source_unit") != source_unit:
            continue
        quality = load_json(run_dir / "evidence_quality.json", {})
        if not quality.get("usable_for_proposal") or quality.get("diagnosis_only"):
            continue
        if not run_mentions_target(run_dir):
            continue
        run_id = run_dir.name.removeprefix("evidence_run_")
        for brief in (briefs_root / f"evidence_brief_{run_id}.md", run_dir / "evidence_brief.md"):
            if brief.exists():
                return str(brief)
    return None


def evidence_gate(round_state: dict, source_unit: str | None, target_generation: str | None) -> dict | None:
    if not source_unit:
        return {"kind": "evidence", "source_unit": None, "target_generation": target_generation, "reason": "missing_source_unit"}

    brief_raw = round_state.get("active_evidence_brief")
    evidence_source = round_state.get("evidence_for_source_unit")
    brief_path = Path(brief_raw) if brief_raw else None
    if not brief_path or not brief_path.exists() or evidence_source != source_unit:
        reusable_brief = reusable_evidence_brief(source_unit, target_generation)
        if reusable_brief:
            return {
                "kind": "activate_evidence",
                "source_unit": source_unit,
                "target_generation": target_generation,
                "mode": round_state.get("evidence_mode") or "balanced",
                "reason": "reusable_proposal_ready_evidence_found",
                "brief": reusable_brief,
                "active_evidence_brief": str(brief_path) if brief_path else None,
                "evidence_for_source_unit": evidence_source,
            }
        return {
            "kind": "evidence",
            "source_unit": source_unit,
            "target_generation": target_generation,
            "mode": round_state.get("evidence_mode") or "balanced",
            "reason": "missing_or_mismatched_active_evidence_brief",
            "active_evidence_brief": str(brief_path) if brief_path else None,
            "evidence_for_source_unit": evidence_source,
        }
    return None


def proposal_files_for_dir(proposal_dir: Path | None) -> list[Path]:
    if not proposal_dir or not proposal_dir.exists():
        return []
    return sorted(proposal_dir.glob("proposal_*.md"))


def proposal_file_is_ready(path: Path) -> tuple[bool, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False, ["unreadable"]
    missing = [token for token in REQUIRED_PROPOSAL_METADATA if token not in text]
    missing.extend(token for token in REQUIRED_PROPOSAL_SECTIONS if token not in text)
    return not missing, missing


def proposal_readiness(proposal_files: list[Path]) -> dict:
    ready = []
    invalid = []
    for path in proposal_files:
        ok, missing = proposal_file_is_ready(path)
        if ok:
            ready.append(path)
        else:
            invalid.append({"file": str(path), "missing": missing})
    return {"ready_count": len(ready), "invalid": invalid}


def proposal_gate(round_state: dict, proposal_dir: Path | None, source_unit: str | None, target_generation: str | None) -> dict | None:
    context_raw = round_state.get("proposal_context_file")
    context_path = Path(context_raw) if context_raw else None
    proposal_files = proposal_files_for_dir(proposal_dir)
    readiness = proposal_readiness(proposal_files)
    ready = bool(round_state.get("proposal_directory_ready"))
    proposal_source = round_state.get("proposal_writer_session") or round_state.get("proposal_source")

    if not context_path or not context_path.exists():
        return {
            "kind": "prepare_proposal_context",
            "source_unit": source_unit,
            "target_generation": target_generation,
            "proposal_directory": str(proposal_dir) if proposal_dir else None,
            "proposal_context_file": str(context_path) if context_path else None,
            "proposal_files_count": len(proposal_files),
            "proposal_ready_count": readiness["ready_count"],
            "invalid_proposals": readiness["invalid"],
            "proposal_directory_ready": ready,
            "proposal_source": proposal_source,
            "reason": "proposal_context_missing",
        }

    if len(proposal_files) < 10 or readiness["ready_count"] < 10 or not ready or not proposal_source:
        return {
            "kind": "proposal_writing",
            "source_unit": source_unit,
            "target_generation": target_generation,
            "proposal_directory": str(proposal_dir) if proposal_dir else None,
            "proposal_context_file": str(context_path),
            "proposal_files_count": len(proposal_files),
            "proposal_ready_count": readiness["ready_count"],
            "invalid_proposals": readiness["invalid"],
            "proposal_directory_ready": ready,
            "proposal_source": proposal_source,
            "reason": "proposal_files_or_provenance_not_ready",
        }
    return None


def selected_proposals_from_selection(selection_path: Path | None) -> list[dict]:
    if not selection_path or not selection_path.exists():
        return []
    selection = load_json(selection_path, {})
    rows = selection.get("selected_proposals", [])
    selected = []
    for row in rows:
        proposal_id = row.get("id") if isinstance(row, dict) else None
        proposal_path = row.get("path") if isinstance(row, dict) else None
        if proposal_id:
            selected.append({"id": str(proposal_id), "path": proposal_path})
    return selected


def missing_selected_units(generation: str | None, selection_path: Path | None) -> list[str]:
    if not generation:
        return []
    selected = selected_proposals_from_selection(selection_path)
    if not selected:
        return []
    generation_root = GENERATIONS / generation
    missing = []
    for row in selected:
        proposal_id = row["id"]
        if not (generation_root / proposal_id).exists():
            missing.append(f"{generation}/{proposal_id}")
    return missing


def selection_target_generation(selection_path: Path | None, fallback_generation: str | None) -> str | None:
    if not selection_path or not selection_path.exists():
        return fallback_generation
    selection = load_json(selection_path, {})
    target = selection.get("target_generation")
    if isinstance(target, str) and target.startswith("generation_"):
        return target
    return fallback_generation


def completed_generation_memory_missing(completed_generation: str | None) -> bool:
    if not completed_generation:
        return False
    summary = LEDGER / "generation_summaries" / f"{completed_generation}.json"
    report = LEDGER / "generation_reports" / f"{completed_generation}.md"
    generation_root = GENERATIONS / completed_generation
    if not summary.exists() or not report.exists():
        return True
    if not generation_root.exists():
        return False
    unit_roots = sorted(path for path in generation_root.glob("proposal_*") if path.is_dir())
    unit_cards_root = LEDGER / "unit_cards"
    for unit_root in unit_roots:
        expected = unit_cards_root / f"{completed_generation}__{unit_root.name}.json"
        if not expected.exists():
            return True
    return False


def round_complete_followup_step(completed_generation: str | None) -> dict:
    next_gen = next_generation_name(completed_generation)
    if not next_gen:
        return {"kind": "idle"}
    round_state = load_round_state(STAGING_RUNTIME_ROOT)

    if completed_generation and completed_generation_memory_missing(completed_generation):
        return {
            "kind": "summarize_generation_memory",
            "completed_generation": completed_generation,
            "target_generation": next_gen,
        }

    decision_path = LEDGER / f"{next_gen}_continuation_decision.json"
    if not decision_path.exists():
        return {"kind": "decide_continuation", "completed_generation": completed_generation, "target_generation": next_gen}

    decision = load_json(decision_path, {})
    source_unit = decision.get("selected_continuation_source")
    proposal_dir = proposal_dir_for_source(source_unit, next_gen)
    evidence_step = evidence_gate(round_state, source_unit, next_gen)
    if evidence_step:
        evidence_step["completed_generation"] = completed_generation
        return evidence_step

    proposal_step = proposal_gate(round_state, proposal_dir, source_unit, next_gen)
    if proposal_step:
        proposal_step["completed_generation"] = completed_generation
        return proposal_step

    selection_path = proposal_dir / "selection.json"
    if not selection_path.exists():
        return {
            "kind": "select_proposals",
            "completed_generation": completed_generation,
            "target_generation": next_gen,
            "source_unit": source_unit,
            "proposal_directory": str(proposal_dir),
        }

    missing_units = missing_selected_units(next_gen, selection_path)
    if missing_units:
        return {
            "kind": "materialize_generation",
            "completed_generation": completed_generation,
            "target_generation": next_gen,
            "source_unit": source_unit,
            "proposal_directory": str(proposal_dir),
            "selection_file": str(selection_path),
            "missing_units": missing_units,
        }

    return {
        "kind": "materialization_ready",
        "current_generation": next_gen,
        "target_generation": next_gen,
        "source_unit": source_unit,
        "proposal_directory": str(proposal_dir),
        "selection_file": str(selection_path),
        "materialized_units_root": str(GENERATIONS / next_gen),
    }


def advance_generation_state_step(current_generation: str | None, round_state: dict) -> dict:
    if not current_generation:
        return {"kind": "idle"}
    generation_root = GENERATIONS / current_generation
    unit_roots = sorted(generation_root.glob("proposal_*"))
    source_unit = None
    if unit_roots:
        source_unit = load_json(unit_roots[0] / "unit_meta.json", {}).get("source_unit")
    proposal_dir = proposal_dir_for_source(source_unit, current_generation)
    selection_file = proposal_dir / "selection.json" if proposal_dir else None
    missing_units = missing_selected_units(current_generation, selection_file)
    if missing_units:
        return {
            "kind": "materialize_generation",
            "current_generation": current_generation,
            "target_generation": current_generation,
            "source_unit": source_unit,
            "proposal_directory": str(proposal_dir) if proposal_dir else None,
            "selection_file": str(selection_file) if selection_file else None,
            "materialized_units_root": str(generation_root),
            "missing_units": missing_units,
        }
    expected_root = str(generation_root.resolve())
    materialized_raw = round_state.get("materialized_units_root")
    materialized_root = str(Path(materialized_raw).expanduser().resolve()) if materialized_raw else None
    if materialized_root != expected_root:
        return {
            "kind": "materialization_ready",
            "current_generation": current_generation,
            "target_generation": current_generation,
            "source_unit": source_unit,
            "proposal_directory": str(proposal_dir) if proposal_dir else None,
            "selection_file": str(selection_file) if selection_file else None,
            "materialized_units_root": str(generation_root),
            "reason": "round_state_missing_materialized_units_root",
        }
    return {
        "kind": "generation_active",
        "current_generation": current_generation,
        "target_generation": current_generation,
        "source_unit": source_unit,
        "proposal_directory": str(proposal_dir) if proposal_dir else None,
        "selection_file": str(selection_file) if selection_file else None,
        "materialized_units_root": str(generation_root),
        "reason": "round_state_current_generation_stale",
    }


def inspect_remote_unit(unit_name: str, config: dict) -> dict:
    paths = remote_status_paths(unit_name, config)
    remote_path = str(Path(remote_runtime_root(config)) / "generations" / unit_name)
    status = fetch_remote_json(paths["implementation_status"], config=config) or {}
    run = fetch_remote_json(paths["run_status"], config=config) or {}
    ps_script = f"""
python3 - <<'PY'
import json, pathlib, subprocess
unit = pathlib.Path({json.dumps(remote_path)})
pid_file = unit / 'outputs' / 'launch.pid'
launch_log = unit / 'outputs' / 'launch.log'
result = {{
  'remote_path': str(unit),
  'remote_exists': unit.exists(),
  'main_exists': (unit / 'main.py').exists(),
  'config_exists': (unit / 'config.json').exists(),
  'launch_log_exists': launch_log.exists(),
  'pid': None,
  'pid_alive': False,
  'metrics_rmd17': (unit / 'outputs' / 'rmd17' / 'benchmark_metrics.json').exists(),
  'metrics_iso17': (unit / 'outputs' / 'iso17' / 'benchmark_metrics.json').exists(),
}}
if pid_file.exists():
  try:
    pid = pid_file.read_text().strip()
    result['pid'] = pid
    if pid:
      check = subprocess.run(['bash', '-lc', f'kill -0 {{pid}}'], capture_output=True)
      result['pid_alive'] = check.returncode == 0
  except Exception:
    pass
print(json.dumps(result))
PY
"""
    proc = run_remote_bash(ps_script, config=config, capture_output=True)
    extra = {}
    if proc.returncode == 0:
        try:
            extra = json.loads((proc.stdout or "{}").strip())
        except json.JSONDecodeError:
            extra = {"inspect_error": proc.stdout}
    else:
        extra = {"inspect_error": proc.stderr}
    return {
        "implementation_status_remote": status,
        "run_status_remote": run,
        **extra,
    }


def status_mismatch_for_unit(local_impl: dict, remote_impl: dict, local_run: dict, remote_run: dict) -> bool:
    if not (local_impl or {}).get("remote_synced") and (local_run or {}).get("run_state") in {None, "not_started"}:
        return False
    impl_keys = ["implementation_state", "remote_synced", "remote_smoke_passed", "remote_path"]
    run_keys = ["run_state", "pid", "failure_class"]
    for key in impl_keys:
        if (local_impl or {}).get(key) != (remote_impl or {}).get(key):
            return True
    for key in run_keys:
        if (local_run or {}).get(key) != (remote_run or {}).get(key):
            return True
    return False


def next_action_for_unit(unit_name: str, impl: dict, run: dict, remote: dict, config: dict) -> str:
    run_state = run.get("run_state")
    if run_state == "terminal_success":
        return "done"
    if impl.get("implementation_state") == "abandoned" or run_state == "terminal_abandoned":
        return "terminal_abandoned"
    if impl.get("implementation_state") == "implementation_needed":
        return "implementation"
    if impl.get("implementation_state") == "repairing":
        return "wait_for_repair_writer"
    if impl.get("implementation_state") == "implemented" and not impl.get("remote_synced"):
        return "sync"
    if impl.get("implementation_state") == "implemented" and impl.get("remote_synced") and not impl.get("remote_smoke_passed"):
        if impl.get("last_failure_class"):
            return "repair"
        return "smoke"
    if impl.get("implementation_state") == "launch_ready" and run_state in {"not_started", "terminal_failure", "terminal_timeout"}:
        impl_updated = impl.get("last_updated_utc")
        run_changed = run.get("last_state_change_utc")
        repaired_after_failure = False
        if impl_updated and run_changed:
            try:
                repaired_after_failure = datetime.fromisoformat(impl_updated.replace("Z", "+00:00")) > datetime.fromisoformat(run_changed.replace("Z", "+00:00"))
            except Exception:
                repaired_after_failure = False
        if run_state == "not_started" or repaired_after_failure:
            return "launch_candidate"
    if run_state == "running":
        if remote.get("pid_alive"):
            launched_at = run.get("launched_at_utc")
            if launched_at:
                try:
                    started = datetime.fromisoformat(launched_at.replace("Z", "+00:00"))
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                    if elapsed > int(config.get("policies", {}).get("max_run_wallclock_sec", 3600)):
                        return "timeout_stop"
                except Exception:
                    pass
            return "running"
        return "collect"
    if run_state in {"terminal_failure", "terminal_timeout"}:
        max_repairs = int(config.get("policies", {}).get("max_repair_attempts_per_unit", 2))
        max_same = int(config.get("policies", {}).get("max_same_failure_class_repairs", 1))
        if int(impl.get("repair_attempts", 0) or 0) >= max_repairs:
            return "abandon"
        if int(impl.get("same_failure_class_repairs", 0) or 0) >= max_same:
            return "abandon"
        return "repair"
    return "waiting"


def unit_q_fields(unit_root: Path) -> dict:
    fields = q_fields_for_unit(unit_root, load_json(STAGING_RUNTIME_ROOT / "benchmark" / "anchors.json", {}))
    return {
        "q_rmd17": fields.get("Q_rmd17"),
        "q_iso17": fields.get("Q_iso17"),
        "q_mad10k": fields.get("Q_mad10k"),
        "q_total": fields.get("Q_total"),
        "benchmark_version": fields.get("benchmark_version"),
    }


def is_terminal(item: dict) -> bool:
    if item.get("next_action") == "launch_candidate":
        return False
    state = item["run_status_local"].get("run_state")
    return state in {"terminal_success", "terminal_failure", "terminal_timeout", "terminal_abandoned"}


def authoritative_round_state_mismatches(local_state: dict, remote_state: dict) -> list[dict]:
    mismatches = []
    for field in ROUND_STATE_AUTHORITY_FIELDS:
        if local_state.get(field) != remote_state.get(field):
            mismatches.append({"scope": "round", "field": field, "local": local_state.get(field), "remote": remote_state.get(field)})
    return mismatches


def selected_set_is_complete(current_generation: str | None, round_state: dict) -> tuple[bool, list[str]]:
    selection_file_raw = round_state.get("active_selection_file")
    selection_file = Path(selection_file_raw) if selection_file_raw else None
    target_generation = selection_target_generation(selection_file, current_generation)
    missing = missing_selected_units(target_generation, selection_file)
    return (not missing, missing)


def launch_batch_is_legal(units: list[dict], current_generation: str | None, round_state: dict) -> bool:
    complete, _missing = selected_set_is_complete(current_generation, round_state)
    if not complete:
        return False
    nonterminal = [u for u in units if not is_terminal(u)]
    if not nonterminal:
        return False
    for item in nonterminal:
        impl = item["implementation_status_local"]
        if impl.get("implementation_state") != "launch_ready":
            return False
        if not impl.get("remote_synced") or not impl.get("remote_smoke_passed"):
            return False
        if item.get("mismatch"):
            return False
    return True


def first_unit_with_action(units: list[dict], actions: set[str]) -> dict | None:
    for item in units:
        if item["next_action"] in actions:
            return item
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config()
    round_state_local = load_round_state(STAGING_RUNTIME_ROOT)
    round_state_remote = fetch_remote_json(f"{remote_runtime_root(config)}/ledger/round_state.json", config=config) or {}
    current_generation = infer_current_generation(round_state_local, round_state_remote)
    round_state_generation_stale = current_generation != round_state_local.get("current_generation")
    active_writer = round_state_local.get("active_writer_unit") or active_openclaw_writer_unit()

    units = []
    any_mismatch = False
    mismatch_details = []

    if current_generation and (GENERATIONS / current_generation).exists():
        for unit_root in sorted((GENERATIONS / current_generation).glob("proposal_*")):
            unit_name = f"{current_generation}/{unit_root.name}"
            impl_local = load_implementation_status(unit_root)
            run_local = load_run_status(unit_root)
            should_inspect_remote = bool(impl_local.get("remote_synced")) or run_local.get("run_state") in {
                "running",
                "terminal_success",
                "terminal_failure",
                "terminal_timeout",
            }
            remote = inspect_remote_unit(unit_name, config) if should_inspect_remote else {}
            impl_remote = remote.get("implementation_status_remote", {}) or {}
            run_remote = remote.get("run_status_remote", {}) or {}
            mismatch = status_mismatch_for_unit(impl_local, impl_remote, run_local, run_remote)
            # If a new generation has been materialized locally but the global
            # round_state still points at the previous generation, the next safe
            # action must be the round-level materialization transition. Do not
            # let newly materialized/synced unit mismatches preempt that with
            # reconcile_status; otherwise polling can loop on materialization_ready.
            if mismatch and not round_state_generation_stale:
                any_mismatch = True
                mismatch_details.append(
                    {
                        "scope": "unit",
                        "unit": unit_name,
                        "local_implementation_state": impl_local.get("implementation_state"),
                        "remote_implementation_state": impl_remote.get("implementation_state"),
                        "local_remote_synced": impl_local.get("remote_synced"),
                        "remote_remote_synced": impl_remote.get("remote_synced"),
                        "local_remote_smoke_passed": impl_local.get("remote_smoke_passed"),
                        "remote_remote_smoke_passed": impl_remote.get("remote_smoke_passed"),
                        "local_run_state": run_local.get("run_state"),
                        "remote_run_state": run_remote.get("run_state"),
                    }
                )
            action = next_action_for_unit(unit_name, impl_local, run_local, remote, config)
            unit_meta = load_json(unit_root / "unit_meta.json", {})
            units.append(
                {
                    "unit": unit_name,
                    "proposal_file": unit_meta.get("proposal_file"),
                    "control_replicate": bool(unit_meta.get("control_replicate", False)),
                    "implementation_status_local": impl_local,
                    "run_status_local": run_local,
                    "implementation_status_remote": impl_remote,
                    "run_status_remote": run_remote,
                    **unit_q_fields(unit_root),
                    "remote": remote,
                    "mismatch": mismatch,
                    "next_action": action,
                }
            )

    blocking_writer = None
    if active_writer:
        for item in units:
            if item["unit"] == active_writer and item["next_action"] in {"implementation", "repair", "wait_for_repair_writer"}:
                blocking_writer = active_writer
                break

    round_mismatch_details = authoritative_round_state_mismatches(round_state_local, round_state_remote)
    round_mismatch = bool(round_mismatch_details) and not round_state_generation_stale
    if round_mismatch:
        any_mismatch = True
        mismatch_details.extend(round_mismatch_details)

    next_safe_step = {"kind": "idle"}
    if blocking_writer:
        next_safe_step = {"kind": "wait_for_writer", "unit": blocking_writer}
    elif round_state_generation_stale:
        next_safe_step = advance_generation_state_step(current_generation, round_state_local)
    elif any_mismatch:
        unit_target = None
        for row in mismatch_details:
            if row.get("scope") == "unit":
                unit_target = row.get("unit")
                break
        next_safe_step = {"kind": "reconcile_status", "unit": unit_target, "round_state": round_mismatch}
    else:
        selection_file_raw = round_state_local.get("active_selection_file")
        selection_file = Path(selection_file_raw) if selection_file_raw else None
        selection_generation = selection_target_generation(selection_file, current_generation)
        missing_units = missing_selected_units(selection_generation, selection_file)
        if missing_units:
            next_safe_step = {
                "kind": "materialize_generation",
                "current_generation": current_generation,
                "target_generation": selection_generation,
                "source_unit": round_state_local.get("continuation_source_unit"),
                "proposal_directory": round_state_local.get("active_proposal_directory"),
                "selection_file": str(selection_file) if selection_file else None,
                "materialized_units_root": str(GENERATIONS / selection_generation) if selection_generation else None,
                "missing_units": missing_units,
            }
        elif target := first_unit_with_action(units, {"collect"}):
            next_safe_step = {"kind": "collect", "unit": target["unit"]}
        elif target := first_unit_with_action(units, {"timeout_stop"}):
            next_safe_step = {"kind": "timeout_stop", "unit": target["unit"]}
        elif target := first_unit_with_action(units, {"repair"}):
            next_safe_step = {"kind": "repair", "unit": target["unit"]}
        elif target := first_unit_with_action(units, {"abandon"}):
            next_safe_step = {"kind": "abandon", "unit": target["unit"]}
        elif target := first_unit_with_action(units, {"sync", "smoke"}):
            next_safe_step = {"kind": target["next_action"], "unit": target["unit"]}
        elif target := first_unit_with_action(units, {"implementation"}):
            next_safe_step = {"kind": "implementation", "unit": target["unit"]}
        elif launch_batch_is_legal(units, current_generation, round_state_local):
            launchable = [item["unit"] for item in units if item["next_action"] == "launch_candidate"]
            if launchable:
                next_safe_step = {"kind": "launch_batch", "generation": current_generation, "units": launchable}
        elif units and all(is_terminal(item) for item in units):
            next_safe_step = round_complete_followup_step(current_generation)

    payload = {
        "generated_at_utc": utc_now(),
        "source_of_truth": "remote",
        "staging_runtime_root": str(STAGING_RUNTIME_ROOT),
        "remote_runtime_root": remote_runtime_root(config),
        "current_generation": current_generation,
        "current_generation_local": round_state_local.get("current_generation"),
        "current_generation_remote": round_state_remote.get("current_generation"),
        "workflow_state_local": round_state_local.get("workflow_state"),
        "workflow_state_remote": round_state_remote.get("workflow_state"),
        "round_state_stale": current_generation != round_state_local.get("current_generation"),
        "round_state_local": round_state_local,
        "round_state_remote": round_state_remote,
        "status_mismatch": any_mismatch,
        "mismatches": mismatch_details,
        "active_writer_unit": blocking_writer,
        "detected_openclaw_writer_unit": active_writer,
        "next_safe_step": next_safe_step,
        "units": units,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
