from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from runtime_common import (
    GENERATIONS,
    STAGING_RUNTIME_ROOT,
    load_config,
    load_json,
    load_round_state,
    now_utc,
    round_state_path,
    save_json,
    sync_round_state_to_remote,
)

REQUIRED_PROPOSAL_FIELDS = ("family", "phase", "jump_type", "budget_class")
REQUIRED_PROPOSAL_HANDOFF_TERMS = (
    "mechanism_refs",
    "evidence_refs",
    "files_to_edit",
    "code_insertion_points",
    "minimal_edit_plan",
    "implementation_checklist",
)
PROPOSAL_ID_RE = re.compile(r"^proposal_\d+$")


def fail(message: str) -> None:
    raise SystemExit(message)


def as_path(value: str | None, *, name: str) -> Path:
    if not value:
        fail(f"{name} is required")
    return Path(value).expanduser().resolve()


def active_proposal_dir(state: dict[str, Any]) -> Path:
    raw = state.get("active_proposal_directory")
    if not raw:
        fail("round_state.active_proposal_directory is required")
    return Path(raw).expanduser().resolve()


def assert_source_and_target(state: dict[str, Any], source_unit: str | None, target_generation: str | None) -> None:
    if source_unit and state.get("continuation_source_unit") != source_unit:
        fail(
            "source-unit does not match round_state.continuation_source_unit: "
            f"{source_unit!r} != {state.get('continuation_source_unit')!r}"
        )
    expected_dir = active_proposal_dir(state)
    if target_generation and not expected_dir.name.startswith(f"{target_generation}_from_"):
        fail(
            "target-generation does not match active_proposal_directory: "
            f"{target_generation!r} not in {expected_dir.name!r}"
        )


def assert_under(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        fail(f"{name} must be under {root}: {path}")


def selected_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = selection.get("selected_proposals")
    if not isinstance(rows, list):
        fail("selection.selected_proposals must be a list")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row, str):
            proposal_id = Path(row).stem
            path = row
            role = None
        elif isinstance(row, dict):
            proposal_id = row.get("id")
            path = row.get("path")
            role = row.get("role") or row.get("jump_type")
        else:
            fail(f"selected_proposals[{index}] must be object or string")
        if not proposal_id:
            fail(f"selected_proposals[{index}] is missing id")
        out.append({"id": str(proposal_id), "path": path, "role": role})
    return out


def parse_proposal_metadata(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata: dict[str, str] = {}
    for line in text.splitlines()[:100]:
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            key, value = stripped[2:].split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def assert_proposals_valid(proposal_dir: Path) -> list[Path]:
    proposal_files = sorted(proposal_dir.glob("proposal_*.md"))
    if len(proposal_files) < 10:
        fail(f"proposal directory must contain at least 10 proposal_*.md files: found {len(proposal_files)}")
    missing: dict[str, list[str]] = {}
    missing_handoff: dict[str, list[str]] = {}
    for path in proposal_files[:10]:
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = parse_proposal_metadata(path)
        absent = [field for field in REQUIRED_PROPOSAL_FIELDS if not metadata.get(field)]
        if absent:
            missing[path.name] = absent
        handoff_absent = [term for term in REQUIRED_PROPOSAL_HANDOFF_TERMS if term not in text]
        if handoff_absent:
            missing_handoff[path.name] = handoff_absent
    if missing:
        fail(f"proposal files missing required metadata: {json.dumps(missing, ensure_ascii=False)}")
    if missing_handoff:
        fail(f"proposal files missing implementation handoff terms: {json.dumps(missing_handoff, ensure_ascii=False)}")
    return proposal_files


def proposal_path_for_row(proposal_dir: Path, row: dict[str, Any]) -> Path:
    raw_path = row.get("path")
    if raw_path:
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else proposal_dir / candidate
    else:
        path = proposal_dir / f"{row['id']}.md"
    return path.expanduser().resolve()


def validate_selection(selection_path: Path, proposal_dir: Path, source_unit: str | None, target_generation: str | None) -> dict[str, Any]:
    if not selection_path.exists():
        fail(f"selection file does not exist: {selection_path}")
    if selection_path.parent.resolve() != proposal_dir.resolve():
        fail("selection file must be inside active_proposal_directory")
    selection = load_json(selection_path, {})
    if source_unit and selection.get("source_unit") and selection.get("source_unit") != source_unit:
        fail("selection source_unit does not match current continuation source")
    if target_generation and selection.get("target_generation") and selection.get("target_generation") != target_generation:
        fail("selection target_generation does not match requested target generation")
    rows = selected_rows(selection)
    if len(rows) != 8:
        fail(f"selection must contain exactly 8 selected proposals, found {len(rows)}")
    roles = [str(row.get("role") or "").lower().replace("_", "-") for row in rows]
    if any(roles):
        if not any("control" in role for role in roles):
            fail("selection must contain a control proposal")
        if sum(1 for role in roles if "exploit" in role) < 2:
            fail("selection must contain at least 2 exploit proposals")
        if sum(1 for role in roles if "jump" in role) < 2:
            fail("selection must contain at least 2 jump proposals")
    for row in rows:
        if not PROPOSAL_ID_RE.fullmatch(row["id"]):
            fail(f"selected proposal id must look like proposal_###: {row['id']}")
        proposal_path = proposal_path_for_row(proposal_dir, row)
        if not proposal_path.exists():
            fail(f"selected proposal file is missing: {proposal_path}")
        assert_under(proposal_path, proposal_dir, name="selected proposal file")
    return selection


def event_evidence_complete(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    assert_source_and_target(state, args.source_unit, args.target_generation)
    if state.get("workflow_state") not in {None, "evidence-needed", "context-preparation"}:
        fail(f"evidence_complete is not valid from workflow_state={state.get('workflow_state')!r}")
    brief = as_path(args.brief, name="--brief")
    if not brief.exists():
        fail(f"brief does not exist: {brief}")
    assert_under(brief, STAGING_RUNTIME_ROOT / "knowledge" / "briefs", name="brief")
    state.update(
        {
            "workflow_state": "context-preparation",
            "active_evidence_brief": str(brief),
            "evidence_for_source_unit": args.source_unit,
            "proposal_context_file": None,
            "proposal_directory_ready": False,
            "proposal_source": None,
            "proposal_writer_session": None,
            "active_evidence_task": None,
            "source_of_truth": "remote",
        }
    )
    return state


def event_proposal_context_ready(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    assert_source_and_target(state, args.source_unit, args.target_generation)
    context = as_path(args.context, name="--context")
    expected = active_proposal_dir(state) / "context.md"
    if context != expected.resolve():
        fail(f"context must equal active_proposal_directory/context.md: {context} != {expected}")
    if not context.exists():
        fail(f"context does not exist: {context}")
    brief = state.get("active_evidence_brief")
    if not brief or not Path(brief).exists():
        fail("round_state.active_evidence_brief must exist before proposal_context_ready")
    if state.get("evidence_for_source_unit") != state.get("continuation_source_unit"):
        fail("round_state.evidence_for_source_unit does not match continuation_source_unit")
    text = context.read_text(encoding="utf-8", errors="replace")
    if str(brief) not in text:
        fail("context does not reference round_state.active_evidence_brief")
    state.update(
        {
            "workflow_state": "proposal-writing",
            "proposal_context_file": str(context),
            "proposal_directory_ready": False,
            "proposal_source": None,
            "proposal_writer_session": None,
            "source_of_truth": "remote",
        }
    )
    return state


def event_proposals_ready(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    assert_source_and_target(state, args.source_unit, args.target_generation)
    proposal_dir = as_path(args.proposal_dir, name="--proposal-dir")
    if proposal_dir != active_proposal_dir(state):
        fail("--proposal-dir must match round_state.active_proposal_directory")
    context = state.get("proposal_context_file")
    if not context or not Path(context).exists():
        fail("proposal_context_file must exist before proposals_ready")
    assert_proposals_valid(proposal_dir)
    proposal_source = args.proposal_source or args.proposal_writer_session
    if not proposal_source:
        fail("--proposal-source or --proposal-writer-session is required")
    state.update(
        {
            "workflow_state": "selection",
            "proposal_directory_ready": True,
            "proposal_source": args.proposal_source,
            "proposal_writer_session": args.proposal_writer_session,
            "source_of_truth": "remote",
        }
    )
    return state


def event_selection_ready(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    assert_source_and_target(state, args.source_unit, args.target_generation)
    proposal_dir = active_proposal_dir(state)
    if not state.get("proposal_directory_ready"):
        fail("proposal_directory_ready must be true before selection_ready")
    selection_file = as_path(args.selection_file, name="--selection-file")
    validate_selection(selection_file, proposal_dir, state.get("continuation_source_unit"), args.target_generation)
    state.update(
        {
            "workflow_state": "materialization",
            "active_selection_file": str(selection_file),
            "source_of_truth": "remote",
        }
    )
    return state


def event_materialization_ready(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    generation = args.generation or args.target_generation
    if not generation:
        fail("--generation or --target-generation is required")
    selection_raw = state.get("active_selection_file")
    if not selection_raw:
        fail("active_selection_file is required before materialization_ready")
    proposal_dir = active_proposal_dir(state)
    selection_file = Path(selection_raw).expanduser().resolve()
    selection = validate_selection(selection_file, proposal_dir, state.get("continuation_source_unit"), generation)
    generation_root = (GENERATIONS / generation).resolve()
    rows = selected_rows(selection)
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        unit_root = generation_root / row["id"]
        if not unit_root.exists():
            missing.append(f"{generation}/{row['id']}")
            continue
        if not (unit_root / "unit_meta.json").exists() or not (unit_root / "implementation_status.json").exists():
            invalid.append(f"{generation}/{row['id']}")
    if missing:
        fail(f"selected units are not materialized: {missing}")
    if invalid:
        fail(f"materialized units are missing required status/meta files: {invalid}")
    state.update(
        {
            "workflow_state": "implementation",
            "current_generation": generation,
            "materialized_units_root": str(generation_root),
            "next_recommended_step": "implement_next_unit",
            "source_of_truth": "remote",
        }
    )
    return state


def event_generation_active(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    generation = args.generation or args.target_generation
    if not generation:
        fail("--generation or --target-generation is required")
    materialized = state.get("materialized_units_root")
    expected = (GENERATIONS / generation).resolve()
    if not materialized or Path(materialized).expanduser().resolve() != expected:
        fail("materialized_units_root must match requested generation before generation_active")
    state.update(
        {
            "workflow_state": "implementation",
            "current_generation": generation,
            "next_recommended_step": "implement_next_unit",
            "source_of_truth": "remote",
        }
    )
    return state


EVENTS = {
    "evidence_complete": event_evidence_complete,
    "proposal_context_ready": event_proposal_context_ready,
    "proposals_ready": event_proposals_ready,
    "selection_ready": event_selection_ready,
    "materialization_ready": event_materialization_ready,
    "generation_active": event_generation_active,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validated event transitions for global round_state fields.")
    parser.add_argument("--event", required=True, choices=sorted(EVENTS))
    parser.add_argument("--source-unit", default=None)
    parser.add_argument("--target-generation", default=None)
    parser.add_argument("--generation", default=None)
    parser.add_argument("--brief", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--proposal-dir", default=None)
    parser.add_argument("--proposal-source", default=None)
    parser.add_argument("--proposal-writer-session", default=None)
    parser.add_argument("--selection-file", default=None)
    parser.add_argument("--sync-remote", action="store_true")
    args = parser.parse_args()

    state = load_round_state(STAGING_RUNTIME_ROOT)
    if not state:
        fail(f"round_state.json not found or empty: {round_state_path()}")
    updated = EVENTS[args.event](state, args)
    updated["last_transition_utc"] = now_utc()
    save_json(round_state_path(), updated)
    if args.sync_remote:
        sync_round_state_to_remote(config=load_config())
    print(json.dumps({"event": args.event, "round_state": str(round_state_path()), "synced_remote": bool(args.sync_remote)}, indent=2))


if __name__ == "__main__":
    main()
