from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

from runtime_common import STAGING_RUNTIME_ROOT, load_json, resolve_unit, save_json, update_implementation_status


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concise_file_diff(old_path: Path, new_path: Path, max_lines: int = 120) -> dict:
    if not old_path.exists() or not new_path.exists():
        return {"available": False, "reason": "missing source or target file"}
    old_text = old_path.read_text(encoding="utf-8", errors="replace")
    new_text = new_path.read_text(encoding="utf-8", errors="replace")
    diff = list(difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=str(old_path),
        tofile=str(new_path),
        lineterm="",
    ))
    changed = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    return {
        "available": True,
        "same_as_source": not changed,
        "added_lines": sum(1 for line in changed if line.startswith("+")),
        "removed_lines": sum(1 for line in changed if line.startswith("-")),
        "diff_truncated": len(diff) > max_lines,
        "unified_diff_excerpt": diff[:max_lines],
    }


def write_implementation_report(unit_root: Path, actor: str, notes: list[str], changed: list[dict]) -> Path:
    meta = load_json(unit_root / "unit_meta.json", {})
    source_unit = meta.get("source_unit") or "base_unit"
    source_root = resolve_unit(source_unit, STAGING_RUNTIME_ROOT)
    context_dir = unit_root / "research_context"
    manifest = load_json(context_dir / "manifest.json", {})

    file_diffs = {}
    for rel in ["model/model.py", "model/train.py"]:
        file_diffs[rel] = concise_file_diff(source_root / rel, unit_root / rel)

    report = {
        "version": "implementation_report.v1",
        "unit": str(unit_root.relative_to(STAGING_RUNTIME_ROOT / "generations")),
        "source_unit": source_unit,
        "actor": actor,
        "notes": notes,
        "changed_files": changed,
        "file_diffs_vs_source": file_diffs,
        "research_context_manifest": manifest,
        "handoff_files_present": {
            "proposal": (context_dir / "proposal.md").exists(),
            "context": (context_dir / "context.md").exists(),
            "evidence_brief": (context_dir / "evidence_brief.md").exists(),
            "mechanism_cards": (context_dir / "evidence_package" / "mechanism_cards.json").exists(),
            "patch_blueprints": (context_dir / "evidence_package" / "patch_blueprints.json").exists(),
            "proposal_constraints": (context_dir / "evidence_package" / "proposal_constraints.json").exists(),
            "selection_row": (context_dir / "selection_row.json").exists(),
        },
    }
    path = context_dir / "implementation_report.json"
    save_json(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--actor", default="implementation_subagent")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()

    unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    changed = []
    for rel in ["model/model.py", "model/train.py"]:
        path = unit_root / rel
        if path.exists():
            changed.append({"path": rel, "sha256": sha256_file(path)})

    report_path = write_implementation_report(unit_root, args.actor, args.note, changed)

    updates = {
        "implementation_state": "implemented",
        "changed_files": changed,
        "implementation_report": str(report_path),
        "remote_synced": False,
        "remote_smoke_passed": False,
        "last_failure_class": None,
        "last_actor": args.actor,
    }
    if args.note:
        status = {}
        try:
            from runtime_common import load_implementation_status
            status = load_implementation_status(unit_root)
        except Exception:
            status = {}
        updates["notes"] = list(status.get("notes", [])) + args.note

    update_implementation_status(
        unit_root,
        **updates,
    )
    print(unit_root / "implementation_status.json")


if __name__ == "__main__":
    main()
