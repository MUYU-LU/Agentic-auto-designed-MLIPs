from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from runtime_common import (
    BASE_UNIT,
    GENERATIONS,
    ROUND_STATE,
    STAGING_RUNTIME_ROOT,
    default_implementation_status,
    default_run_status,
    infer_control_from_proposal_name,
    load_json,
    resolve_source_unit,
    save_json,
    unit_label,
    write_implementation_status,
    write_run_status,
)

BENCHMARK = STAGING_RUNTIME_ROOT / "benchmark"

MODEL_CODE_DEFAULTS = {
    "hidden_dim": 96,
    "num_rbf": 32,
    "cutoff": 5.0,
}

TRAIN_CODE_DEFAULTS = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "energy_weight": 1.0,
    "force_weight": 20.0,
}

CONFIG_TO_CODE_KEYS = tuple(MODEL_CODE_DEFAULTS) + tuple(TRAIN_CODE_DEFAULTS)
CODE_DEFAULT_MARKER_START = "# BEGIN MLIP code-level defaults"
CODE_DEFAULT_MARKER_END = "# END MLIP code-level defaults"


def rewrite_unit_config(target: Path) -> None:
    cfg_path = target / "config.json"
    if not cfg_path.exists():
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["benchmark_root"] = os.path.relpath(BENCHMARK, start=target)
    base_cfg = load_json(BASE_UNIT / "config.json", {})
    if isinstance(base_cfg.get("benchmark_datasets"), list):
        cfg["benchmark_datasets"] = list(base_cfg["benchmark_datasets"])
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def code_literal(value: object) -> str:
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, int):
        return repr(int(value))
    return repr(value)


def read_config_to_code_defaults(cfg: dict) -> dict:
    defaults = {}
    for key, fallback in {**MODEL_CODE_DEFAULTS, **TRAIN_CODE_DEFAULTS}.items():
        defaults[key] = cfg.get(key, fallback)
    return defaults


def upsert_code_default_block(text: str, defaults: dict) -> str:
    block_lines = [
        CODE_DEFAULT_MARKER_START,
        f"MODEL_HIDDEN_DIM = {code_literal(defaults['hidden_dim'])}",
        f"MODEL_NUM_RBF = {code_literal(defaults['num_rbf'])}",
        f"MODEL_CUTOFF = {code_literal(defaults['cutoff'])}",
        f"TRAIN_LEARNING_RATE = {code_literal(defaults['learning_rate'])}",
        f"TRAIN_WEIGHT_DECAY = {code_literal(defaults['weight_decay'])}",
        f"TRAIN_ENERGY_WEIGHT = {code_literal(defaults['energy_weight'])}",
        f"TRAIN_FORCE_WEIGHT = {code_literal(defaults['force_weight'])}",
        CODE_DEFAULT_MARKER_END,
    ]
    block = "\n".join(block_lines)
    pattern = re.compile(
        rf"{re.escape(CODE_DEFAULT_MARKER_START)}.*?{re.escape(CODE_DEFAULT_MARKER_END)}",
        flags=re.S,
    )
    if pattern.search(text):
        return pattern.sub(block, text)

    config_line = 'CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))'
    if config_line in text:
        return text.replace(config_line, f"{config_line}\n\n{block}", 1)
    return f"{block}\n\n{text}"


def migrate_train_defaults_to_code(train_path: Path, defaults: dict) -> bool:
    if not train_path.exists():
        return False
    text = train_path.read_text(encoding="utf-8")
    original = text
    text = upsert_code_default_block(text, defaults)

    replacements = {
        'hidden_dim=int(CONFIG.get("hidden_dim", 96)),': "hidden_dim=MODEL_HIDDEN_DIM,",
        'num_rbf=int(CONFIG.get("num_rbf", 32)),': "num_rbf=MODEL_NUM_RBF,",
        'cutoff=float(CONFIG.get("cutoff", 5.0)),': "cutoff=MODEL_CUTOFF,",
        'learning_rate = float(lr if lr is not None else CONFIG.get("learning_rate", 1e-3))': (
            "learning_rate = float(lr if lr is not None else TRAIN_LEARNING_RATE)"
        ),
        'weight_decay=float(CONFIG.get("weight_decay", 1e-5)),': "weight_decay=TRAIN_WEIGHT_DECAY,",
        'energy_weight = float(CONFIG.get("energy_weight", 2.4))': "energy_weight = TRAIN_ENERGY_WEIGHT",
        'force_weight = float(CONFIG.get("force_weight", 10.0))': "force_weight = TRAIN_FORCE_WEIGHT",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    if text != original:
        train_path.write_text(text, encoding="utf-8")
        return True
    return False


def migrate_model_defaults_to_code(model_path: Path, defaults: dict) -> bool:
    if not model_path.exists():
        return False
    text = model_path.read_text(encoding="utf-8")
    original = text
    text = re.sub(
        r"hidden_dim: int = \d+",
        f"hidden_dim: int = {int(defaults['hidden_dim'])}",
        text,
        count=1,
    )
    text = re.sub(
        r"num_rbf: int = \d+",
        f"num_rbf: int = {int(defaults['num_rbf'])}",
        text,
        count=1,
    )
    text = re.sub(
        r"cutoff: float = [0-9.]+",
        f"cutoff: float = {float(defaults['cutoff'])}",
        text,
        count=1,
    )
    if text != original:
        model_path.write_text(text, encoding="utf-8")
        return True
    return False


def migrate_safe_force_gradient_fallback(model_path: Path) -> bool:
    """Make force-from-energy robust for no-neighbor structures.

    Some periodic benchmark structures can have no pair inside the current
    cutoff. If atomref is frozen, the energy can be independent of positions;
    force evaluation must then return zero forces instead of crashing.
    """
    if not model_path.exists():
        return False
    text = model_path.read_text(encoding="utf-8")
    original = text
    old = "forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]"
    new = (
        "if torch.is_tensor(energy) and energy.requires_grad:\n"
        "            grad = torch.autograd.grad(energy, positions, create_graph=True, allow_unused=True)[0]\n"
        "        else:\n"
        "            grad = None\n"
        "        forces = torch.zeros_like(positions) if grad is None else -grad"
    )
    text = text.replace(old, new)
    if text != original:
        model_path.write_text(text, encoding="utf-8")
        return True
    return False


def migrate_config_defaults_to_code(target: Path) -> dict | None:
    cfg_path = target / "config.json"
    if not cfg_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    defaults = read_config_to_code_defaults(cfg)
    removed = {}
    for key in CONFIG_TO_CODE_KEYS:
        if key in cfg:
            removed[key] = cfg.pop(key)

    train_changed = migrate_train_defaults_to_code(target / "model" / "train.py", defaults)
    model_changed = migrate_model_defaults_to_code(target / "model" / "model.py", defaults)
    force_fallback_changed = migrate_safe_force_gradient_fallback(target / "model" / "model.py")

    if removed:
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "schema": "config_to_code_migration.v1",
        "description": "Mechanical materialization-time migration. Benchmark/runtime config stays fixed; MLIP architecture/training knobs become code-level defaults in model.py/train.py.",
        "migrated_keys": defaults,
        "removed_from_config": sorted(removed),
        "model_defaults_written": {key: defaults[key] for key in MODEL_CODE_DEFAULTS},
        "train_defaults_written": {key: defaults[key] for key in TRAIN_CODE_DEFAULTS},
        "files_changed": {
            "model/model.py": model_changed or force_fallback_changed,
            "model/train.py": train_changed,
            "config.json": bool(removed),
        },
        "force_gradient_fallback_written": force_fallback_changed,
        "note": "Proposal-specific changes should edit model/model.py and/or model/train.py, not config.json.",
    }


def remove_pycache(root: Path) -> None:
    for path in root.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)


def remove_copied_runtime_artifacts(target: Path) -> None:
    """Drop source-unit execution artifacts that must not survive materialization."""
    shutil.rmtree(target / "outputs", ignore_errors=True)
    shutil.rmtree(target / "research_context", ignore_errors=True)
    for name in (
        "launch.log",
        "seed_launch.log",
        "remote_smoke.log",
        "smoke.log",
        "smoke_output.log",
        "run.log",
    ):
        path = target / name
        if path.exists():
            path.unlink()
    for pattern in ("*.pid", "*.tmp", "*.bak"):
        for path in target.glob(pattern):
            if path.is_file():
                path.unlink()


def refresh_fixed_benchmark_harness(target: Path) -> list[str]:
    """Keep benchmark entry/eval/data reading fixed when branching old units."""
    refreshed = []
    for relative in ("main.py", "model/dataloader.py", "model/eval.py"):
        src = BASE_UNIT / relative
        dst = target / relative
        if src.exists() and dst.exists():
            shutil.copy2(src, dst)
            refreshed.append(relative)
    return refreshed


def write_unit_meta(
    target: Path,
    source: Path,
    proposal_file: str | None,
    control_replicate: bool,
    config_to_code_migration: dict | None = None,
    refreshed_harness_files: list[str] | None = None,
    force_gradient_fallback_written: bool = False,
) -> None:
    rel_target = target.relative_to(GENERATIONS)
    generation_round = rel_target.parts[0]
    proposal_unit = rel_target.parts[1]
    payload = {
        "source_unit": unit_label(source, STAGING_RUNTIME_ROOT),
        "generation_round": generation_round,
        "proposal_unit": proposal_unit,
        "proposal_file": proposal_file,
        "control_replicate": bool(control_replicate),
        "config_to_code_migration": {
            "schema": config_to_code_migration.get("schema"),
            "migrated_keys": config_to_code_migration.get("migrated_keys"),
            "removed_from_config": config_to_code_migration.get("removed_from_config"),
        }
        if config_to_code_migration
        else None,
        "refreshed_fixed_benchmark_harness": refreshed_harness_files or [],
        "force_gradient_fallback_written": bool(force_gradient_fallback_written),
    }
    save_json(target / "unit_meta.json", payload)


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


def copy_file_if_exists(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def evidence_package_dir_from_brief(brief_path: Path) -> Path | None:
    if not brief_path.exists():
        return None
    text = brief_path.read_text(encoding="utf-8", errors="replace")
    candidates: list[Path] = []
    for match in re.findall(r"(/[^\s`\"']*/knowledge/evidence_runs/[^\s`\"']+)", text):
        path = Path(match.rstrip(".,)"))
        if path.name.endswith(".json") or path.name.endswith(".md"):
            path = path.parent
        candidates.append(path)
    for path in candidates:
        if path.exists() and path.is_dir() and (path / "evidence_run.json").exists():
            return path
    return None


def selected_row_for_unit(selection_file: Path, proposal_id: str) -> dict:
    selection = load_json(selection_file, {}) if selection_file.exists() else {}
    for row in selection.get("selected_proposals", []):
        if isinstance(row, str):
            if Path(row).stem == proposal_id:
                return {"id": proposal_id, "path": row}
        elif isinstance(row, dict) and row.get("id") == proposal_id:
            return row
    return {}


def infer_control_replicate(target: Path, proposal_file: str | None, explicit_control: bool) -> bool:
    if explicit_control or infer_control_from_proposal_name(proposal_file):
        return True

    round_state = load_json(ROUND_STATE, {})
    selection_raw = round_state.get("active_selection_file")
    selection_file = Path(selection_raw).expanduser() if selection_raw else None
    if selection_file and selection_file.exists():
        row = selected_row_for_unit(selection_file, target.name)
        if row:
            if "control_replicate" in row:
                return bool(row.get("control_replicate"))
            role = str(row.get("role") or "").lower().replace("_", "-")
            if role:
                return role == "control"

    proposal_path = Path(proposal_file).expanduser() if proposal_file else None
    if proposal_path and proposal_path.exists():
        text = proposal_path.read_text(encoding="utf-8", errors="replace").lower()
        if re.search(r"^\s*-\s*jump_type\s*:\s*control\s*$", text, flags=re.M):
            return True
        if re.search(r"^\s*-\s*relation_to_source\s*:\s*control\s*$", text, flags=re.M):
            return True

    return False


def write_research_context(target: Path, proposal_file: str | None, config_to_code_migration: dict | None = None) -> None:
    context_dir = target / "research_context"
    context_dir.mkdir(parents=True, exist_ok=True)
    round_state = load_json(ROUND_STATE, {})
    proposal_path = Path(proposal_file).expanduser() if proposal_file else None

    copied = {
        "proposal": copy_file_if_exists(proposal_path, context_dir / "proposal.md"),
        "proposal_context": None,
        "evidence_brief": None,
        "selection_row": None,
        "evidence_package_files": {},
        "config_to_code_migration": None,
    }

    if config_to_code_migration:
        save_json(context_dir / "config_to_code_migration.json", config_to_code_migration)
        copied["config_to_code_migration"] = str(context_dir / "config_to_code_migration.json")

    if proposal_path and proposal_path.exists():
        copied["proposal_context"] = copy_file_if_exists(proposal_path.parent / "context.md", context_dir / "context.md")

    brief_raw = round_state.get("active_evidence_brief")
    brief_path = Path(brief_raw).expanduser() if brief_raw else None
    copied["evidence_brief"] = copy_file_if_exists(brief_path, context_dir / "evidence_brief.md")

    if brief_path:
        package_dir = evidence_package_dir_from_brief(brief_path)
        if package_dir:
            for name in EVIDENCE_PACKAGE_FILES:
                copied["evidence_package_files"][name] = copy_file_if_exists(package_dir / name, context_dir / "evidence_package" / name)

    selection_raw = round_state.get("active_selection_file")
    selection_file = Path(selection_raw).expanduser() if selection_raw else None
    if selection_file:
        row = selected_row_for_unit(selection_file, target.name)
        if row:
            save_json(context_dir / "selection_row.json", row)
            copied["selection_row"] = str(context_dir / "selection_row.json")

    manifest = {
        "version": "research_context.v1",
        "unit": str(target.relative_to(GENERATIONS)),
        "source_round_state": str(ROUND_STATE),
        "active_evidence_brief": brief_raw,
        "active_selection_file": selection_raw,
        "copied": copied,
        "implementation_handoff": {
            "read_first": [
                "proposal.md",
                "context.md",
                "evidence_package/mechanism_cards.json",
                "evidence_package/patch_blueprints.json",
                "evidence_package/proposal_constraints.json",
                "config_to_code_migration.json",
            ],
            "write_after_edit": "implementation_report.json",
        },
    }
    save_json(context_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="generation_### or generation_###/proposal_###")
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--proposal-file", default=None)
    parser.add_argument("--control-replicate", action="store_true")
    args = parser.parse_args()

    target = GENERATIONS / args.target
    if target.exists():
        raise SystemExit(f"target already exists: {target}")

    if "/" not in args.target:
        target.mkdir(parents=True, exist_ok=False)
        print(target)
        return

    source = resolve_source_unit(args.source_path, STAGING_RUNTIME_ROOT) if args.source_path else BASE_UNIT
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    remove_copied_runtime_artifacts(target)
    (target / "outputs").mkdir(parents=True, exist_ok=True)
    remove_pycache(target)
    refreshed_harness_files = refresh_fixed_benchmark_harness(target)
    rewrite_unit_config(target)
    control = infer_control_replicate(target, args.proposal_file, args.control_replicate)
    config_to_code_migration = None if control else migrate_config_defaults_to_code(target)
    force_gradient_fallback_written = migrate_safe_force_gradient_fallback(target / "model" / "model.py")
    if config_to_code_migration and config_to_code_migration.get("force_gradient_fallback_written"):
        force_gradient_fallback_written = True
    write_unit_meta(
        target,
        source,
        args.proposal_file,
        control,
        config_to_code_migration,
        refreshed_harness_files,
        force_gradient_fallback_written,
    )
    write_implementation_status(
        target,
        default_implementation_status(
            target,
            source_unit=unit_label(source, STAGING_RUNTIME_ROOT),
            proposal_file=args.proposal_file,
            control_replicate=control,
        ),
    )
    write_run_status(target, default_run_status())
    write_research_context(target, args.proposal_file, config_to_code_migration)
    print(target)


if __name__ == "__main__":
    main()
