from __future__ import annotations

import re
from typing import Any


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def canonical_jump_type(value: Any) -> str:
    slug = _slug(value)
    if slug in {"backward_simplify", "backward_simplification"}:
        return "backward-simplify"
    if slug in {"wild_card"}:
        return "wildcard"
    return slug.replace("_", "-")


def _joined_text(*parts: Any) -> str:
    out: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            out.extend(_joined_text(v) for v in part.values() if v)
        elif isinstance(part, list):
            out.extend(_joined_text(v) for v in part)
        elif part is not None:
            out.append(str(part))
    return _slug(" ".join(out))


def mechanism_tags(
    *,
    family: Any = None,
    implementation_status: dict[str, Any] | None = None,
    model_text: str = "",
    train_text: str = "",
    proposal_file: Any = None,
) -> list[str]:
    meta_text = _joined_text(family, implementation_status or {}, proposal_file)
    code_text = _joined_text(model_text, train_text)
    text = _joined_text(meta_text, code_text)
    tags: set[str] = set()

    if (implementation_status or {}).get("control_replicate") or "control_replicate" in meta_text or "source_control" in meta_text:
        tags.add("control_replicate")
    if any(token in meta_text for token in ("atomref", "offset", "baseline", "composition", "lstsq", "energy_adapter", "energy_scale", "calibration", "calibrator")) or any(
        token in code_text for token in ("lstsq", "composition", "offset_head", "energy_adapter", "energy_scale", "calibrator")
    ):
        tags.add("energy_baseline_calibration")
    if any(token in text for token in ("optimizer", "scheduler", "schedule", "loss_rebalance", "loss_weight", "learning_rate", "warmup", "training_schedule")):
        tags.add("training_objective")
    if any(token in text for token in ("readout", "energy_mlp", "energy_head", "per_atom_energy", "vector_norm_layer")):
        tags.add("local_readout_calibration")
    if any(token in text for token in ("manybody", "body_order", "higher_order", "triplet", "angle", "late_invariant", "three_body")):
        tags.add("local_manybody_invariant")
    if any(token in text for token in ("scalar_mix", "scalar_reentry", "balanced_scalar", "scalar_residual")):
        tags.add("scalar_mixing")
    if any(token in text for token in ("vector_state", "equivariant", "lowrankequivariant", "directional", "unit_vector", "vector_message", "radial_proj", "message_passing", "message_update", "edge_filter")):
        tags.add("local_equivariant_message_passing")
    elif any(token in text for token in ("message_mlp", "message_update", "neighbor_message", "index_add", "edge_index", "rbf")):
        tags.add("local_message_passing")
    if "geometry" in text and "energy" in text:
        tags.add("geometry_energy_safeguard")

    return sorted(tags)


def canonical_family_from_slug(value: Any) -> str:
    slug = _slug(value)
    if slug == "unknown":
        return "unknown"
    if "control" in slug:
        return "control_replicate"
    if any(token in slug for token in ("atomref", "offset", "baseline", "composition", "lstsq", "energy_adapter", "energy_balance", "energy_scale", "calibration", "calibrator")):
        return "energy_baseline_calibration"
    if any(token in slug for token in ("optimizer", "schedule", "loss", "training")):
        return "training_objective"
    if "readout" in slug:
        return "local_readout_calibration"
    if any(token in slug for token in ("manybody", "body_order", "higher_order", "late_invariant", "triplet", "angle")):
        return "local_manybody_invariant"
    if any(token in slug for token in ("scalar_mix", "scalar_reentry", "balanced_scalar", "scalar_residual")):
        return "scalar_mixing"
    if any(token in slug for token in ("local_equivariant", "equivariant_stream", "equivariant_local", "nequip_style", "directional_alignment", "message_passing", "vector_simplified", "vector_residual")):
        return "local_equivariant_message_passing"
    if "geometry" in slug and "energy" in slug:
        return "geometry_energy_safeguard"
    return slug


def canonical_family(
    value: Any,
    *,
    implementation_status: dict[str, Any] | None = None,
    model_text: str = "",
    train_text: str = "",
    proposal_file: Any = None,
) -> str:
    """Return a stable analysis family without rewriting proposal metadata."""
    raw = _slug(value)
    if raw != "unknown":
        return canonical_family_from_slug(raw)

    tags = mechanism_tags(
        family=value,
        implementation_status=implementation_status,
        model_text=model_text,
        train_text=train_text,
        proposal_file=proposal_file,
    )
    if "control_replicate" in tags:
        return "control_replicate"
    if "local_manybody_invariant" in tags:
        return "local_manybody_invariant"
    if "local_equivariant_message_passing" in tags:
        return "local_equivariant_message_passing"
    if "local_message_passing" in tags:
        return "local_message_passing"
    if "local_readout_calibration" in tags:
        return "local_readout_calibration"
    if "scalar_mixing" in tags:
        return "scalar_mixing"
    if "training_objective" in tags:
        return "training_objective"
    if "energy_baseline_calibration" in tags:
        return "energy_baseline_calibration"
    if "geometry_energy_safeguard" in tags:
        return "geometry_energy_safeguard"
    return "unknown"


def family_record(
    family: Any,
    *,
    implementation_status: dict[str, Any] | None = None,
    model_text: str = "",
    train_text: str = "",
    proposal_file: Any = None,
) -> dict[str, Any]:
    raw = _slug(family)
    if raw == "unknown":
        tags = mechanism_tags(
            family=family,
            implementation_status=implementation_status,
            model_text=model_text,
            train_text=train_text,
            proposal_file=proposal_file,
        )
    else:
        tags = mechanism_tags(family=family, proposal_file=proposal_file)
    canonical = canonical_family(
        family,
        implementation_status=implementation_status,
        model_text=model_text,
        train_text=train_text,
        proposal_file=proposal_file,
    )
    source = "proposal_metadata" if raw != "unknown" else "mechanism_trace"
    if canonical == "unknown":
        source = "unknown"
    return {
        "canonical_family": canonical,
        "mechanism_tags": tags,
        "family_source": source,
    }
