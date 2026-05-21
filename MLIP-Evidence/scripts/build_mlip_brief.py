#!/usr/bin/env python3
"""Build an MLIP-Evidence dossier from structured internal retrieval results.

Retrieval is now an internal helper of the public MLIP-Evidence skill.
This script interprets structured results and renders the required dossier sections.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

WS_RE = re.compile(r"\s+")
PATTERN_KEYWORDS: dict[str, list[str]] = {
    "equivariance": ["equivariant", "e3nn", "e(3)", "se(3)", "等变", "painn", "equiformer"],
    "symmetry / invariance": ["symmetry", "invariant", "permutation", "rotation", "translation", "不变", "置换"],
    "atom reference energy": ["atomref", "reference energy", "atomic energy"],
    "angular / three-body geometry": ["angular", "angle", "triplet", "three-body", "many-body", "many body", "多体", "三体", "ace", "tace", "cace", "原子簇"],
    "energy/force joint loss weighting": ["force loss", "joint loss", "force_weight", "energy_weight"],
    "force normalization": ["force normalization", "force matching", "gradient-domain", "unit scaling"],
    "energy conservation / force consistency": ["energy conserving", "energy conservation", "conservative", "sgdml", "能量守恒", "力场"],
    "stress supervision": ["stress", "virial"],
    "locality / cutoff": ["cutoff", "locality", "neighbor", "neighbour", "radius", "局域", "soap", "描述符"],
    "message passing depth": ["message passing", "interaction block", "num_interactions", "num_layers", "消息传递", "schnet", "连续滤波"],
    "long-range effects": ["long-range", "electrostatics", "charge", "dipole", "长程", "静电", "电荷"],
}
NOTE_PROMOTION_MIN_CONFIDENCE = {"A", "B"}
NOTE_PROMOTION_MIN_SOURCE_TYPES = 2
NOTE_PROMOTION_MIN_SUPPORT = 3


@dataclass
class Config:
    workspace_root: Path
    briefs_dir: Path
    notes_dir: Path
    index_path: Path
    slug: str
    auto_promote_notes: bool


@dataclass
class SearchRun:
    question: str
    local_context: str
    constraints: list[str]
    source_priority: list[str]
    generated_at_utc: str
    query_plan: list[dict[str, Any]]
    results: list[dict[str, Any]]


@dataclass
class SearchResult:
    id: str
    source_type: str
    title: str
    url: str
    snippet: str
    content: str
    score: float
    tier: str
    role: str
    signals: dict[str, float]
    metadata: dict[str, Any]
    tags: list[str]


@dataclass
class ResultBuckets:
    strong: list[SearchResult]
    weak: list[SearchResult]
    background: list[SearchResult]


def compact_ws(text: str) -> str:
    return WS_RE.sub(" ", text or "").strip()


def slugify(text: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:limit] or "brief"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    lower = (text or "").lower()
    return any(keyword.lower() in lower for keyword in keywords)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_search_run(path: Path) -> SearchRun:
    payload = read_json(path)
    if payload.get("version") != "search-run.v1":
        raise SystemExit("Unsupported search-run version. Expected search-run.v1")
    if not payload.get("question"):
        raise SystemExit("search-run is missing question")
    if not isinstance(payload.get("results"), list):
        raise SystemExit("search-run results must be a list")
    return SearchRun(
        question=compact_ws(payload.get("question", "")),
        local_context=compact_ws(payload.get("local_context", "")),
        constraints=[compact_ws(item) for item in payload.get("constraints", [])],
        source_priority=[str(item) for item in payload.get("source_priority", [])],
        generated_at_utc=str(payload.get("generated_at_utc", "")),
        query_plan=list(payload.get("query_plan", [])),
        results=list(payload.get("results", [])),
    )


def parse_results(results: Sequence[dict[str, Any]]) -> list[SearchResult]:
    parsed: list[SearchResult] = []
    for item in results:
        parsed.append(
            SearchResult(
                id=str(item.get("id", "")),
                source_type=str(item.get("source_type", "unknown")),
                title=compact_ws(str(item.get("title", ""))),
                url=str(item.get("url", "")),
                snippet=compact_ws(str(item.get("snippet", ""))),
                content=compact_ws(str(item.get("content", ""))),
                score=float(item.get("score", 0.0) or 0.0),
                tier=str(item.get("tier", "background")),
                role=str(item.get("role", "background")),
                signals=dict(item.get("signals", {})),
                metadata=dict(item.get("metadata", {})),
                tags=[str(tag) for tag in item.get("tags", [])],
            )
        )
    return parsed


def bucket_by_tier(results: Sequence[SearchResult]) -> ResultBuckets:
    strong = [item for item in results if item.tier == "strong"]
    weak = [item for item in results if item.tier == "weak"]
    background = [item for item in results if item.tier not in {"strong", "weak"}]
    return ResultBuckets(strong=strong, weak=weak, background=background)


def classify_results(results: Sequence[SearchResult]) -> tuple[list[SearchResult], list[SearchResult], list[SearchResult], list[SearchResult]]:
    local_literature: list[SearchResult] = []
    repos: list[SearchResult] = []
    papers: list[SearchResult] = []
    other: list[SearchResult] = []
    for item in results:
        if item.source_type == "local_pdf" or "local-library" in item.tags:
            local_literature.append(item)
            papers.append(item)
        elif item.source_type == "github" or "repo" in item.tags:
            repos.append(item)
        elif item.source_type == "arxiv" or "paper" in item.tags:
            papers.append(item)
        else:
            other.append(item)
    return local_literature, repos, papers, other




def _jsonish_scalar(value: str) -> Any:
    value = value.strip().strip(',')
    if value.lower() in {"null", "none"}:
        return None
    if value.lower() == "nan":
        return None
    try:
        return float(value)
    except ValueError:
        return value.strip('"')


def _field_occurrences(text: str, names: Sequence[str]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {name: [] for name in names}
    for name in names:
        pattern = re.compile(rf'"?{re.escape(name)}"?\s*[:=]\s*("[^"\n]+"|[-+0-9.eE]+|NaN|null)', re.IGNORECASE)
        for match in pattern.finditer(text):
            out[name].append(_jsonish_scalar(match.group(1)))
    return {key: value for key, value in out.items() if value}


def extract_benchmark_dossier(local_context: str) -> dict[str, Any]:
    """
    Best-effort extraction of benchmark-centric information from proposal context.
    The context may include embedded JSON blocks for runtime summary, dataset dossiers,
    generation summary, and frontier tail. Preserve completeness warnings even when
    precise parsing is not reliable.
    """
    text = local_context or ""
    lower = text.lower()
    metric_fields = [
        "mixed_force_mae",
        "mixed_energy_mae",
        "gap_penalty",
        "Q_dataset",
        "Q_rmd17",
        "Q_iso17",
        "Q_total",
        "G_delta",
        "mild_ood_energy_mae",
        "mild_ood_force_mae",
        "hard_ood_energy_mae",
        "hard_ood_force_mae",
        "within_energy_mae",
        "within_force_mae",
        "other_energy_mae",
        "other_force_mae",
    ]
    runtime_fields = [
        "run_state",
        "failure_class",
        "implementation_state",
        "remote_smoke_passed",
        "remote_synced",
        "repair_attempts",
        "control_replicate",
    ]
    dossier: dict[str, Any] = {
        "context_length": len(text),
        "datasets": {
            "rmd17": {"mentioned": "rmd17" in lower},
            "iso17": {"mentioned": "iso17" in lower},
        },
        "q_fields": {},
        "metric_fields": _field_occurrences(text, metric_fields),
        "runtime": _field_occurrences(text, runtime_fields),
        "training_dynamics": {
            "train_history_mentioned": "train_history" in lower or "history_summary" in lower,
            "best_val_force_mentioned": "best_val_force_mae" in lower,
            "best_val_energy_mentioned": "best_val_energy_mae" in lower,
        },
        "control_comparison": {
            "control_mentioned": "control_replicate" in lower or "control replicate" in lower,
            "best_control_q_mentioned": "best_control_q_total" in lower,
        },
        "warnings": [],
    }
    dossier["q_fields"] = {key: dossier["metric_fields"].get(key, []) for key in ["Q_dataset", "Q_rmd17", "Q_iso17", "Q_total", "G_delta"] if dossier["metric_fields"].get(key)}

    required = ["mixed_force_mae", "mixed_energy_mae", "gap_penalty"]
    missing = [token for token in required if token not in text]
    if missing:
        dossier["warnings"].append("local_context appears benchmark-incomplete or force-biased; missing: " + ", ".join(missing))
    richer_required = ["Q_dataset", "Q_total", "run_state"]
    richer_missing = [token for token in richer_required if token not in text]
    if richer_missing:
        dossier["warnings"].append("local_context is missing richer Q/runtime fields: " + ", ".join(richer_missing))
    if "mixed_force_mae" in text and "mixed_energy_mae" not in text:
        dossier["warnings"].append("force metrics are present but energy metrics are missing")
    return dossier

def bucket_by_role(results: Sequence[SearchResult]) -> dict[str, list[SearchResult]]:
    buckets: dict[str, list[SearchResult]] = {
        "framework": [],
        "representation": [],
        "design": [],
        "decision": [],
        "background": [],
    }
    for item in results:
        buckets.setdefault(item.role, []).append(item)
    return buckets


def extract_emergent_themes(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    themes: list[dict[str, Any]] = []
    combined = results
    for pattern, keywords in PATTERN_KEYWORDS.items():
        matched = []
        source_types: set[str] = set()
        for item in combined:
            text = " ".join([item.title, item.snippet, item.content, " ".join(item.tags)])
            if contains_any(text, keywords):
                matched.append(item)
                source_types.add(item.source_type)
        if matched:
            themes.append(
                {
                    "pattern": pattern,
                    "support_count": len(matched),
                    "source_types": sorted(source_types),
                    "examples": [
                        {"title": item.title, "url": item.url, "source_type": item.source_type}
                        for item in matched[:3]
                    ],
                }
            )
    themes.sort(key=lambda item: (len(item["source_types"]), item["support_count"]), reverse=True)
    return themes


def assess_implementation_fit(repos: Sequence[SearchResult], constraints: Sequence[str]) -> tuple[str, list[str]]:
    if not repos:
        return "Low", ["No strong repository evidence was available in the search run."]
    pytorch_hits = 0
    python_hits = 0
    runnable_hits = 0
    for repo in repos:
        text = " ".join([repo.title, repo.snippet, repo.content, " ".join(repo.tags)]).lower()
        languages = [str(x).lower() for x in repo.metadata.get("languages", [])]
        if "python" in languages or "python" in text:
            python_hits += 1
        if "pytorch" in text or "torch" in text:
            pytorch_hits += 1
        if any(word in text for word in ["config", "example", "script", "train"]):
            runnable_hits += 1
    bullets = [
        f"{python_hits} repository result(s) look Python-oriented.",
        f"{pytorch_hits} repository result(s) explicitly suggest PyTorch or torch fit.",
        f"{runnable_hits} repository result(s) mention runnable structure such as configs, examples, or training scripts.",
    ]
    score = python_hits + pytorch_hits + runnable_hits
    if any("rewrite" in item.lower() for item in constraints) and pytorch_hits == 0:
        bullets.append("Current constraints discourage rewrites, but strong PyTorch-specific evidence is limited.")
        score -= 1
    if score >= 6:
        return "High", bullets
    if score >= 3:
        return "Medium", bullets
    return "Low", bullets


def summarize_roles(results: Sequence[SearchResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        counts[item.role] = counts.get(item.role, 0) + 1
    return counts



def generate_key_findings(search_run: SearchRun, strong: Sequence[SearchResult], weak: Sequence[SearchResult], themes: Sequence[dict[str, Any]], local_literature: Sequence[SearchResult], repos: Sequence[SearchResult], papers: Sequence[SearchResult], benchmark_dossier: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    findings.append(f"The search run produced {len(strong)} strong result(s), {len(weak)} weak-but-relevant result(s), and {max(len(search_run.results) - len(strong) - len(weak), 0)} background result(s).")
    strong_roles = summarize_roles(strong)
    if strong_roles:
        findings.append("Strong-result role mix: " + ", ".join(f"{k}={v}" for k, v in sorted(strong_roles.items())) + ".")
    if themes:
        findings.append(f"Emergent themes from the retrieved evidence include: {', '.join(item['pattern'] for item in themes[:4])}.")
    if local_literature:
        findings.append(f"Local MLIP library contributed {len(local_literature)} PDF-backed result(s), led by `{local_literature[0].title}`.")
    if repos:
        findings.append(f"Repository evidence is headed by `{repos[0].title}`, which is the strongest implementation-oriented reference in this run.")
    if papers:
        findings.append(f"Recovered {len(papers)} paper-like result(s), which helps with method precedent even when implementation details remain thinner.")
    if benchmark_dossier.get("warnings"):
        findings.append("The current proposal/evidence local_context is benchmark-incomplete; force metrics are present but energy/gap/Q/runtime/control context is missing or under-specified.")
    if not repos:
        findings.append("Implementation-oriented evidence is still thin, so proposal confidence should stay conservative.")
    return findings[:7]


def generate_negative_evidence(repos: Sequence[SearchResult], papers: Sequence[SearchResult], strong: Sequence[SearchResult]) -> list[str]:
    negatives: list[str] = []
    if not strong:
        negatives.append("No strong evidence was recovered in the current search run.")
    if not repos:
        negatives.append("No repository results were strong enough to anchor implementation-fit judgment.")
    if not papers:
        negatives.append("No paper results were strong enough to anchor method precedent.")
    return negatives or ["No major negative evidence beyond the listed risks."]


def generate_risks(constraints: Sequence[str], repos: Sequence[SearchResult], papers: Sequence[SearchResult]) -> list[str]:
    risks: list[str] = []
    if any("rewrite" in item.lower() for item in constraints) and not repos:
        risks.append("Current constraints discourage major rewrites, but the retrieved evidence does not yet offer strong low-friction implementation guidance.")
    if repos and not papers:
        risks.append("The current evidence leans more implementation-oriented than method-justified.")
    if papers and not repos:
        risks.append("The current evidence leans more methodological than implementation-ready.")
    return risks or ["Main risk: overgeneralizing from incomplete retrieved evidence."]


def generate_proposal_angles(themes: Sequence[dict[str, Any]], implementation_fit: str, repos: Sequence[SearchResult], role_buckets: dict[str, list[SearchResult]]) -> list[str]:
    angles: list[str] = []
    pattern_names = {item["pattern"] for item in themes}
    framework_hits = role_buckets.get("framework", [])
    representation_hits = role_buckets.get("representation", [])
    decision_hits = role_buckets.get("decision", [])

    if decision_hits:
        angles.append(f"Low-risk angle: inspect the decision-oriented evidence led by `{decision_hits[0].title}` and extract the smallest training/config changes before attempting a new architecture.")
    if "atom reference energy" in pattern_names:
        angles.append("Low-risk angle: test an atom reference energy or atomref-style target decomposition if the current training targets permit it.")
    if "energy/force joint loss weighting" in pattern_names:
        angles.append("Low-risk angle: revisit energy-force loss weighting before attempting architectural changes.")
    if representation_hits:
        angles.append(f"Medium-risk angle: compare representation choices suggested by `{representation_hits[0].title}` before committing to a larger framework change.")
    if "locality / cutoff" in pattern_names:
        angles.append("Medium-risk angle: test cutoff sensitivity or neighborhood radius as a targeted ablation.")
    if "message passing depth" in pattern_names:
        angles.append("Medium-risk angle: compare a small number of interaction-depth settings before considering heavier redesign.")
    if framework_hits and not decision_hits:
        angles.append(f"Conservative next step: inspect framework-level evidence such as `{framework_hits[0].title}` and borrow only the lowest-friction training or evaluation ideas.")
    if implementation_fit == "Low" and not angles:
        angles.append("Conservative next step: gather better implementation evidence before committing to a new MLIP proposal direction.")
    elif not angles and repos:
        angles.append(f"Conservative next step: inspect the training setup in `{repos[0].title}` and translate only the lowest-friction ideas.")
    return angles[:6]


def generate_design_moves(themes: Sequence[dict[str, Any]], repos: Sequence[SearchResult], papers: Sequence[SearchResult], role_buckets: dict[str, list[SearchResult]]) -> list[str]:
    moves: list[str] = []
    pattern_names = {item["pattern"] for item in themes}
    design_hits = role_buckets.get("design", [])
    repo_ref = repos[0].title if repos else "no strong repo"
    paper_ref = papers[0].title if papers else "no strong paper"

    if "atom reference energy" in pattern_names:
        moves.append(f"Atom reference decomposition | principle: separate composition baseline from geometry residual | math: E=sum_i E_ref(Z_i)+E_res(x) | code: initialize/freeze or regularize species_energy_bias and train residual energy | expected: lower ISO17 energy drift | control: current learnable species_energy_bias only | source: {repo_ref}; {paper_ref}.")
    if "angular / three-body geometry" in pattern_names:
        moves.append(f"Angular local descriptor | principle: pair distances miss bond-angle chemistry | math: add cos(theta_ijk) or triplet RBF basis to messages | code: compute neighbor triplets inside cutoff and fuse a small angle MLP into atom_state | expected: better OOD geometry generalization | control: pair-distance-only model | source: {repo_ref}; {paper_ref}.")
    if "symmetry / invariance" in pattern_names or "equivariance" in pattern_names:
        moves.append(f"Symmetry audit before equivariant rewrite | principle: preserve translation, rotation, and permutation behavior | math: use scalar invariants unless adding a true equivariant block | code: replace ad hoc direction gates with distance/angle invariant gates or isolate them in an ablation | expected: less unstable force gradients | control: current direction_gate | source: {repo_ref}; {paper_ref}.")
    if "locality / cutoff" in pattern_names or "long-range effects" in pattern_names:
        moves.append(f"Physics-split short/long residual | principle: short-range repulsion and longer-range smooth interactions have different scales | math: E=E_short(r<r_s)+E_long(r<r_c)+E_atomref | code: separate heads with normalization and residual sum, not an unnormalized bypass | expected: stabilize energy without harming forces | control: current fused short/long stream | source: {repo_ref}; {paper_ref}.")
    if "force normalization" in pattern_names or "energy/force joint loss weighting" in pattern_names:
        moves.append(f"Dimensionally scaled energy-force loss | principle: energy and force errors have different units and sample-size scaling | math: normalize energy per atom and force by robust training-set scale before weighting | code: estimate scales in dataloader/train and use normalized joint loss | expected: less dataset-specific loss tuning | control: fixed energy_weight/force_weight | source: {repo_ref}; {paper_ref}.")
    if "energy conservation / force consistency" in pattern_names:
        moves.append(f"Conservative force-field constraint | principle: forces must remain exact gradients of one scalar energy surface | math: F_i=-dE/dR_i with smooth descriptors and no detached force head | code: add smoothness tests/regularizers around RBF, gates, and cutoff instead of adding independent force outputs | expected: fewer unstable force artifacts | control: current autograd-force model without smoothness audit | source: {repo_ref}; {paper_ref}.")
    if design_hits and not moves:
        moves.append(f"Inspect design evidence led by `{design_hits[0].title}` and convert one physics/math pattern into a controlled code ablation; avoid pure optimizer-only proposals.")
    if not moves:
        moves.append("Evidence is not design-specific enough; run follow-up searches for atomref, angular/triplet descriptors, force normalization, and invariant/equivariant MLIP implementations before proposing code.")
    return moves[:6]


def generate_followup_queries(themes: Sequence[dict[str, Any]], strong: Sequence[SearchResult], role_buckets: dict[str, list[SearchResult]]) -> list[str]:
    queries: list[str] = []
    for item in themes[:4]:
        queries.append(f"MLIP {item['pattern']} implementation")
    if role_buckets.get("framework"):
        queries.append("MLIP framework implementation comparison with explicit equations and code paths")
    if role_buckets.get("representation"):
        queries.append("equivariant vs invariant representation MLIP comparison")
    if role_buckets.get("decision"):
        queries.append("MLIP training config loss weighting ablation")
    if not strong:
        queries.append("MLIP training config comparison with loss weights data splits and code paths")
    return queries[:6]


def confidence_grade(strong: Sequence[SearchResult], weak: Sequence[SearchResult], themes: Sequence[dict[str, Any]], implementation_fit: str) -> tuple[str, str]:
    source_types = {item.source_type for item in strong}
    if len(strong) >= 4 and len(source_types) >= 2 and implementation_fit != "Low":
        return "A", "Strong retrieved evidence with multiple sources and usable implementation fit."
    if len(strong) >= 2 and themes:
        return "B", "Useful retrieved evidence exists, though some gaps remain."
    if strong or weak:
        return "C", "Some directional evidence exists, but it is still partial or weak."
    return "D", "Very weak or exploratory evidence; do not let it dominate proposal ranking."


def markdown_list(items: Iterable[str], empty_text: str = "None") -> str:
    values = [item for item in items if item]
    if not values:
        return f"- {empty_text}"
    return "\n".join(f"- {item}" for item in values)


def format_results(results: Sequence[SearchResult], empty_text: str) -> str:
    if not results:
        return f"- {empty_text}"
    lines = []
    for item in results:
        lines.append(f"- [{item.title}]({item.url}) — source: {item.source_type}; role: {item.role}; tier: {item.tier}; score: {item.score:.2f}. {item.snippet}")
    return "\n".join(lines)


def format_themes(themes: Sequence[dict[str, Any]]) -> str:
    if not themes:
        return "- No strong recurring themes were extracted."
    lines = []
    for item in themes:
        examples = "; ".join(f"[{ex['title']}]({ex['url']}) ({ex['source_type']})" for ex in item["examples"])
        lines.append(f"- **{item['pattern']}** — support count: {item['support_count']}; source types: {', '.join(item['source_types'])}. Examples: {examples}")
    return "\n".join(lines)


def build_brief_markdown(search_run: SearchRun, strong: Sequence[SearchResult], weak: Sequence[SearchResult], background: Sequence[SearchResult], local_literature: Sequence[SearchResult], repos: Sequence[SearchResult], papers: Sequence[SearchResult], other: Sequence[SearchResult], role_buckets: dict[str, list[SearchResult]], themes: Sequence[dict[str, Any]], key_findings: Sequence[str], negative_evidence: Sequence[str], risks: Sequence[str], implementation_fit: tuple[str, list[str]], design_moves: Sequence[str], proposal_angles: Sequence[str], followup_queries: Sequence[str], confidence: tuple[str, str], benchmark_dossier: dict[str, Any]) -> str:
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return f"""# Question

{search_run.question}

# Local context

{search_run.local_context or 'No extra local context was provided.'}

# benchmark_dossier

```json
{json.dumps(benchmark_dossier, ensure_ascii=False, indent=2)}
```

# Strong evidence

{format_results(strong, 'No strong evidence was recovered.')}

# Weak but relevant evidence

{format_results(weak, 'No weak-but-relevant evidence was recovered.')}

# Background context

{format_results(background, 'No background-only evidence was recorded.')}

# Key findings

{markdown_list(key_findings)}

# mathematical_forms

{format_themes(themes) or 'No explicit mathematical forms were extracted from the structured evidence.'}

# physical_principles

{format_results(role_buckets.get('design', []), 'No explicit physics/design evidence was recovered.')}

# chemical_regime

{format_results(local_literature or papers, 'No chemistry-regime evidence was recovered.')}

# textual_evidence

{format_results(list(strong) + list(weak), 'No textual paper/source evidence was recovered.')}

# code_evidence

{format_results(repos, 'No repository or key-code evidence was recovered.')}

# Framework evidence

{format_results(role_buckets.get('framework', []), 'No framework-oriented evidence was recovered.')}

# Representation evidence

{format_results(role_buckets.get('representation', []), 'No representation-oriented evidence was recovered.')}

# Decision evidence

{format_results(role_buckets.get('decision', []), 'No decision-oriented evidence was recovered.')}

# Design evidence

{format_results(role_buckets.get('design', []), 'No physics/math/design-oriented evidence was recovered.')}

# Local MLIP library

{format_results(local_literature, 'No local MLIP library evidence was recovered.')}

# Relevant papers

{format_results(papers, 'No paper-like evidence was recovered.')}

# Relevant repos

{format_results(repos, 'No repository evidence was recovered.')}

# Other sources

{format_results(other, 'No additional source types were recovered.')}

# Useful patterns

{format_themes(themes)}

# Negative evidence

{markdown_list(negative_evidence)}

# Risks / mismatches

{markdown_list(risks)}

# Implementation fit

**{implementation_fit[0]}**

{markdown_list(implementation_fit[1])}

# capability_gap

{markdown_list(implementation_fit[1])}

# Implementable design moves

{markdown_list(design_moves)}

# implementable_design_moves

{markdown_list(design_moves)}

# Suggested proposal angles

{markdown_list(proposal_angles)}

# exploit_angles

{markdown_list(proposal_angles)}

# jump_angles

{markdown_list(followup_queries)}

# Follow-up queries

{markdown_list(followup_queries)}

# Confidence

**{confidence[0]}** — {confidence[1]}

# Metadata

- generated_at_utc: {today}
- source_priority: {', '.join(search_run.source_priority) if search_run.source_priority else 'none'}
- constraints: {', '.join(search_run.constraints) if search_run.constraints else 'none'}
"""


def append_index_record(index_path: Path, record: dict[str, Any]) -> None:
    ensure_parent(index_path)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def note_path_for_pattern(notes_dir: Path, pattern_name: str) -> Path:
    return notes_dir / f"{slugify(pattern_name).replace('-', '_')}.md"


def promote_notes(config: Config, question: str, patterns: Sequence[dict[str, Any]], confidence: tuple[str, str], brief_rel_path: str, constraints: Sequence[str]) -> list[Path]:
    promoted: list[Path] = []
    if confidence[0] not in NOTE_PROMOTION_MIN_CONFIDENCE:
        return promoted
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    for item in patterns:
        if item["support_count"] < NOTE_PROMOTION_MIN_SUPPORT:
            continue
        if len(item["source_types"]) < NOTE_PROMOTION_MIN_SOURCE_TYPES:
            continue
        note_path = note_path_for_pattern(config.notes_dir, item["pattern"])
        body = f"""---
title: {item['pattern']}
topic: mlip
confidence: {confidence[0]}
last_verified: {today}
sources: [{', '.join(item['source_types'])}]
---

## Update {today}

### Claim

The pattern **{item['pattern']}** appeared in {item['support_count']} retrieved result(s) across {', '.join(item['source_types'])} while investigating:

> {question}

### Constraints

- {chr(10).join(constraints) if constraints else 'none provided'}

### Source brief

- `{brief_rel_path}`
"""
        ensure_parent(note_path)
        note_path.write_text(body, encoding="utf-8")
        promoted.append(note_path)
    return promoted


def build_output_paths(config: Config) -> tuple[Path, Path]:
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    stem = f"{today}_{config.slug}"
    return config.briefs_dir / f"{stem}.md", config.briefs_dir / f"{stem}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an MLIP brief from a search-run JSON file.")
    parser.add_argument("--search-run", required=True, help="Path to a search-run.v1 JSON file.")
    parser.add_argument("--workspace-root", help="Workspace root. Defaults to the repository root inferred from this script.")
    parser.add_argument("--briefs-dir", help="Override brief output directory.")
    parser.add_argument("--notes-dir", help="Override notes directory.")
    parser.add_argument("--index-path", help="Override index path.")
    parser.add_argument("--slug", help="Optional output slug override.")
    parser.add_argument("--auto-promote-notes", action="store_true", help="Promote repeated high-confidence themes into notes.")
    return parser.parse_args()


def infer_workspace_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    script_path = Path(__file__).resolve()
    parents = list(script_path.parents)
    if len(parents) >= 4:
        return parents[3]
    return Path.cwd().resolve()


def build_config(args: argparse.Namespace, search_run: SearchRun) -> Config:
    workspace_root = infer_workspace_root(args.workspace_root)
    briefs_dir = Path(args.briefs_dir).expanduser().resolve() if args.briefs_dir else workspace_root / "research_runtime" / "knowledge" / "briefs"
    notes_dir = Path(args.notes_dir).expanduser().resolve() if args.notes_dir else workspace_root / "research_runtime" / "knowledge" / "notes"
    index_path = Path(args.index_path).expanduser().resolve() if args.index_path else workspace_root / "research_runtime" / "knowledge" / "index.jsonl"
    slug = args.slug or slugify(search_run.question)
    return Config(workspace_root=workspace_root, briefs_dir=briefs_dir, notes_dir=notes_dir, index_path=index_path, slug=slug, auto_promote_notes=args.auto_promote_notes)


def main() -> int:
    args = parse_args()
    search_run = load_search_run(Path(args.search_run).expanduser().resolve())
    config = build_config(args, search_run)

    config.briefs_dir.mkdir(parents=True, exist_ok=True)
    config.notes_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(config.index_path)
    if not config.index_path.exists():
        config.index_path.write_text("", encoding="utf-8")

    benchmark_dossier = extract_benchmark_dossier(search_run.local_context)
    parsed = parse_results(search_run.results)
    buckets = bucket_by_tier(parsed)
    role_buckets = bucket_by_role(parsed)
    local_literature, repos, papers, other = classify_results(parsed)
    themes = extract_emergent_themes(buckets.strong + buckets.weak)
    implementation_fit = assess_implementation_fit(repos, search_run.constraints)
    key_findings = generate_key_findings(search_run, buckets.strong, buckets.weak, themes, local_literature, repos, papers, benchmark_dossier)
    negative_evidence = generate_negative_evidence(repos, papers, buckets.strong)
    risks = generate_risks(search_run.constraints, repos, papers)
    design_moves = generate_design_moves(themes, repos, papers, role_buckets)
    proposal_angles = generate_proposal_angles(themes, implementation_fit[0], repos, role_buckets)
    followup_queries = generate_followup_queries(themes, buckets.strong, role_buckets)
    confidence = confidence_grade(buckets.strong, buckets.weak, themes, implementation_fit[0])
    local_context_completeness = "low" if benchmark_dossier.get("warnings") else "benchmark-centric"

    brief_md = build_brief_markdown(
        search_run=search_run,
        strong=buckets.strong,
        weak=buckets.weak,
        background=buckets.background,
        local_literature=local_literature,
        repos=repos,
        papers=papers,
        other=other,
        role_buckets=role_buckets,
        themes=themes,
        key_findings=key_findings,
        negative_evidence=negative_evidence,
        risks=risks,
        implementation_fit=implementation_fit,
        design_moves=design_moves,
        proposal_angles=proposal_angles,
        followup_queries=followup_queries,
        confidence=confidence,
        benchmark_dossier=benchmark_dossier,
    )

    brief_md_path, brief_json_path = build_output_paths(config)
    brief_md_path.write_text(brief_md, encoding="utf-8")
    brief_payload = {
        "question": search_run.question,
        "local_context": search_run.local_context,
        "benchmark_dossier": benchmark_dossier,
        "constraints": search_run.constraints,
        "source_priority": search_run.source_priority,
        "strong_evidence": [asdict(item) for item in buckets.strong],
        "weak_evidence": [asdict(item) for item in buckets.weak],
        "background_evidence": [asdict(item) for item in buckets.background],
        "local_literature": [asdict(item) for item in local_literature],
        "relevant_repos": [asdict(item) for item in repos],
        "relevant_papers": [asdict(item) for item in papers],
        "other_sources": [asdict(item) for item in other],
        "role_buckets": {key: [asdict(item) for item in value] for key, value in role_buckets.items()},
        "useful_patterns": themes,
        "key_findings": key_findings,
        "negative_evidence": negative_evidence,
        "risks_or_mismatches": risks,
        "implementation_fit": {"level": implementation_fit[0], "details": implementation_fit[1]},
        "implementable_design_moves": design_moves,
        "suggested_proposal_angles": proposal_angles,
        "followup_queries": followup_queries,
        "confidence": {"grade": confidence[0], "reason": confidence[1], "local_context_completeness": local_context_completeness},
    }
    brief_json_path.write_text(json.dumps(brief_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    append_index_record(config.index_path, {
        "kind": "brief",
        "title": search_run.question,
        "slug": config.slug,
        "path": str(brief_md_path.relative_to(config.workspace_root)),
        "confidence": confidence[0],
        "updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
    })

    promoted: list[Path] = []
    if config.auto_promote_notes:
        promoted = promote_notes(
            config=config,
            question=search_run.question,
            patterns=themes,
            confidence=confidence,
            brief_rel_path=str(brief_md_path.relative_to(config.workspace_root)),
            constraints=search_run.constraints,
        )

    print(f"Brief markdown: {brief_md_path}")
    print(f"Brief json: {brief_json_path}")
    if promoted:
        print("Promoted notes:")
        for note in promoted:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
