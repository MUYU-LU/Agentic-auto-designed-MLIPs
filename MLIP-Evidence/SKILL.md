---
name: MLIP-Evidence
description: Build self-contained MLIP-Evidence packages for proposal-time or round-review decisions. Use when an upstream workflow needs local-literature reading, paper/arXiv reading, repository discovery, repository deep-read, code-pattern extraction, implementation-fit judgment, and exploit/jump recommendation. This skill is evidence-only and must not run experiments or advance the round.
---

# MLIP-Evidence

Use this skill to produce evidence that can be consumed by a different proposal or implementation agent with no prior chat context.

The output is an **evidence package**. The markdown brief is only a compact human-readable index.

## 1. Scope

This skill may:
- profile the current runnable unit or continuation source
- read local MLIP PDFs
- read papers / arXiv
- discover repository links from papers and record attempt results
- search and deep-read repositories
- extract equations, algorithms, training rules, code patterns, and implementation constraints
- compare external mechanisms to current code
- write one bounded evidence package and one compact brief

This skill must not:
- edit runnable-unit code
- materialize runnable units
- launch runs
- choose the next round
- write `selection.json`
- redefine benchmark semantics

## 2. Evidence Environment

Use the normal OpenClaw shell for evidence gathering:
- shell: `/bin/bash`
- workdir: `<OPENCLAW_WORKSPACE>`

Do not default to conda for `MLIP-Evidence`. Evidence gathering is separate from model sanity and benchmark Python environments.

Allowed evidence tools:
- `/usr/bin/pdftotext` through `scripts/extract_local_paper.py`
- `pypdf` only as fallback
- local `git clone` / `git pull` through `scripts/read_github_repo.py`
- the shell Python available in this `/bin/bash` environment for evidence scripts

Forbidden by default:
- OpenClaw built-in `pdf` tool for local host paths such as `<LOCAL_PATH>`
- conda model environments for PDF/repo extraction
- unspecified shell environments
- automatic package installation

Repository cache:
- default cache root: `<OPENCLAW_WORKSPACE>/.cache/MLIP-Evidence/repos`
- do not copy full git repositories into `research_runtime`
- package artifacts should store compact repo JSON with cache path, commit, files read, and code traces

Git/proxy rule:
- preserve current shell proxy environment by default
- use `scripts/read_github_repo.py --no-proxy` only for explicit one-off diagnostics when proxy settings are known to be wrong

Evidence helper failures are diagnostic only. They are not benchmark gates.

## 3. Workflow

For proposal-time evidence:

1. Read the calling task and local context.
2. Read `references/local_mlip_literature.md`.
3. Read relevant notes under `research_runtime/knowledge/notes/`.
4. Read relevant recent briefs under `research_runtime/knowledge/briefs/`.
5. Read canonical completed-generation memory:
   - `ledger/generation_summaries/`
   - `ledger/generation_reports/`
   - `ledger/all_attempts.jsonl`
   - `ledger/unit_cards/`
   - `ledger/mechanism_outcomes.jsonl`
   - `ledger/negative_patterns.jsonl`
   - `ledger/partial_positive_patterns.jsonl`
6. Profile the current source unit.
7. Write `source_plan.json`.
8. Run `scripts/run_evidence.py --source-plan <PLAN> --require-external-evidence` unless caller explicitly requests diagnosis-only evidence.
9. Inspect generated paper/repo artifacts.
10. Write a mechanism-card draft grounded in those artifacts.
11. Run `scripts/materialize_mechanism_cards.py --package-dir <PACKAGE_DIR> --mechanism-cards <DRAFT_JSON>`.
12. If `evidence_quality.needs_source_expansion == true`, expand the source plan within budget and repeat.
13. Use the package/brief for proposal context only when `evidence_quality.usable_for_proposal == true`, unless caller explicitly accepts diagnosis-only evidence.

## 4. Source Plan And Novelty

The skill must not use a fixed default list of named papers or repositories. The evidence agent chooses sources from the current question, local literature, repository links, search results, and recent source history.

A valid proposal-time source plan includes:

```json
{
  "version": "source_plan.v1",
  "source_unit": "generation_xxx/proposal_yyy",
  "mode": "exploit | jump | balanced",
  "selection_intent": "mechanism gap this evidence run tries to fill",
  "selected_pdfs": [
    {
      "path": "/path/to/local.pdf",
      "why_selected": "expected mechanism, not title-only relevance",
      "expected_evidence": ["equation", "algorithm", "training rule"]
    }
  ],
  "selected_repos": [
    {
      "repo": "owner/name or URL",
      "why_selected": "expected implementation pattern relevant to current insertion points",
      "expected_evidence": ["repo file", "class/function", "config/training pattern"]
    }
  ],
  "novelty_policy": {
    "lookback_runs": 3,
    "min_attempted_external_sources": 2,
    "min_new_sources": 1,
    "max_recent_reused_sources": 1
  },
  "source_discovery_budget": {
    "min_successful_pdf_sources": 1,
    "min_successful_repo_sources": 1,
    "min_strong_mechanism_cards": 1,
    "max_pdf_attempts": 6,
    "max_repo_attempts": 4,
    "max_external_source_attempts": 10,
    "max_source_expansion_rounds": 2
  }
}
```

Source reuse is allowed only when it extracts a new mechanism or the caller explicitly allows reuse.

`run_evidence.py` must canonicalize source-plan inputs:
- local PDFs are staged under `research_runtime/knowledge/evidence_runs/source_inputs/`
- package `source_plan.json` records staged path plus `original_path`
- repositories are normalized to canonical `owner/repo`
- package `source_plan.json` records both `repo` and `original_repo` when input was a full URL

Proposal-time evidence should keep expanding sources until it either materializes at least the configured strong mechanism-card count or exhausts source budget. If the budget is exhausted, mark the package diagnosis-only instead of forcing a mechanism.

## 5. Mechanism Standard

Evidence must extract transferable mechanisms, not summarize papers.

For proposal-time evidence, a strong mechanism must include:
- current-run artifact refs such as `paper_artifact:paper_001` or `repo_artifact:repo_001`
- equation, algorithm update, loss, basis, message update, training rule, or concrete code pattern
- formula derivation or reasoning from source expression/code to current-code implication
- tensor shapes or data-flow meaning when model code is affected
- physical / chemical constraint and implementation implication
- repository path / class / function and implementation pattern when repo evidence exists
- current-code insertion point in `model/model.py` or `model/train.py`
- bounded edit plan
- ablation/control
- expected effect on energy, force, gap, Q, runtime, and failure risk

Do not use model-family labels, architecture-style labels, analogy labels, paper titles, repository names, or abstracts as evidence by themselves.

If artifacts do not contain explicit formula, derivation, algorithm, code pattern, or implementation detail, do not invent it. Record the absence and downgrade the item to weak/background evidence.

## 6. Artifact-To-Mechanism Materialization

Source collection alone is not a complete evidence run.

Required materialization:

1. Run `run_evidence.py`.
2. Read:
   - `paper_artifacts/*.json`
   - `repo_artifacts/*.json`
   - `source_artifacts_index.json`
   - `source_analysis_requirements.json`
3. Write mechanism-card draft JSON with derivations and code traces grounded in inspected artifacts.
4. Run `scripts/materialize_mechanism_cards.py --package-dir <PACKAGE_DIR> --mechanism-cards <DRAFT_JSON>`.
5. Use updated package/brief only if quality gates allow it.

Mechanism-card draft may be a list or an object with `cards`.

Each strong card must include:
- `source_refs`
- `mathematical_form` or algorithmic update
- `formula_derivation`
- `tensor_shapes` or `data_flow`
- `repo_code_path` and `repo_code_trace` when repo evidence exists
- `current_code_insertion_point`
- `bounded_edit`
- `expected_benchmark_effect`
- `ablation_or_control`

If this step is skipped, the package remains bootstrap/diagnosis evidence even if PDFs/repos were successfully read.

## 7. Benchmark And Memory Context

Proposal-time evidence must preserve benchmark context, not collapse to force-only summaries.

Use all present fields and list absent fields as `missing_context_fields`:
- split-level energy metrics
- split-level force metrics
- `mixed_force_mae`
- `mixed_energy_mae`
- `gap_penalty`
- `Q_dataset`
- `Q_rmd17`
- `Q_iso17`
- `Q_total`
- `G_delta`
- train-history trend summaries
- last epoch summary
- best validation force/energy
- runtime / timeout / failure class
- repair count
- implementation state
- control replicate comparison

Write compact and full memory audit to `generation_memory.json`.

The rendered brief must reference memory files and summarize counts/latest relevant outcomes. It must not paste full generation summaries.

Legacy flat files such as `ledger/generation_006_summary.json` are compatibility artifacts. Prefer canonical memory under `ledger/generation_summaries/` when present.

## 8. Evidence Package Contract

The evidence package is the source of truth.

Each run must create:

```text
research_runtime/knowledge/evidence_runs/<run_id>/
  evidence_brief.md
  evidence_run.json
  evidence_quality.json
  evidence_provenance.json
  current_code_profile.json
  benchmark_diagnosis.json
  generation_memory.json
  mechanism_cards.json
  patch_blueprints.json
  proposal_constraints.json
  audit_report.json
  source_plan.json
  source_artifacts_index.json
  source_novelty.json
  source_analysis_requirements.json
  paper_artifacts/*.json
  repo_artifacts/*.json
```

Also write a backward-compatible rendered brief under:
- `research_runtime/knowledge/briefs/`

Hard gates:
- no provenance -> no strong evidence
- no mechanism card -> no proposal mechanism
- no formula derivation / algorithm / repo code trace / current insertion point -> weak evidence only
- no patch blueprint -> not implementation-ready
- benchmark diagnosis is internal diagnosis, not external mechanism evidence
- prior briefs and notes are anchors / hypotheses unless backed by current package provenance
- if external evidence is required and no source-plan PDF/repo is successfully read, the package is diagnosis-only
- if all external sources are recent repeats and reuse was not explicitly allowed, source novelty fails and the package is diagnosis-only

For proposal-time evidence, `evidence_quality.usable_for_proposal` should be true only when at least one mechanism card has formula reasoning, code trace, provenance, insertion point, and bounded edit.

## 9. Brief Contract

The rendered brief is a proposal-facing delta index. It must be compact.

It should include only:
- evidence_package_index
- evidence_quality
- context_references
- source_attempt_summary
- what_is_new_for_proposal
- new_mechanism_cards_summary
- new_patch_blueprints_summary
- proposal_constraints
- audit_report
- what_not_to_use_as_strong_evidence
- followup_queries

It must not restate:
- full local_context
- full current-code profile
- full benchmark dossier
- full generation summaries
- full prior briefs
- full notes

Machine-readable details live in package JSON files. Follow `references/brief_template.md` for the compact rendering.

## 10. Local Literature Rules

A local PDF is not verified evidence merely because its filename looks relevant.

A local PDF counts as locally verified only when text extraction succeeds. If extraction fails:
- mark it unreadable / unverified
- do not promote it to strong evidence
- do not cite filename guesses as paper content

Read `references/local_mlip_literature.md` for local library and PDF extraction rules.

## 11. Repository Rules

Repository metadata alone is not enough for strong implementation fit.

For strong repo/code evidence, record:
- README text
- repository tree
- key code files
- config / examples
- training entrypoints
- dependency files
- exact path / class / function names
- code pattern mapped to current unit insertion point

If only repository search metadata is available, downgrade confidence.

## 12. Guardrails

- Do not replace `MLIP-Autoresearch`.
- Do not act as the round controller.
- Do not overclaim implementation fit from a repository name or paper abstract.
- Do not recommend large rewrites as bounded next steps.
- Distinguish method plausibility, implementation fit, benchmark suitability, and runtime feasibility.
- Promote notes only when confidence is high and support is multi-source.

## 13. Persistence

Save one package per evidence run:
- `research_runtime/knowledge/evidence_runs/<run_id>/`

Save one backward-compatible brief:
- `research_runtime/knowledge/briefs/`

The value of this skill is not finding links. The value is a self-contained MLIP research dossier that compounds across generations.
