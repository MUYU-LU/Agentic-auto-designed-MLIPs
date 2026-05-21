# MLIP-Evidence Delta Brief

This markdown is a proposal-facing evidence delta index. It must not restate the full proposal context, full benchmark dossier, full current-code profile, full prior briefs, or full notes. JSON package files are the handoff source of truth.

# evidence_package_index

Paths to package JSON artifacts.

# package_contract

- This brief is an evidence-delta index, not a full proposal context dump.
- JSON files in evidence_package_index are the handoff source of truth.
- No provenance -> no strong evidence.
- No mechanism card -> no proposal mechanism.
- No formula derivation / algorithm / repo code path / code trace / current insertion point -> weak evidence only.
- No patch blueprint -> not implementation-ready.
- Benchmark diagnosis is internal diagnosis, not external mechanism evidence.

# evidence_quality

Grade, fresh/reused source status, diagnosis_only flag, and whether this package is usable for proposal or implementation.

# context_references

Paths and compact warnings for source unit, proposal context, current_code_profile.json, benchmark_diagnosis.json, and generation_memory.json. Do not paste full context or full generation summaries here.

# source_attempt_summary

Compact summary of sources read or attempted, grouped by type. Include failed and reused sources.

# what_is_new_for_proposal

The concrete new evidence, mechanism ids, allowed mechanisms, blocked mechanisms, and whether anything is proposal-ready.

# new_mechanism_cards_summary

Compact summaries only. Full cards live in `mechanism_cards.json`.

Each strong card must include:
- formula / algorithm / training rule or concrete code pattern
- formula derivation or reasoning into implementation consequences
- tensor/data-flow meaning
- physical or chemical constraint
- source refs that can support strong evidence
- repo path / class / function when repo evidence is used
- code trace from repo pattern to current unit
- current-code insertion point
- bounded edit
- expected benchmark effect
- ablation/control

# new_patch_blueprints_summary

Blueprint summaries only. Full pseudo-diff / insertion-point details live in `patch_blueprints.json`.

# proposal_constraints

Allowed strong mechanisms, weak/hypothesis mechanisms, blocked mechanisms, and required proposal sections.

# audit_report

Evidence gaps and downgrade reasons.

# what_not_to_use_as_strong_evidence

Weak mechanisms, blocked mechanisms, filename-only PDF matches, repo metadata-only hits, prior notes/briefs without current provenance, and benchmark diagnosis.

# followup_queries

Narrow next-step questions tied to missing mechanism evidence.
