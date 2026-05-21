# Contracts

## Fixed benchmark contract

Keep these fixed:
- benchmark data
- split semantics
- metric field names
- eval meaning

The benchmark may emit derived fields such as:
- `mixed_force_mae`
- `mixed_energy_mae`
- `gap_penalty`
- `Q_dataset`
- `Q_total`

But the benchmark schema itself must remain stable.

## Adaptation profiles

Adaptation profiles are optional external diagnostics. They are not part of the fixed benchmark schema and must not change `Q_total`.

Current supported profile:
- `mad10k`: deterministic stratified 10,000-frame MAD training subset plus full MAD validation/test files.

Use profile outputs under `outputs/profiles/<profile_name>/` to diagnose whether a runnable unit's architecture and train code transfer to a broader multi-domain dataset. Profile failures should be recorded as profile failures, not full benchmark failures.

## Runnable unit contract

A runnable unit may be:
- `base_unit`
- `seed_unit`
- `generations/generation_xxx/proposal_xxx`

Each runnable unit must remain executable through its entrypoint.

Expected outputs under `outputs/<dataset>/`:
- `model.pt`
- `train_history.json`
- `benchmark_metrics.json`

Expected cross-dataset outputs:
- `outputs/summary.json` (recommended)
- ledger append performed by the collection step

## Editable scope

The agent may edit:
- `model/model.py`
- `model/train.py`

The agent may not silently change:
- benchmark split definitions
- metric field names
- eval entrypoint contract

## Milestone contract

For a successful child unit:
- compute `Q_dataset` for each benchmark using fixed seed/base anchors
- compute `Q_total` using the active q schema in `benchmark/anchors.json`
- write `benchmark_version` beside `Q_total`
- compute `G_delta = Q_total_child - Q_total_parent`

`G_delta > 0` means progress.
`G_delta < 0` means regression.

## Benchmark-centric decision contract

Proposal writing, evidence local_context, and round comparison must not collapse benchmark behavior to force-only summaries.

Whenever a unit is summarized for proposal or evidence use, the summary must include, per dataset:
- split-level energy metrics
- split-level force metrics
- `mixed_force_mae`
- `mixed_energy_mae`
- `gap_penalty`
- `Q_dataset`

And across datasets:
- `Q_rmd17`
- `Q_iso17`
- `Q_total`
- `G_delta` if reference exists

The summary should also include:
- training-history trend summary
- runtime / timeout / failure status
- whether the unit is a control replicate
- comparison to the current control replicate if one exists

Do not treat “force is best” as equivalent to “benchmark is best”.

## Local execution shell contract

The workflow uses three execution concepts:

1. local staging runtime
   - editable workspace mirror
   - proposal files
   - local code edits
   - mirrored status files

2. local execution shell
   - fixed local Python environment used for `MLIP-Autoresearch` helper scripts and advisory sanity checks
   - must not be treated as benchmark source of truth

3. active remote runtime
   - authoritative execution source for remote smoke, full launches, logs, and benchmark outputs

All `MLIP-Autoresearch` local Python-related commands must use the configured local execution shell.

Use one of:
- `source <conda_sh> && conda activate <conda_env> && python ...`
- direct `local_python`

Do not use:
- bare `python3`
- `/usr/bin/python3`
- system Python
- an unspecified shell environment

This contract does not override `MLIP-Evidence`. Evidence gathering follows the `MLIP-Evidence` Evidence Execution Environment for local PDF and repository reading.

Every local Python sanity check must print:
- `which python`
- `python -V`
- `sys.executable`

If a local dependency import fails:
- do not install packages automatically
- do not block the round automatically
- mark the local sanity check as advisory-failed or advisory-skipped
- rely on remote smoke as the required launch gate

## Ledger contract

Successful runnable units must append one record to:
- `research_runtime/ledger/frontier.jsonl`

The record must include:
- unit
- source_unit
- family
- phase
- jump_type
- Q_rmd17
- Q_iso17
- Q_total
- G_delta
- runtime_sec
- status
- capabilities

A round-level summary file should also be written:
- `research_runtime/ledger/generation_xxx_summary.json`

The round summary must include, for each unit:
- `proposal_file`
- `control_replicate`
- `run_state`
- `implementation_state`
- per-dataset split metrics
- per-dataset mixed metrics
- per-dataset `gap_penalty`
- per-dataset `Q_dataset`
- `Q_total`
- training-history trend summary
- runtime / timeout / failure summary

## State machine contract

Materialization is not implementation.

Each non-base runnable unit must have `implementation_status.json` with one of:
- `materialized`
- `implementation_needed`
- `implemented`
- `remote_smoke_pending`
- `remote_smoke_passed`
- `launch_ready`
- `repairing`
- `abandoned`

Each runnable unit must have `run_status.json` with one of:
- `not_started`
- `queued`
- `running`
- `terminal_success`
- `terminal_failure`
- `terminal_timeout`
- `terminal_abandoned`

A full benchmark launch is valid only when:
- `implementation_state == "launch_ready"`
- `remote_smoke_passed == true`
- `remote_synced == true`
- `run_state` is launchable (`not_started`, `terminal_failure`, or `terminal_timeout`)

## Round-state contract

`ledger/round_state.json` is a global control artifact.
The main agent owns its logical contents, but scripts must mirror the authoritative copy to remote when `runtime == remote`.

At minimum it must include:
- `workflow_state`
- `current_generation`
- `active_writer_unit`
- `active_evidence_task`
- `source_of_truth`
- `blocked_reason`
- `last_completed_generation`
- `next_recommended_step`
- `last_transition_utc`

When preparing a new round, it must also carry explicit evidence and proposal provenance:
- `continuation_source_unit`
- `active_evidence_brief`
- `evidence_for_source_unit`
- `evidence_mode`
- `proposal_context_file`
- `proposal_writer_session` or `proposal_source`
- `proposal_directory_ready`

Proposal writing is not ready unless:
- `active_evidence_brief` exists
- `evidence_for_source_unit == continuation_source_unit`
- `proposal_context_file` exists and was generated from the active evidence brief

Selection and materialization are not ready unless:
- the proposal directory contains the required proposal files
- `proposal_directory_ready == true`
- `proposal_writer_session` or `proposal_source` records where the proposal set came from

If local and remote round-state differ, no launch is allowed until reconciliation is performed.

## Tree-State Contract

`ledger/tree_state.json` and `ledger/lineage_stats.json` are derived artifacts.
They must be rebuilt from runtime history rather than hand-edited.

The tree state must preserve:
- unit parent/source relationship
- terminal status
- active dataset Q fields such as `Q_rmd17`, `Q_iso17`, `Q_mad10k`, plus `Q_total`
- `benchmark_version`
- `G_delta = Q_total_child - Q_total_parent` when a parent exists
- proposal family / phase / jump type / budget class
- selected-as-continuation counts
- family, jump-type, and phase aggregate statistics

Continuation decisions may use tree state for PUCT-like exploration accounting, but they must not change benchmark metrics, Q formulas, or runnable contracts.

## Implementation handoff contract

Every materialized runnable unit should contain `research_context/` copied by `create_unit.py`.

Implementation and repair subagents must read, at minimum:
- `research_context/manifest.json`
- `research_context/proposal.md`
- `research_context/context.md`

When present, they must also read:
- `research_context/evidence_package/mechanism_cards.json`
- `research_context/evidence_package/patch_blueprints.json`
- `research_context/evidence_package/proposal_constraints.json`
- `research_context/selection_row.json`

The implementation must follow the proposal's `files_to_edit`, `code_insertion_points`, `minimal_edit_plan`, and `implementation_checklist`.

If `research_context/` or its core files are missing, the subagent must stop and report missing handoff instead of inventing an implementation from chat memory or source code alone.

After edits, `mark_unit_implemented.py` writes `research_context/implementation_report.json`, including tracked diffs for `model/model.py` and `model/train.py` against the source unit.

## Remote state mirroring contract

When `runtime == remote`, every script that mutates local unit status or local round state must mirror the updated status file(s) to remote immediately.

At minimum, the following files must be mirrored:
- `implementation_status.json`
- `run_status.json`
- `unit_meta.json` (when changed)
- `ledger/round_state.json` (when changed)

## Proposal contract

Every proposal must declare:
- `family`
- `phase`
- `jump_type`
- `budget_class`
- `expected_capability_gain`

Capacity-scaling proposals are valid only when they are explicit bounded bottleneck tests. They must keep benchmark/runtime semantics fixed, edit only code-level MLIP defaults in `model/model.py` or `model/train.py`, and state the capacity/depth/resolution/training-budget hypothesis plus compute/runtime risk.

Proposal context must be evidence-bound.
The context generator must not select evidence by latest modification time or by filename guessing.
It must read `ledger/round_state.json`, validate that `active_evidence_brief` exists, and validate that `evidence_for_source_unit` matches the current `continuation_source_unit`.
