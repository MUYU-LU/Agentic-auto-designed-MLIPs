---
name: MLIP-Autoresearch
description: Orchestrate MLIP evolution under fixed benchmark semantics. Use when inspecting active remote runtime state, preparing seed/base runs, creating generation rounds, materializing selected runnable units, enforcing implementation and remote-smoke gates, launching full-benchmark batches, collecting outputs, computing milestone deltas, and preparing proposal context for phase-aware MLIP evolution.
---

# MLIP-Autoresearch

Use this skill to run the outer loop of MLIP evolution. The skill keeps the benchmark fixed while evolving runnable-unit code.

## 1. Fixed Boundary

Keep these fixed:
- benchmark data
- split semantics
- eval semantics
- benchmark metric field names
- runnable unit entrypoint contract

Do not redefine benchmark formulas during a generation loop.

`config.json` is benchmark/runtime configuration, not the search surface for model quality.

MLIP architecture/training knobs belong in code:
- `model/model.py`: architecture, representation, cutoff/basis defaults, message passing, readout
- `model/train.py`: optimizer, scheduler, training budget, loss/objective weights

New materialized units may include `research_context/config_to_code_migration.json`. Treat it as handoff context, not as a proposal-specific edit target.

Read details when needed:
- `references/layout.md`
- `references/contracts.md`
- `references/proposal_format.md`

## 2. Source Of Truth

The active runtime surface is declared in `config.json`.

This workflow has three execution concepts:

- **local staging runtime**: editable mirror for proposals, local unit code, status mirrors, logs, and bundled scripts.
- **local execution shell**: configured local Python/shell environment for `MLIP-Autoresearch` helper scripts and advisory sanity checks.
- **active remote runtime**: authoritative source for remote smoke, launch logs, benchmark outputs, terminal states, and benchmark state when `runtime = remote`.

If `runtime = remote`, remote runtime state is authoritative. Do not substitute local files or chat memory for remote status.

Local Python commands for `MLIP-Autoresearch` helper checks must use exactly:
- `local_python` from `config.json`, or
- `source <conda_sh> && conda activate <conda_env> && python ...`

Do not use bare `python3`, `/usr/bin/python3`, system Python, or unspecified local environments for `MLIP-Autoresearch` helper checks.

This does not override the `MLIP-Evidence` Evidence Execution Environment. Evidence gathering follows the `MLIP-Evidence` skill.

## 3. One-Tick Rule

Each main-session polling tick must:

1. run `scripts/remote_inspect.py --json`
2. treat `inspection.next_safe_step` as the only authoritative next action
3. if `inspection.status_mismatch == true`, reconcile first and stop
4. otherwise execute exactly one safe step
5. stop after that step

If `next_safe_step.kind` is `idle`, `wait`, `running`, or missing, stop. Do not guess.

The main agent must not:
- repeatedly spawn the same subagent
- relaunch or rematerialize units without an inspection step requiring it
- treat a cron tick as a new workflow from scratch
- use chat memory, compaction summaries, or latest-file guesses instead of `round_state.json`

## 4. Concrete Next-Step Kinds

When inspection returns a concrete kind, do the matching action and stop:

- `reconcile_status`: run bundled reconcile script only.
- `summarize_generation_memory`: run `scripts/summarize_generation_outcomes.py --generation <GENERATION> --sync-remote`.
- `decide_continuation`: run `scripts/decide_continuation.py`.
- `activate_evidence`: reuse the returned brief via `scripts/transition_round_state.py --event evidence_complete ...`.
- `evidence`: spawn exactly one `MLIP-Evidence` subagent and explicitly instruct it to use the `MLIP-Evidence` skill.
- `prepare_proposal_context`: run `scripts/prepare_proposal_context.py`.
- `proposal_writing`: spawn exactly one proposal subagent.
- `select_proposals`: write/select `selection.json`, then call `transition_round_state.py --event selection_ready`.
- `materialize_generation`: materialize selected units only, then stop.
- `materialization_ready`: call `transition_round_state.py --event materialization_ready --generation <GENERATION> --source-unit <SOURCE_UNIT> --sync-remote`.
- `generation_active`: call `transition_round_state.py --event generation_active --generation <GENERATION> --source-unit <SOURCE_UNIT> --sync-remote`.
- `implementation`: spawn exactly one implementation subagent for the target unit.
- `sync`: run `scripts/remote_sync_unit.py` for the target unit.
- `smoke`: run `scripts/remote_smoke_unit.py` for the target unit.
- `launch_batch`: run `scripts/remote_launch_batch.py` once for the generation.
- `collect`: run `scripts/remote_collect_unit.py` for the target unit.
- `timeout_stop`: run `scripts/remote_stop_unit.py` for the target unit.
- `repair`: spawn exactly one repair subagent for the target unit.
- `abandon`: mark only the target unit terminal-abandoned, then stop.

If a writer subagent is active or inspection reports `wait_for_writer`, do not spawn another writer. Before spawning an implementation/repair writer, set `round_state.active_writer_unit` when possible and sync state. Clear it after completion is handled.

## 5. Main Agent Role

The main agent is the orchestrator and sole owner of global round state.

The main agent must:
- inspect active runtime state through `remote_inspect.py --json`
- choose continuation sources
- decide whether evidence, proposals, repair, or abandonment are needed
- spawn subagents only for bounded work
- own global artifacts such as `round_state.json`, `selection.json`, frontier records, and milestone records
- invoke bundled scripts for mechanical work
- decide when a round is terminal and whether the next round may begin

The main agent should not do long-form evidence reading, long-form proposal writing, or runnable-unit code rewrites by default when a suitable subagent exists.

Normal round-level workflow transitions must go through `scripts/transition_round_state.py`. Do not manually edit `ledger/round_state.json` for normal evidence, proposal, selection, materialization, or generation activation transitions.

Do not handwrite normal remote orchestration commands such as `scp`, `ssh`, `nohup`, `kill`, manual remote config rewrites, or manual remote state rewrites. Use bundled scripts / remote bridge utilities.

## 6. Subagent Roles

Subagents are bounded workers. They must not independently advance the round.

### `MLIP-Evidence` Subagent

Purpose:
- use the `MLIP-Evidence` skill
- profile the current source unit
- read local PDFs, papers/arXiv, and repositories
- build one bounded evidence package / brief
- compare external mechanisms to the current runnable unit

Must not:
- edit runnable-unit code
- materialize runnable units
- choose the next round
- write `selection.json`
- launch runs

If this subagent runs local evidence helper scripts, it must follow the `MLIP-Evidence` Evidence Execution Environment, not the `MLIP-Autoresearch` local execution shell.

### Proposal Subagent

Purpose:
- read prepared proposal context and the active evidence package
- write proposal files under `research_runtime/proposals/<source_unit>/`
- follow `references/proposal_format.md`

Must not:
- materialize runnable units
- edit runnable-unit code
- launch runs
- decide final selection
- write frontier / milestone records

### Implementation / Repair Subagent

Purpose:
- edit exactly one target runnable unit
- start by reading `research_context/`
- modify only `model/model.py` and `model/train.py`
- implement the proposal handoff as faithfully as possible
- call `mark_unit_implemented.py` after code edits
- optionally run `local_sanity_unit.py` for advisory checks

Required first reads:
- `research_context/manifest.json`
- `research_context/proposal.md`
- `research_context/context.md`
- `research_context/evidence_package/mechanism_cards.json` if present
- `research_context/evidence_package/patch_blueprints.json` if present
- `research_context/evidence_package/proposal_constraints.json` if present
- `research_context/config_to_code_migration.json` if present

If `research_context/` or its required handoff files are missing, stop and report the missing handoff. Do not invent an implementation from chat memory or from model files alone.

Must not:
- edit `config.json` for MLIP-quality changes
- edit another runnable unit
- write `selection.json`
- create generation containers
- launch full benchmark batches
- decide the next round

### Summary Subagent

Purpose:
- read completed outputs
- write one bounded round summary or diagnosis note

Must not:
- advance the workflow
- change global control artifacts
- edit runnable units

## 7. Subagent Completion Handoff

Subagent completion is a workflow transition, not just a chat message.

For `MLIP-Evidence` completion:
- call `scripts/transition_round_state.py --event evidence_complete --brief <BRIEF> --source-unit <SOURCE_UNIT> --target-generation <TARGET_GENERATION> --sync-remote`
- stop

For proposal completion:
- call `scripts/transition_round_state.py --event proposals_ready --proposal-dir <PROPOSAL_DIR> --proposal-source <SOURCE_OR_SESSION> --source-unit <SOURCE_UNIT> --target-generation <TARGET_GENERATION> --sync-remote`
- stop

For implementation/repair completion:
- verify the unit was marked by bundled script
- clear `active_writer_unit`
- sync state if changed
- stop

If `remote_inspect.py` returns `activate_evidence`, reuse the returned brief through the same `evidence_complete` transition and stop.

## 8. Same-Workspace Safety

Default OpenClaw setup uses one agent workspace.

Safety rules:
- only one writer subagent should be active at a time by default
- the main agent must not edit runnable-unit code while a writer subagent is active
- proposal writing and runnable-unit code editing must never target the same path concurrently
- `MLIP-Evidence` is read-heavy and may run concurrently with at most one writer subagent
- the main agent remains the only owner of global control files

Every spawned subagent task must explicitly state:
- which skill to use, if any
- role
- target artifact
- write boundary
- what it must not do
- applicable execution environment for local helpers

Do not rely on the main-session bootstrap alone to communicate subagent behavior.

## 9. Evolution Loop

For each generation round:
- write 10 proposals
- include family / phase metadata in every proposal
- include at least 3 exploit, 3 jump, 1 backward-simplify, 1 control, and 2 wildcard proposals
- select 8 proposals
- ensure selected set contains at least 2 exploit, 2 jump, and 1 control proposal
- materialize selected proposals as separate runnable units
- do not treat materialization as launch-ready
- start a new round only after all selected runnable units reach terminal states
- stop if all selected runnable units are terminal non-successes

Evolution phases:
- Phase 0: contract / smoke
- Phase 1: invariant local pair
- Phase 2: invariant message passing
- Phase 3: angular / body-order hybrid
- Phase 4: lightweight equivariant
- Phase 5: full equivariant / higher-order
- Phase 6: beyond

Each proposal must declare:
- family
- phase
- jump_type
- budget_class
- expected_capability_gain

## 10. Search Policies

### Continuation Tree

Continuation selection uses a PUCT-like tree policy when enabled in `config.json`.

The tree policy must:
- keep benchmark semantics and Q formulas fixed
- treat `best_known_unit` separately from `next_to_expand_unit`
- rebuild lineage statistics from completed runtime history
- preserve hard gates for terminal success, controls, near duplicates, train-only non-breakthroughs, and missing handoff files
- use `Q_total` rank as the exploitation term
- use exact unit expansion count as the exploration denominator
- use mechanism role, family novelty, historical family `G_delta`, repeated concrete-pattern detection, and negative-pattern matches only as bounded priors or penalties
- avoid penalizing broad families merely because related attempts existed

`G_delta` is an edge signal for lineage and family learning. It must not replace `Q_total` as the main objective.

`scripts/decide_continuation.py` must write auditable candidate scores into the continuation decision.

### Capacity Scaling

Capacity scaling is a valid MLIP evolution axis, but not a replacement for mechanism search.

Legitimate code-level axes:
- model width / hidden dimension
- message-passing depth
- radial / angular basis resolution
- body, tensor-product, or global-context branch rank

Scaling proposals must:
- edit `model/model.py` and/or `model/train.py`, not `config.json`
- be bounded and falsifiable
- state whether they test capacity, depth, or resolution bottlenecks
- state compute/runtime/OOM risk
- distinguish pure scaling from new mechanism proposals

### Training Objective

Training-objective evolution is a valid MLIP evolution axis.

Legitimate code-level axes:
- energy / force / gap loss weights when present
- fixed or scheduled loss weighting
- optimizer and scheduler choices
- bounded training-budget constants

Training-objective proposals must:
- edit `model/train.py` and/or code-level defaults, not `config.json`
- preserve benchmark evaluation semantics, split semantics, and metric definitions
- state the targeted failure mode, such as force noise, energy drift, gap instability, undertraining, or optimizer instability
- distinguish objective/training changes from architecture or mechanism changes

For selection:
- include capacity/objective changes when supported by benchmark history or evidence
- avoid selecting a batch dominated by pure scaling or pure objective tuning
- prefer at most one pure scaling proposal per round unless the main agent explicitly decides current bottleneck is capacity
- preserve mechanism coverage, control replicate, and benchmark semantics

## 11. Evidence And Proposal Handoff

Before proposal writing for every new round:
- invoke a `MLIP-Evidence` subagent and explicitly state: use the `MLIP-Evidence` skill
- request `exploit`, `jump`, or `balanced` mode
- require a compact brief plus structured provenance, mechanism cards, patch blueprints, and proposal constraints
- record the returned brief through `transition_round_state.py --event evidence_complete`
- prepare proposal context through `prepare_proposal_context.py`

The evidence brief is advisory. It must not make the round decision automatically.

Proposal context and evidence local context must be benchmark-centric, not force-only. When summarizing a source unit, frontier unit, or generation, include:
- split-level energy metrics
- split-level force metrics
- `mixed_force_mae`
- `mixed_energy_mae`
- `gap_penalty`
- `Q_dataset`
- `Q_rmd17`
- `Q_iso17`
- `Q_total`
- `G_delta` when parent exists
- train-history trend summary
- runtime / timeout / failure information
- control replicate comparison if present

Do not collapse round comparison to force-only ranking.

## 12. Adaptation Profiles

Adaptation profiles are external diagnostics for architecture/train-code transfer. They do not redefine the fixed RMD17/ISO17 benchmark and do not change `Q_total` unless a future policy explicitly says so.

Current profile:
- `mad10k`: deterministic stratified MAD train subset plus full MAD val/test.
- Prepare it with `scripts/prepare_mad10k_profile.py`.
- Run it with `scripts/run_adaptation_profile.py --unit <UNIT>`.

MAD-10k semantics:
- train: 10,000 stratified frames sampled from MAD `mad-train.xyz`
- val: full MAD `mad-val.xyz`
- test: full MAD `mad-test.xyz`
- primary profile metrics: test force MAE, test energy MAE per atom, failure rate, and subset breakdown
- stress is recorded as unsupported unless a runnable unit explicitly implements stress support

Use adaptation profiles as proposal/evidence context for broad transfer capability. Do not treat a missing or failed adaptation profile as a full benchmark failure.

`prepare_proposal_context.py` must consume the active evidence brief recorded in `round_state`; it must not choose a brief by modification time, filename guess, or chat history.

Proposal files must include:
- `mechanism_refs`
- `evidence_refs`
- `files_to_edit`
- `code_insertion_points`
- `minimal_edit_plan`
- `implementation_checklist`

`transition_round_state.py --event proposals_ready` validates these terms before selection.

`create_unit.py` must copy proposal, proposal context, active evidence brief, structured evidence package files, and selection row into each runnable unit under `research_context/`.

`mark_unit_implemented.py` writes `research_context/implementation_report.json` with `model/model.py` and `model/train.py` diffs against the source unit.

## 13. Implementation And Run Gates

Materialization does not mean launch-ready.

Every selected runnable unit must pass:
- `implementation_needed`
- `implemented`
- `remote_smoke_pending`
- `launch_ready`

Do not launch a full benchmark run unless:
- `implementation_status.json` exists
- `implementation_state == "launch_ready"`
- expected changed files are present
- active remote smoke passed
- authoritative remote runtime has synchronized status files and target unit contents

Control replicas may skip proposal-specific code edits, but must still write `implementation_status.json`, pass remote smoke, and become `launch_ready`.

Failed units may be repaired only up to a bounded retry count. Repeated identical failure classes or repair without real state change must trigger abandonment so the generation can continue.

## 14. GPU And Remote Launch

- check `nvidia-smi` before launch
- available GPUs are GPUs with no active compute processes
- batch launch must use explicit validated numeric GPU ids for `CUDA_VISIBLE_DEVICES`
- if auto GPU discovery cannot find available numeric GPU ids, fail instead of launching
- launch logs must record effective `CUDA_VISIBLE_DEVICES`
- verify active CUDA occupancy after launch
- distribute runnable units across available GPUs when possible
- for `launch_batch`, call `scripts/remote_launch_batch.py` once; do not hand-roll loops over `remote_launch_unit.py`

## 15. Milestones And Memory

After a successful full benchmark:
- compute active benchmark Q fields according to `benchmark/anchors.json.active_q_version`
- always preserve `benchmark_version` beside `Q_total`
- compute `G_delta = Q_total_child - Q_total_parent`
- append a frontier record into `research_runtime/ledger/frontier.jsonl`

After all selected units in a generation reach terminal states:
- run `scripts/summarize_generation_outcomes.py --generation <GENERATION> --sync-remote`
- write outcome memory before choosing the next continuation source
- use canonical memory under `ledger/generation_summaries/`, `ledger/generation_reports/`, `ledger/all_attempts.jsonl`, `ledger/tree_state.json`, and `ledger/lineage_stats.json`
- classify non-winning attempts as tradeoffs, partial positives, negative methods, failures, or controls

## 16. Bundled Scripts

Required script layer:
- `scripts/runtime_common.py`
- `scripts/set_round_state.py`
- `scripts/transition_round_state.py`
- `scripts/reconcile_remote_state.py`
- `scripts/create_unit.py`
- `scripts/mark_unit_implemented.py`
- `scripts/local_sanity_unit.py`
- `scripts/remote_materialize.py`
- `scripts/remote_sync_unit.py`
- `scripts/remote_smoke_unit.py`
- `scripts/remote_launch_unit.py`
- `scripts/remote_launch_batch.py`
- `scripts/remote_collect_unit.py`
- `scripts/remote_stop_unit.py`
- `scripts/remote_inspect.py`
- `scripts/rebuild_tree_state.py`
- `scripts/decide_continuation.py`
- `scripts/prepare_mad10k_profile.py`
- `scripts/run_adaptation_profile.py`
- `scripts/remote_run_adaptation_profile.py`
- `scripts/prepare_proposal_context.py`
- `scripts/summarize_generation_outcomes.py`
- `scripts/compute_milestone_g.py`

## Main Idea

The purpose of this skill is to make the loop behave like MLIP evolution while remaining executable:
- exploit strong branches
- measure variance with controls
- detect stagnation
- jump phases when needed
- enforce implementation and smoke gates
- keep evidence, proposal, and implementation handoff aligned
- maintain searchable long-term memory
- keep remote state authoritative and mirrored consistently
