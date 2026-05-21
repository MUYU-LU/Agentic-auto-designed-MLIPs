# Proposal format

## Purpose

A proposal is a structured research hypothesis and implementation handoff.
It is not the benchmark, the runnable unit itself, or the selection record.

A valid proposal must be specific enough that an implementation subagent can edit exactly one runnable unit without reconstructing the idea from chat memory.

## File naming

Keep stable numbered proposal files inside the proposal directory:
- `proposal_001.md`
- `proposal_002.md`
- ...
- `proposal_010.md`

Materialized runnable-unit directories remain:
- `generations/generation_xxx/proposal_001/`
- `generations/generation_xxx/proposal_002/`

## Required metadata

Every proposal must include these top-level metadata bullets near the top:

```markdown
- family: <short_family_name>
- phase: <0-6>
- jump_type: exploit | jump | backward-simplify | control | wildcard
- budget_class: tiny | small | medium | large
- expected_capability_gain: <short concrete expectation>
```

## Required handoff fields

Every proposal must contain these exact section names or exact field labels. `transition_round_state.py --event proposals_ready` checks that these terms exist before selection:

- `mechanism_refs`
- `evidence_refs`
- `files_to_edit`
- `code_insertion_points`
- `minimal_edit_plan`
- `implementation_checklist`

Use `mechanism_refs` from the active evidence package when available. If no strong mechanism card exists, write `mechanism_refs: []` and explicitly mark the proposal as diagnosis-driven or control/simplification rather than external-mechanism-driven.

## Code-level MLIP knobs

Benchmark and runtime semantics are fixed in `config.json`. Do not propose changing `config.json` to improve MLIP quality.

The following are MLIP architecture/training knobs and must be changed in code when a proposal needs them:
- `model/model.py`: `hidden_dim`, `num_interactions`, `num_rbf`, `cutoff`, branch rank/width, global-context direction count
- `model/train.py`: `learning_rate`, `weight_decay`, `energy_weight`, `force_weight`, bounded schedule or training-budget constants

New materialized units may contain a `research_context/config_to_code_migration.json` file documenting the mechanical migration of these values from inherited config into code-level defaults. Treat that file as context, not as an edit target.

## Capacity scaling proposals

Capacity scaling is a valid MLIP evolution axis when it tests a clear bottleneck:
- capacity bottleneck: hidden width or branch width is too small
- depth bottleneck: message-passing depth is too small
- radial / angular resolution bottleneck: basis count or branch rank is too small

Keep scaling proposals bounded. A good scaling proposal changes a small number of code-level defaults, states the compute/runtime risk, and explains what result would show that the previous bottleneck was capacity rather than mechanism.

Do not let capacity scaling replace mechanism coverage. A proposal set should not become pure hyperparameter search.

## Training-objective proposals

Training-objective evolution is a valid MLIP evolution axis. It includes:
- `energy_weight`, `force_weight`, and other loss weights when present
- fixed or scheduled loss weighting
- optimizer and scheduler choices
- bounded training-budget constants

A good training-objective proposal states the targeted failure mode, such as force noise, energy drift, gap instability, undertraining, or optimizer instability. It must preserve benchmark evaluation semantics, split semantics, and metric definitions.

Do not disguise objective/training changes as architecture changes. If the proposal changes only training behavior, say so explicitly.

## Required proposal skeleton

```markdown
# Proposal NNN: <short concrete name>

- family: <short_family_name>
- phase: <0-6>
- jump_type: exploit | jump | backward-simplify | control | wildcard
- budget_class: tiny | small | medium | large
- expected_capability_gain: <one line>

## one_sentence_hypothesis
<One sentence linking the code edit to the expected benchmark effect.>

## mechanism_refs
- <mechanism_id from mechanism_cards.json, or [] if this is control / diagnosis-only>

## evidence_refs
- <evidence source id, paper/repo/source id, or benchmark memory path>
- <cite evidence_quality/proposal_constraints if relevant>

## historical_relation
- source_unit: <generation_xxx/proposal_yyy>
- relation_to_source: exploit | jump | simplify | control | ablation
- not_a_duplicate_of: <recent failed/successful units and why>
- lesson_used: <negative or partial-positive pattern from outcome memory>

## benchmark_rationale
Discuss expected effects on all relevant benchmark signals, not force-only:
- capacity/scaling hypothesis, if any:
- rmd17 energy:
- rmd17 force:
- rmd17 gap / Q:
- iso17 energy:
- iso17 force:
- iso17 gap / Q:
- training stability / runtime risk:
- control comparison expectation:

## files_to_edit
- `model/model.py`
- `model/train.py` if needed, otherwise `none`
- never `config.json` for MLIP-quality changes

## code_insertion_points
- `model/model.py::<ClassOrFunction>`: <what to change>
- `model/train.py::<ClassOrFunction>`: <what to change or none>
- code-level MLIP knobs may include `EvolutionMLIP.__init__` defaults, `MODEL_*` constants, or `TRAIN_*` constants when present

## minimal_edit_plan
1. <smallest edit step>
2. <smallest edit step>
3. <smallest edit step>

## implementation_checklist
- [ ] Preserve `E = sum_i E_i` and force-from-energy contract.
- [ ] Preserve benchmark metric field names and runnable entrypoint contract.
- [ ] Modify only the allowed target unit files.
- [ ] Do not edit `config.json`; change MLIP knobs in `model/model.py` or `model/train.py`.
- [ ] Keep tensor shapes compatible with current dataloader and model forward.
- [ ] Add no unbounded cubic neighbor/triplet loops unless explicitly justified by budget.
- [ ] Call `mark_unit_implemented.py --unit <UNIT> --actor implementation_subagent` after edits.

## expected_benchmark_effect
- primary expected gain:
- expected tradeoff:
- failure signal that would falsify this proposal:

## ablation_or_control
- required control or comparison:
- optional zero-gate / source-fallback / readout-only ablation:

## implementation_notes_for_subagent
<Concrete instructions that survive materialization into `research_context/proposal.md`. Avoid chat-dependent references.>
```

## What not to write

Do not write proposals that only say:
- “MACE-like”
- “ACE-inspired”
- “better equivariance”
- “improve readout”
- “use attention”
- “follow evidence brief”

Those phrases are allowed only when backed by concrete `mechanism_refs`, `evidence_refs`, insertion points, tensor-shape implications, and a bounded edit plan.

## Control proposal

A control proposal still needs the required handoff fields. Use:

```markdown
## mechanism_refs
- []

## evidence_refs
- current source unit
- round policy control requirement

## files_to_edit
- none

## code_insertion_points
- none

## minimal_edit_plan
1. Exact copy of source unit.

## implementation_checklist
- [ ] Do not change `model/model.py`.
- [ ] Do not change `model/train.py`.
- [ ] Mark implemented as control replicate if required by workflow.
```
