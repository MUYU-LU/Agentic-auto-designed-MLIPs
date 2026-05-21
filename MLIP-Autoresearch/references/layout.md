# Layout

## Skill vs runtime

The skill and the runtime are separate.

### Skill
The skill lives under `MLIP-Autoresearch/` and contains:
- `SKILL.md`
- `config.json`
- `scripts/`
- `references/`

### Runtime
The runtime contains experiment state:
- `benchmark/`
- `base_unit/`
- `generations/`
- `proposals/`
- `knowledge/`
- `ledger/`

## Execution locations

### Local staging runtime
This is the local editable mirror used by the main session and subagents for:
- proposal files
- local runnable-unit code
- local status mirrors
- local logs
- orchestration scripts

### Local execution shell
This is the fixed local shell / Python environment used by `MLIP-Autoresearch` helper scripts and advisory sanity checks.

It is separate from the `MLIP-Evidence` Evidence Execution Environment used for local PDF and repository reading.

### Active remote runtime
This is the authoritative experiment runtime when `runtime == remote`.

## Runtime roots

The active runtime root is resolved by scripts.
They must not assume a fragile relative path from the skill directory.
Robust resolution should prefer:
1. explicit environment override
2. existing candidate runtime roots near the skill root
3. configured relative fallback

The local staging runtime may be a mirror of the active remote runtime, but it is not authoritative by itself.

## benchmark/
`benchmark/` is fixed.
Runnable units must reference it but must not embed their own benchmark copy.

Optional adaptation-profile data lives under:

```text
benchmark/profiles/<profile_name>/
  split_manifest.json
  train/
  val/
  test/
```

Profile data is diagnostic context, not part of the main RMD17/ISO17 `Q_total` benchmark.

## base_unit/
`base_unit/` is the executable seed unit.
It is not a formal generation round.

Recommended contents:
- `config.json`
- `main.py`
- `model/`
- `outputs/`

## generations/
`generations/` contains formal generation rounds.
A generation round is a container, not a runnable unit.

Each `proposal_xxx/` runnable unit should contain:
- `config.json`
- `main.py`
- `model/`
- `outputs/`
- `unit_meta.json`
- `implementation_status.json`
- `run_status.json`

## proposals/
`proposals/` contains proposal artifacts and proposal context files.
They are not runnable units.

## knowledge/
`knowledge/` stores evidence memory:
- `briefs/`
- `notes/`
- optional paper / repo cards

## ledger/
`ledger/` stores:
- `frontier.jsonl`
- `milestones/`
- `round_state.json`

When `runtime == remote`, authoritative status must exist remotely as well, not only in local mirrors.
