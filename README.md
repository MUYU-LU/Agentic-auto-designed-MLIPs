# AutoResearch-MLIP: Agentic Auto-Designed MLIPs

This repository contains the public skill bundles used by **AutoResearch-MLIP**, a skill-bounded autonomous research protocol for machine-learned interatomic potential (MLIP) design evolution.

AutoResearch-MLIP treats MLIP development as an auditable closed loop: evidence is gathered from papers, repositories, and code audits; candidate designs are implemented as bounded code edits; fixed molecular and periodic-material assays evaluate runnable candidates; and failures, controls, and continuation decisions are recorded as part of the research state.

## Repository structure

```text
.
├── MLIP-Autoresearch/
│   ├── SKILL.md
│   ├── config.json
│   ├── references/
│   └── scripts/
└── MLIP-Evidence/
    ├── SKILL.md
    ├── README.md
    ├── references/
    └── scripts/
```

### `MLIP-Autoresearch`

`MLIP-Autoresearch` is the outer-loop orchestration skill. It defines the runtime boundary for MLIP design evolution, including:

- fixed evaluator and benchmark semantics;
- source selection and continuation policy;
- proposal, implementation, smoke-test, launch, collection, and repair gates;
- terminal trial records, failure-memory records, and continuation records;
- separation between the main orchestrating agent and bounded worker subagents.

The included `config.json` is a public template. Fields such as `<LOCAL_RUNTIME_ROOT>`, `<REMOTE_HOST>`, `<REMOTE_USER>`, `<REMOTE_WORKDIR>`, and `<REMOTE_RUNTIME_ROOT>` must be replaced in a private local configuration before execution. Do not commit credentials or machine-specific paths.

### `MLIP-Evidence`

`MLIP-Evidence` is the evidence-gathering skill. It builds proposal-time evidence packages from:

- local MLIP papers;
- arXiv or web papers;
- repository discovery and code-reading;
- implementation-fit analysis;
- design-evidence cards, patch blueprints, source plans, and compact briefs.

This skill is evidence-only. It does not edit runnable units, launch benchmarks, choose continuation sources, or redefine evaluator semantics.

## Runtime data

The runtime ledger, evaluated trial records, generation reports, continuation records, source snapshots, failure-memory records, and other large audit artifacts are hosted separately on Hugging Face:

**Dataset:** https://huggingface.co/datasets/DeerEyeRain/research_runtime

The Hugging Face dataset is the external artifact location for the research runtime rather than a lightweight tabular dataset. Some files are structured logs, JSON/JSONL records, source snapshots, or runtime artifacts, so the Hugging Face dataset viewer may not render all files as tables.

## How to use

1. Clone this repository.
2. Copy the relevant skill directory into your local agent/skill environment.
3. Create a private runtime configuration from `MLIP-Autoresearch/config.json` by replacing placeholder paths and credentials.
4. Set local environment variables where needed, for example:

```bash
export OPENCLAW_WORKSPACE=/path/to/workspace
export MLIP_RUNTIME_ROOT=/path/to/research_runtime
export MLIP_MAD10K_RAW_DIR=/path/to/mad10k/raw
export MLIP_EVIDENCE_REPO_CACHE=/path/to/repo/cache
```

5. Use `MLIP-Evidence` to build evidence packages and `MLIP-Autoresearch` to orchestrate bounded candidate implementation, evaluation, and continuation.

## Privacy and release notes

The public skill bundles use placeholders for private machine paths, remote hosts, usernames, work directories, and credentials. The repository should not contain:

- passwords or API keys;
- private SSH keys;
- local user paths such as `/home/<user>` or Windows desktop paths;
- machine-specific runtime roots;
- compiled Python caches.

Before redistributing modified versions, re-run a secret/path scan and remove `__pycache__/` directories.

## Citation

If you use this repository, please cite the associated AutoResearch-MLIP manuscript when available.

## License

A license file has not yet been added. Until a license is specified, reuse is restricted by default copyright terms.
