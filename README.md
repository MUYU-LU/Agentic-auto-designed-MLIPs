# AutoResearch-MLIP

<p align="center">
  <b>Skill-bounded autonomous design evolution for machine-learned interatomic potentials</b>
</p>

<p align="center">
  <a href="https://github.com/MUYU-LU/Agentic-auto-designed-MLIPs">
    <img src="https://img.shields.io/badge/repository-public-blue" alt="public repository">
  </a>
  <a href="https://huggingface.co/datasets/DeerEyeRain/research_runtime">
    <img src="https://img.shields.io/badge/runtime-Hugging%20Face-yellow" alt="runtime data on Hugging Face">
  </a>
  <img src="https://img.shields.io/badge/domain-MLIP%20autoresearch-green" alt="MLIP autoresearch">
  <img src="https://img.shields.io/badge/contents-skill%20contracts-lightgrey" alt="skill contracts">
</p>

AutoResearch-MLIP studies whether autonomous agents can perform auditable research on
machine-learned interatomic potentials (MLIPs), rather than only generating isolated
candidate architectures.

This repository contains the two public skill bundles used to organize that process:

| Skill | Role |
| --- | --- |
| `MLIP-Autoresearch` | Bounds the research loop: candidate creation, implementation gates, evaluator interfaces, repair limits, continuation decisions and runtime records. |
| `MLIP-Evidence` | Builds evidence packages from papers, repositories and code audits before proposals become executable MLIP trials. |

Large runtime artifacts are hosted separately on Hugging Face:

**Runtime dataset:** <https://huggingface.co/datasets/DeerEyeRain/research_runtime>

---

## What this repository contains

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

### MLIP-Autoresearch

`MLIP-Autoresearch` is the outer-loop research skill. It defines what the
agent may do during MLIP design evolution and what remains outside agent
control.

It covers:

- generation-level trial orchestration;
- allowed source-code edit surfaces;
- smoke tests, remote launch, collection and repair gates;
- fixed evaluator interfaces and assay semantics;
- terminal trial records, failure-memory records and continuation records;
- non-greedy continuation decisions that can preserve a best observed
  candidate while expanding a diagnostically useful branch.

The public `config.json` is a template with placeholders such as
`<REMOTE_HOST>`, `<REMOTE_USER>`, `<REMOTE_WORKDIR>` and
`<REMOTE_RUNTIME_ROOT>`. Replace these in a private local configuration
before execution. Do not commit credentials or machine-specific paths.

### MLIP-Evidence

`MLIP-Evidence` is the evidence skill. It prepares proposal-time research
context from:

- local or web-accessible MLIP papers;
- repository discovery and source-code reading;
- mechanism cards;
- patch blueprints;
- implementation-fit checks;
- compact evidence briefs for proposal writing.

This skill is evidence-only. It does not edit runnable MLIP units, launch
benchmarks, choose continuation sources or redefine evaluator metrics.

---

## Runtime artifacts

The full research runtime is not stored in this GitHub repository because it
contains large logs, source snapshots and generation-level records. The
runtime artifact bundle is available here:

<https://huggingface.co/datasets/DeerEyeRain/research_runtime>

It includes, depending on release state:

- evaluated trial records;
- generation reports;
- continuation decisions;
- failure-memory and partial-positive records;
- source snapshots or diffs;
- assay summaries;
- figure and manuscript-support tables.

The Hugging Face dataset is an artifact store rather than a simple tabular
dataset. Some records are JSON, JSONL, source snapshots or runtime logs, so
the dataset viewer may not render every file as a table.

---

## Quick start

Clone the repository:

```bash
git clone https://github.com/MUYU-LU/Agentic-auto-designed-MLIPs.git
cd Agentic-auto-designed-MLIPs
```

Copy the skills into your agent environment, or point your agent runtime to
the two skill folders directly.

Create a private runtime configuration from `MLIP-Autoresearch/config.json`.
At minimum, replace placeholder values for local workspace paths, remote
execution paths and runtime artifact locations.

Example environment variables:

```bash
export OPENCLAW_WORKSPACE=/path/to/workspace
export MLIP_RUNTIME_ROOT=/path/to/research_runtime
export MLIP_MAD10K_RAW_DIR=/path/to/mad10k/raw
export MLIP_EVIDENCE_REPO_CACHE=/path/to/repo/cache
```

Typical workflow:

1. Use `MLIP-Evidence` to create an evidence package for the current source.
2. Use `MLIP-Autoresearch` to materialize bounded MLIP candidate edits.
3. Run smoke tests and fixed assays through the configured evaluator.
4. Collect terminal records and failure-memory records.
5. Apply the continuation policy to choose the next research source.

---

## Release and privacy notes

The public skill bundles intentionally use placeholders for private paths,
remote hosts, usernames, work directories and credentials.

Before publishing derived versions, check that the repository does not
contain:

- passwords, API keys or tokens;
- SSH keys or private cluster credentials;
- personal machine paths;
- private runtime roots;
- compiled Python caches such as `__pycache__/` or `.pyc` files;
- redistributed third-party datasets or source caches without permission.

The large `research_runtime` artifact bundle should also be reviewed before
public release if it contains local paths, third-party source mirrors or
private execution metadata.

---

## Citation

If you use these skill bundles or runtime artifacts, please cite the
associated AutoResearch-MLIP manuscript when available.

---

## License

No license file has been added yet. Until a license is specified, reuse is
restricted by default copyright terms.
