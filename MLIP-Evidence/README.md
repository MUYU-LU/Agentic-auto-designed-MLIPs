# MLIP-Evidence Bundle

This bundle exposes one public evidence skill:

- `skills/MLIP-Evidence/`

`MLIP-Evidence` owns the full evidence workflow:
- current runnable-unit profiling
- local MLIP PDF reading
- paper / arXiv reading
- repository discovery
- repository deep-read
- code-pattern extraction
- implementation-fit judgment
- exploit / jump recommendation
- persistent paper and repo evidence cards

There is no separate public `search-layer` skill.
If helper scripts exist, they are internal implementation details of `MLIP-Evidence`.

## What this skill is for

The goal is to create a dossier that is useful even to a different agent opening it later with no context.
That means every good dossier should contain:
- mathematics
- physics
- chemistry
- textual paper evidence
- code / repo evidence
- current unit profile
- capability gap
- exploit angles
- jump angles
- handoff summary

## Required local-first order

Prefer:
1. current unit code + runtime state
2. local MLIP PDFs
3. previous notes / briefs
4. external papers / arXiv
5. repos and code

## Subagent use

When used as a subagent, the task must explicitly say:
- use the `MLIP-Evidence` skill
- return one bounded dossier
- do not edit runnable units
- do not advance the round

## Output locations

Persistent outputs live under:

## Handoff expectation

The evidence output should be readable by an unrelated agent with no prior chat context.

A good evidence dossier therefore needs:
- mathematical evidence
- physical evidence
- chemical evidence
- textual evidence
- code evidence
- explicit capability-gap analysis
- exploit angles
- jump angles
- implementation-fit judgment

If a dossier only contains titles, links, or vague repository mentions, it is not sufficient for reliable handoff.
