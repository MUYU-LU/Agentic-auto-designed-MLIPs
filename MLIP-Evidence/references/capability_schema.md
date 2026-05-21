# Capability schema

Use this controlled vocabulary when profiling current code, papers, or repos.

## Core capabilities
- invariant-scalar
- relative-geometry
- cutoff-locality
- neighbor-list
- pair-only
- message-passing
- angular-triplets
- body-order
- vector-features
- equivariance
- e3nn
- force-from-autograd
- stress
- atomref
- long-range-head
- batching
- config-system
- benchmark-adapter
- graph-batching
- loss-weighting
- energy-decomposition
- composition-baseline
- molecular-generalization
- materials-generalization

## Current profile fields
- representation
- symmetry
- geometry_handling
- energy_force_pathway
- training_regime
- known_failure_modes
- current_phase
- current_family
- current_bottleneck

## Fit rules
High fit:
- adds <= 2 capabilities
- no e3nn
- no full runtime rewrite
- bounded edits fit inside current `model/model.py` and `model/train.py`

Medium fit:
- adds 3-4 capabilities
- may require neighbor-list or triplets
- may need helper utilities or moderate batching changes

Low fit:
- requires e3nn / full equivariance
- requires incompatible runtime or output contract
- requires framework-scale rewrite or a new dependency stack

## Evidence layers
Every dossier must explicitly cover:
- mathematics
- physics
- chemistry
- textual evidence
- code evidence
