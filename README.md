# Multi-state VidOSC experiment

This contributor branch, `shray/feat/multistate-vidosc`, extends pretrained
VidOSC with a trainable object-conditioned text scorer. It predicts named,
overlapping states over time, using supplied candidate names or descriptions.
Candidate count can change at inference without replacing the output layer.
It does not automatically discover or name states.

The pipeline is **video features → VidOSC temporal context → text-conditioned
state scores → overlapping intervals → evaluation**. It includes a SlowFast
feature comparison and optional fine-tuning of a separate VidOSC encoder copy.
Full-frame features are used; object tracking and crops remain unimplemented.

## Status

The [accuracy follow-up](experiments/shray_multistate/v2/README.md) uses broader
human-labelled training and a revised loss. On the same four new recordings,
the validation-selected model scored **53.1% multi-label F1**, versus **35.5%**
for the unchanged old checkpoint and **46.9%** for a constant-prior baseline.
The retrained original architecture scored **56.2%**; the new architecture
changes did not improve test F1. Full label-set accuracy is only **1.5%**.

See [what changed and what the gains mean](experiments/shray_multistate/v2/CHANGES.md)
and the [results, chart and limitations](experiments/shray_multistate/v2/REPORT.md).
All 23 existing tests and four follow-up tests pass. The earlier pilot's
30.3% F1 was on a different test set; its evidence remains preserved. Reliable
recognition and unseen-object/state generalization remain unresolved.

## Run checks

With the external dependency workspace and Python environment described in
[setup](experiments/shray_multistate/README.md#local-use) prepared, run from
this repository root:

```powershell
$env:SHRAY_WORKSPACE = (Resolve-Path '../..').Path
$python = Join-Path $env:SHRAY_WORKSPACE 'work/vidosc-env/Scripts/python.exe'
& $python experiments/shray_multistate/test_temporal_states.py
& $python experiments/shray_multistate/broader/test_broader.py
```

The workspace supplies upstream VidOSC, its checkpoint, MOSCATO data and feature
caches. Large artifacts are excluded from Git.

## Documentation and contribution scope

- [Setup, inference example and folder map](experiments/shray_multistate/README.md)
- [Differences from teammates' approaches](experiments/shray_multistate/METHOD_COMPARISON.md)
- [Architecture](experiments/shray_multistate/broader/ARCHITECTURE.md)
- [Evaluation results and limitations](experiments/shray_multistate/broader/REPORT.md)
- [Latest accuracy changes and direct comparisons](experiments/shray_multistate/v2/CHANGES.md)
- [Latest fresh-recording evaluation](experiments/shray_multistate/v2/REPORT.md)

Implementation is in `experiments/shray_multistate/`.
