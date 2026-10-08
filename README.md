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

Feature extraction, training, inference and evaluation completed for a custom
CMU MOSCATO pilot. All 23 tests passed. On five fresh videos, the
validation-selected model scored **30.3% multi-label F1**, below the stronger
constant-prior baseline's **38.1%**. Reliable recognition is unresolved.

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

Implementation is in `experiments/shray_multistate/`.
