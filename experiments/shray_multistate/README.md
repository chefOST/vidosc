# Shray: text-conditioned multi-state VidOSC experiment

Experimental contribution to ChefOST's VidOSC work. This folder contains only
Shray's experiment code and saved evidence. Shared inference, descriptions,
MOSCATO modules, and upstream VidOSC source are unchanged.

## What it contributes

Video features -> pretrained VidOSC temporal encoder -> object-conditioned text
scoring head -> overlapping state intervals -> independent evaluation.

- Candidate names/descriptions are supplied at inference; candidate count can
  change without changing the output layer or retraining.
- An object-name query conditions the scores. This is full-frame analysis, with
  no object crops or tracker; VidOSC's object-feature input is zero-filled.
- A frozen SlowFast feature-fusion comparison and optional fine-tuning of an own
  VidOSC encoder copy are included.
- Subject-disjoint protocols, constant-prior baselines, held-out-state checks,
  frame alignment, and multi-label/interval metrics are included.

This does not automatically discover or name unknown states. The transformer
uses the whole video; this is offline analysis rather than causal streaming.

## Status and measured results

The software interface works; the recognition accuracy goal is not achieved.
On five fresh videos / seven object-video units, the validation-selected model
has 30.3% mean unit multi-label F1 and 42.6% highest-label agreement. The stronger
training-frequency baseline has 38.1% F1 and 48.0% highest-label agreement.
All learned variants score below that baseline on overall F1. Held-out peeled
state F1 is zero. SlowFast did not improve this fresh test. These are custom
CMU MOSCATO pilot results, not an official benchmark or a claim of improvement.

See [architecture](broader/ARCHITECTURE.md), [full report](broader/REPORT.md),
[protocol and experiment details](broader/README.md), and
[saved comparisons](broader/results/comparison.json).

## Folder map

| Location | Purpose |
| --- | --- |
| `temporal_states.py` | Shared text-scoring head, smoothing, interval utilities |
| `scoring.py` | Label alignment and legacy pilot metrics |
| `prepare.py` | Dependency loading and legacy preparation utilities |
| `broader/model.py`, `fit.py` | Object-conditioned model and training |
| `broader/features.py` | CLIP, SlowFast and pending InternVideo adapters |
| `broader/predict_custom.py` | Inference with supplied descriptions |
| `broader/protocol*.json`, `metrics.py`, `evaluate_broader.py` | Locked splits and evaluation |
| `broader/results/` | Original JSON results, selections and provenance; no weights |
| `test_temporal_states.py`, `broader/test_broader.py` | Existing 9 + 14 functional/metric tests |

## Local use

This copy resolves a dependency workspace via `SHRAY_WORKSPACE`, or by finding
an ancestor with `work/VidOSC/model.py`. Dependencies and large artifacts remain
outside this repository. In the current workspace, from the repository root:

```powershell
$env:SHRAY_WORKSPACE = (Resolve-Path '../..').Path
$python = Join-Path $env:SHRAY_WORKSPACE 'work/vidosc-env/Scripts/python.exe'
$experiment = 'experiments/shray_multistate'
& $python "$experiment/test_temporal_states.py"
& $python "$experiment/broader/test_broader.py"

# Reuse a saved model without copying weights into Git.
$checkpoint = Join-Path $env:SHRAY_WORKSPACE 'outputs/shray-multistate/broader/results/clip-manual/model.pt'
& $python "$experiment/broader/predict_custom.py" --checkpoint $checkpoint --video-id S15_Salad_7150991-451 --object vegetables --candidates "$experiment/broader/candidate-states-cached-example.json" --output "$experiment/broader/local-predictions/salad.json"
```

On another machine, point `SHRAY_WORKSPACE` at your dependency workspace. The
current scripts expect `work/VidOSC` with the official chopping checkpoint,
`work/chefost-data/MOSCATO`, the `work/shray-broader` features/text cache, and
`work/model-cache`. Feature extraction currently also expects the workspace's
Windows FFmpeg distribution under `work/ffmpeg`. DirectML is a Windows option;
the Torch extraction backend is available. `broader/requirements-lock.txt`
records the original run environment; it is not a cross-platform installer.

The official InternVideo adapter has not been verified with its gated checkpoint.
No InternVideo accuracy results are claimed.

## Preserved evidence and future runs

JSON evidence is copied byte-for-byte, including original paths and hashes.
`ORIGIN.json` records the source files and any packaging changes. Weights,
predictions in tensor form, videos, and feature caches stay in the original
workspace. Do not rerun the training runner into these saved result folders;
its existing-run protection intentionally rejects that. A clean experimental
copy with fresh held-out recordings is needed for the next accuracy experiment.

The report's verification statements describe the original run.
`folder-verification.json` records checks performed on this relocated copy before
publication; its commit/push flags describe that packaging stage. Publication is
on Shray's own branch, with the experiment contained in this folder.

## Integration point

This is an optional experimental state predictor alongside the existing pipeline.
It does not alter shared `run_inference.py`, `describe_states.py`, or `moscato/`.
Any future integration should adapt its frame-indexed multi-label scores and
half-open intervals to the agreed evaluator, preserving unknown-label masking.
The current own metrics should not be assumed to match the shared evaluator's
protocol without a parity check.
