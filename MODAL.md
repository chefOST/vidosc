# Modal detector smoke test

This is the next setup check for the MOSCATO baseline. It runs the pretrained
100DOH hand/object detector on **five frames**, not the entire clip. It does not
produce VidOSC state predictions or evaluation scores.

## Run from the repository root

```bash
.venv/bin/python -m pip install -r requirements-modal.txt
.venv/bin/python -m modal setup
.venv/bin/python -m modal run scripts/modal_detector.py
```

Authentication stays in Modal's local configuration, outside the repository.
The local video must exist at `data/moscato/videos/S29_Salad_7150991-1.avi`.
The script sequentially decodes frames 1469, 2800, 3000, 3700, and 4600 and sends
only those five PNGs plus a video hash to Modal. No annotations are uploaded.

## Remote environment

The script defines a Python 3.11 Linux image with pinned primary dependencies,
PyTorch 2.6.0 and torchvision 0.21.0. It downloads the official detector source
at commit `e6eec712a498ec7844b97893c8d012cea1a71e09` and the published
`handobj_100K` checkpoint (Google Drive ID `166IM6CXA32f9L6V7-EMd9m8gin6TFpim`).
The original Drive link is unavailable. The script downloads a copy from
[Imperial College London's repository](https://github.com/ImperialCollegeLondon/MessagePassingFramework/blob/825297bb9c8a20426ade07d3ab5b3310ed0a6210/tas_perception/hand_object/models/faster_rcnn_1_8_89999.pth),
pinned to that commit. This mirror was linked in [upstream issue 38](https://github.com/ddshan/hand_object_detector/issues/38).
The download is checked against the pinned repository's Git LFS size and SHA256.
The run manifest also records that SHA256; no author-published checksum was
available to independently compare the mirror with the deleted Drive file.
The checkpoint is loaded with `weights_only=True` and strict model key matching.
Downloads occur during the cached image build, before GPU inference.

Inference defaults to two remote CPU cores, a five-minute function timeout,
zero retries, and at most one container. This consumes Modal compute credits.
GPU use was blocked because the account needs a payment method. Once billing is
configured, explicitly request T4 with:

```bash
VIDOSC_DETECTOR_GPU=T4 .venv/bin/python -m modal run scripts/modal_detector.py
```

The default CPU run does not request any GPU resources. It is a one-off
`modal run`, not a deployed service. Results are returned to the local caller.

## Compatibility and interpretation

`moscato/hand_object.py` uses the upstream ResNet-101 detector architecture,
preprocessing, and box transforms. It replaces the old compiled ROI operations
with torchvision ROIAlign (`aligned=False`) and inclusive-coordinate NMS
(the original implementation adds one to widths/heights). A regression test
checks this coordinate convention. Full numerical equivalence to the original
compiled extension has **not** been demonstrated.

The threshold is fixed at 0.5 for both classes. Green boxes are candidate target
objects; cyan boxes are hands. Contact predictions are included in JSON but
this smoke test does not associate hands with objects or choose feature crops.
Objects are not named (carrot versus knife), and detections are not tracks.
The exact detector variant and crop-selection policy used for VidOSC's released
features remain unverified. Do not claim feature-extraction parity.

## Outputs

Each successful run creates a new UTC-timestamped folder in
`runs/modal-detector/`, containing five annotated JPEGs and `detections.json`.
The JSON records boxes, confidence scores, frame indices, file hashes, source
version, GPU, and compatibility limitations. Outputs are ignored by Git.

Inspect these images before implementing crop selection and processing all 107
seconds. Complete label mapping and annotation-clock verification before scoring.

## Verified run — October 11, 2026

Run `ap-4f4or7OXEiA4uO9oMhOoFq` completed on Modal CPU in approximately
27 seconds of detector loading and inference (excluding build/startup/upload).
Local outputs: `runs/modal-detector/20261011T183048692769Z/`.
Checkpoint SHA256: `e7f57309da0536d4bb6c4cdf15ab53256790cf4486c6eef59cb8698a9c33bc43`.

| Zero-based frame | Object boxes | Hand boxes |
| --- | ---: | ---: |
| 1469 | 1 | 1 |
| 2800 | 0 | 0 |
| 3000 | 1 | 2 |
| 3700 | 2 | 2 |
| 4600 | 0 | 1 |

Visual review: the first object box covers much of the cutting board; later
boxes include the knife and an object in the hand. Candidate object detection
is not reliable vegetables-only localization. Two samples have no object box.
These are five selected diagnostic frames, not an accuracy benchmark.
Crop-selection and missing-detection policies must be specified before full
feature extraction. No ground-truth-guided box correction was performed.

Validation: 27 local evaluation/extraction/coordinate tests passed; the remote
run verifies strict pretrained loading and real-frame inference. T4 inference
remains untested because Modal requires a payment method on this account.
