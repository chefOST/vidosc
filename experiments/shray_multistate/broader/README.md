> Relocated experiment copy: see [the parent README](../README.md) for setup.
> Commands below run from the dependency workspace root. Result JSON is preserved,
> but large weights/features/videos stay in that workspace. The saved result folders
> are evidence, not clean destinations for rerunning training.

# Shray's broader multi-state VidOSC experiment

Own module only. The VidOSC source and checkpoint are read as dependencies; teammates' files are not edited. Nothing is pushed to GitHub.

The model scores textual state candidates independently, so a video can have any number of active states and multiple states at once. This experiment supplies the 77 CMU vocabulary names. It does not discover or name previously unknown states automatically.

## Architecture

Frozen CLIP image features (768 dimensions) feed the pretrained VidOSC temporal encoder. The original four-class head is bypassed. Its 512-dimensional temporal context, CLIP visual features, and a 768-dimensional CLIP object-name embedding are concatenated. An own MLP maps this to CLIP text space, then compares against centered candidate text embeddings. Independent sigmoid scores plus a validation-selected threshold produce intervals. The candidate count is a forward-pass input. These scores are not calibrated probabilities.

The motion comparison adds a frozen official SlowFast R50 Kinetics400 embedding (2304 dimensions) from a 64-frame window: 32 fast-path and eight slow-path frames. Both methods fit an own head and optionally fine-tune an own copy of the VidOSC encoder; validation determines whether fine-tuning is retained.

There are no object crops or tracker. An object-name query conditions the scorer on full-frame features. The original VidOSC object-feature slot is filled with zeros. CLIP differs from the original InternVideo backbone; even the pending InternVideo comparison would still lack the original object-feature stream.

## Locked evaluation

`protocol.json` fixes 12 pseudo-labelled training videos, four manually labelled validation videos, and five fresh manually labelled test videos, with subjects separate across splits. Train recipes: salad, eggs, brownies, sandwiches. Pizza, cheese, and batter are test-only. The brownie target is eggs, not brownie batter.

The primary score is mean per-object/video multi-label micro-F1, with equal weight for each unit. Also report top-one human-label hit rate, per-state precision/recall/F1, state-count error, exact label-set accuracy, and one-to-one segment F1 at IoU 0.1/0.25/0.5. Empty human-label frames are excluded rather than treated as negative labels. Predictions are sampled every two seconds and repeated onto raw 30fps frames; those frames are correlated. Matched interval endpoint error must be read alongside segment recall.

Targets are dataset ingredient/object names such as vegetables or eggs, potentially combining multiple physical instances. These scores measure agreement with the annotation label sets. Per-instance tracking, object presence, and segmentation accuracy are not measured.

One validation recording, S13 Brownie, has 20,510 decoded frames but a published annotation array of 18,131 frames. Original annotation indices are preserved; the uncovered trailing 2,379 frames are marked unknown and excluded from scoring. The video at frame 7,200 shows egg cracking consistent with the indexed human labels; the visual check is in `validation-audit/`. Annotation clocks and vocabularies passed for all 20 train/validation object/video units. Fourteen broader tests and nine legacy tests pass.

Training reads train and validation labels only. Model, epoch, threshold, and smoothing choices are locked from validation results. Predictions for all compared models are saved and hashed before the evaluator opens test labels. Split planning inspected video/object names and label inventories, so this is a custom pilot rather than a blinded or full official MOSCATO benchmark. All videos are from the same CMU kitchen. No cross-kitchen or arbitrary-food accuracy is established.

The peeled experiment removes that name from supervised training candidates/loss and excludes supervised train/validation frames carrying it. Evaluation adds peeled back into the candidate set. Temporal context and pretrained encoders may still have seen its visual appearance. This is a labelled-state holdout, not a guarantee of zero prior exposure.

`protocol-manual.json` adds a separate comparison using four human-labelled training videos from subjects S08, S12, S24, and S41, one per recipe, with four target units. Its validation/test videos stay identical to the primary protocol and remain subject-disjoint. The primary protocol is unchanged. This extra method was declared after the pseudo-supervised model nearly tied its prior on validation, before fresh test predictions or scores existed. Both recordings and supervision source change, so the comparison does not isolate a label-quality effect. S08 has no separate egg-mixture training annotation. Across both protocols there are 25 unique videos.

Training labels are automatic MOSCATO pseudo labels. `training-audit/` contains coverage counts and a training-only inspection sheet. No manual corrections were applied. Do not confuse these training labels with the separate human evaluation annotations.

## Running locally

From the workspace root, using the existing environment:

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'work/model-cache'
$env:HF_HUB_CACHE = Join-Path (Get-Location) 'work/model-cache/open_clip'
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/test_broader.py
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/features.py --backbone clip --backend directml
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/features.py --backbone slowfast --backend directml_half
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/prepare_manual.py
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/download.py --protocol work/chefost-vidosc/experiments/shray_multistate/broader/protocol-manual.json --manifest sources-manual.json
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/features.py --backbone clip --backend directml --protocol work/chefost-vidosc/experiments/shray_multistate/broader/protocol-manual.json
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/run_experiments.py
```

The DirectML encoders require the ONNX models and verification manifests already generated in the workspace's own model cache. `--backend torch` can use the original CPU implementations. SlowFast half precision was accepted only after synthetic and real training-frame equivalence checks. CLIP half precision failed a check and is not used. Run directories are protected against overwriting or selecting again after inspecting test scores.

Those commands describe reproduction from a clean run directory. Existing completed runs are intentionally protected. Use their saved reports and checkpoints for review instead of repeating test-driven model selection.

For InternVideo, obtain `InternVideo-MM-L-14.ckpt` through the access rules at [OpenGVLab/InternVideo1.0](https://huggingface.co/OpenGVLab/InternVideo1.0). The [GitHub project](https://github.com/OpenGVLab/InternVideo) provides source code, not access to that checkpoint. Its extraction adapter is present, but the comparison cannot run until the weights are available.

## State descriptions

`predict_custom.py` accepts a nonempty JSON list of names, or objects containing `state` and `description`. Each description is used as the full CLIP text prompt. `candidate-descriptions-example.json` shows the format. The output includes state intervals and scores. Use an existing fully extracted video ID and an own checkpoint. Custom descriptions and new object names are supported inputs; their accuracy is not established by this evaluation.

```powershell
& work/vidosc-env/Scripts/python.exe work/chefost-vidosc/experiments/shray_multistate/broader/predict_custom.py --checkpoint outputs/shray-multistate/broader/results/clip/model.pt --video-id S15_Salad_7150991-451 --object vegetables --candidates work/chefost-vidosc/experiments/shray_multistate/broader/candidate-descriptions-example.json --output work/chefost-vidosc/experiments/shray_multistate/broader/custom-description-prediction.json
```

Report and results are complete in REPORT.md and results/. The accuracy goal is not achieved: all learned variants are below the stronger constant-prior baseline on this fresh test. Further changes motivated by these test scores need a new held-out test split for a fresh generalization claim.
