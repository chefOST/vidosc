# Weights & Biases

The default destination is `chefOST/vidosc-experiments`, selected by
`default_project` in [settings.json](settings.json). `WANDB_ENTITY` and
`WANDB_PROJECT` override those defaults; the logger's `--entity` and `--project`
flags override the environment. This workflow exports outputs without evaluation
scores. The separately recorded full-evaluation project is `vidosc - full evals`.

`scripts/run_full_batch.py` invokes the logger after each successful video and
sets its group to the experiment name (`full-r2-v1` by default). Standalone logger
calls retain the historical `cmu30-active-object-v1` default unless given `--group`.
The receipt records the actual destination and group; changing either explicitly
can upload a completed bundle without repeating model inference.

CMU artifacts keep the eight existing files: the feature manifest, combined/object
features, raw logits/probabilities and three mapped-prediction files. EGTEA artifacts
contain only the five raw files and record `dataset: EGTEA`, `mapping_applied: false`.
No EGTEA mapping or evaluation scores are invented, and no separate global-feature
artifact is added. Completion and crop coverage are the only run summaries.

Validate a saved bundle without uploading:

```sh
.venv/bin/python scripts/log_inference_wandb.py --run /path/to/video/model --dry-run
```

Omitting `--dry-run` uploads the bundle. For credentials, see
[account setup](../docs/account-setup.md); for files and resume behavior, see
[the handoff](../docs/FULL_RUN_HANDOFF.md#outputs-and-reuse).
