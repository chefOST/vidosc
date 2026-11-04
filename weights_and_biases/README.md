# Weights & Biases

The default destination is `chefOST/vidosc-experiments`, selected by
`default_project` in [settings.json](settings.json). `WANDB_ENTITY` and
`WANDB_PROJECT` override those defaults; the logger's `--entity` and `--project`
flags override the environment. The separately recorded full-evaluation project
is `vidosc - full evals`; the current workflow exports outputs without evaluation scores.

The coordinator invokes `scripts/log_inference_wandb.py` after each successful
video. It uploads features, logits, source/mapped probabilities and provenance,
plus completion and crop-coverage summaries. The default group is
`cmu30-active-object-v1`. The upload receipt records the actual destination.
When the destination changes, the coordinator uploads an existing completed bundle
to the new destination during execution without repeating model inference.

To validate an existing output bundle without uploading, run from the repository root:

```sh
.venv/bin/python scripts/log_inference_wandb.py --run /path/to/video/model --dry-run
```

Omitting `--dry-run` uploads that bundle. For credentials, see
[account setup](../docs/account-setup.md); for output files and resume behavior,
see [the handoff](../docs/FULL_RUN_HANDOFF.md#outputs-and-reuse).
