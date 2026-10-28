# Weights & Biases

Entity: `chefOST`.

- Experiments: `vidosc-experiments`
- Full evaluations: `vidosc - full evals`

This branch defaults to `chefOST/vidosc-experiments`, selected by `default_project`
in [settings.json](settings.json). `WANDB_ENTITY` and `WANDB_PROJECT` override the
file defaults; explicit CLI flags override the environment. Full-evaluation
logging must use the exact project name, including spaces.

`run_inference.py` and `describe_states.py` use these defaults when logging is
enabled. For `python -m moscato.run_experiment`, a bare `--wandb-project` opts in
to the configured project; `--wandb-project NAME` selects a different one.
Omitting the flag keeps that evaluator's logging off.

The VidOSC-only Modal pilot has no W&B logger yet. These defaults are also the
destination for the later two-video trial; configuring them does not launch it.

See [account setup](../docs/account-setup.md) for your API key and Modal/R2 setup.
