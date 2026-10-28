# Experiment accounts

This branch uses `chefOST/vidosc-experiments` for W&B. No experiment is started
by configuring these accounts. The two-video timing/cost trial is a separate step.

## Modal: use your own account

In a Python environment with [requirements-modal.txt](../requirements-modal.txt)
installed, authenticate through the browser and choose your own workspace:

```sh
python -m modal token new
python -m modal profile list
python -m modal profile activate YOUR_PROFILE_NAME
python -m modal profile current
```

Use the actual profile name printed by `profile list`. Modal stores login tokens
in `~/.modal.toml`; do not put them in the repository. If your shell already has
`MODAL_TOKEN_ID` or `MODAL_TOKEN_SECRET` exported for another account, unset those
two variables before using your saved profile: they override profile credentials.
Set `MODAL_PROFILE` to pin a shell to your profile, and `MODAL_ENVIRONMENT` to the
environment containing your secrets (`main` by default).

References: [profiles](https://modal.com/docs/cli/latest/profile),
[credential precedence](https://modal.com/docs/sdk/py/latest/config).

## R2: access the shared videos

In the [Modal dashboard](https://modal.com), select your workspace and `main`
environment, open **Secrets**, and create a custom Secret named `chefost-r2`.
Add these two fields using the read-only values from the private team Discord:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

These are R2's S3 credentials, not your Modal login token. The Secret must exist
in your own workspace/environment even if Pritika already created hers.

Use the canonical [R2 mount and local-download instructions](https://github.com/chefOST/chefOST-data#videos).
The shared release currently contains 41 MP4s: 29 CMU and 12 EGTEA; EgoPER is
pending. The current annotation release has 87 recordings; older copies with 88
include a CMU recording now excluded for a video/annotation length mismatch.

## W&B: experiments project

Copy [.env.example](../.env.example) to `.env` in the repository root and fill in
`WANDB_API_KEY` from your [W&B authorization page](https://wandb.ai/authorize).
Keep `WANDB_ENTITY=chefOST` and `WANDB_PROJECT=vidosc-experiments`. `.env` is ignored
by Git. Load it into the shell that will launch the later experiment:

```sh
set -a
. ./.env
set +a
```

Python does not load `.env` automatically. Explicit CLI settings override these
environment values; otherwise [settings.json](../weights_and_biases/settings.json)
supplies the defaults. See [W&B setup](../weights_and_biases/README.md) for logging flags.

## Before the later two-video trial

Pritika's current `scripts/modal_clip.py` still uploads one fixed local AVI and
cached features. It does not yet mount R2 or log to W&B. The trial runner must add
that wiring and record timing/resource usage. Shared MP4s need fresh feature
manifests and frame-alignment checks; the old AVI manifest cannot be reused.
Model weights also remain prerequisites. No data download or paid run is needed
for the account setup above.
