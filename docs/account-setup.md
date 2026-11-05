# Experiment accounts

Install the environment in [setup and workflow](FULL_RUN_HANDOFF.md#setup) first.

## Modal

Authenticate through the browser and select your own workspace/profile:

```sh
.venv/bin/python -m modal token new
.venv/bin/python -m modal profile list
.venv/bin/python -m modal profile activate YOUR_PROFILE_NAME
.venv/bin/python -m modal profile current
```

Use the profile name printed by `profile list`. Tokens live in `~/.modal.toml`,
outside the repository. Exported `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` override
profile credentials; unset them if they belong to another account. `MODAL_PROFILE`
pins the selected shell profile, and `MODAL_ENVIRONMENT` selects the environment
containing its secrets and volumes (`main` by default).

References: [profiles](https://modal.com/docs/cli/latest/profile) and
[credential precedence](https://modal.com/docs/sdk/py/latest/config).

## Shared R2 videos

In the [Modal dashboard](https://modal.com), choose your workspace and environment,
then create a custom Secret named `chefost-r2` with:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

Use the read-only values shared in the private team Discord channel. These are
R2 S3 credentials; your Modal login token is separate. The Secret must exist in
your workspace even if another contributor already created one in theirs.

The [shared-data video guide](https://github.com/chefOST/chefOST-data#videos) is
canonical for the bucket, endpoint, release folder and local-download commands.
The current full-video pipeline takes local AVIs; R2 MP4 integration is future
work and requires fresh source hashes, sampling manifests and alignment checks.
It uploads videos and weights from the local machine, then extracts global/object
features and runs the model on Modal CPU.

## W&B

Copy [.env.example](../.env.example) to `.env` and fill in `WANDB_API_KEY` from
your [W&B authorization page](https://wandb.ai/authorize). Keep the experiment
destination from the example unless intentionally overriding it. Load the ignored
file into the shell that will run the coordinator:

```sh
set -a
. ./.env
set +a
```

Python and Modal do not automatically load this file. See
[W&B configuration](../weights_and_biases/README.md) for destination precedence
and uploads. Keep all credential values out of Git.
