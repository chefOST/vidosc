# Experiment accounts

Install the environment in [setup and workflow](FULL_RUN_HANDOFF.md#setup) first.
You need to sign into your own Modal account and authorize W&B locally, then
approve a budget before the two-video test. Complete both local logins before
running `setup_accounts.py`. Do not paste keys into chat.

## Modal

Authenticate in your terminal and select your own workspace/profile:

```sh
.venv/bin/python -m modal token new
.venv/bin/python -m modal profile list
.venv/bin/python -m modal profile activate YOUR_PROFILE_NAME
.venv/bin/python -m modal profile current
```

Use the profile name printed by `profile list`. Tokens live in `~/.modal.toml`,
outside the repository. Exported `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` override
profile credentials; unset them if they belong to another account. `MODAL_PROFILE`
pins a shell profile, and `MODAL_ENVIRONMENT` selects its secrets and volumes
(`main` by default).

## R2

The existing local AWS profile `chefost-r2` supplies read-only R2 credentials.
[`configs/r2_release.json`](../configs/r2_release.json) holds the bucket, endpoint,
prefix and release manifest. No AWS access keys belong in repository files.

After Modal and W&B login, copy credentials directly from that profile into the
`chefost-r2` Secret in your selected Modal workspace/environment:

```sh
AWS_PROFILE=chefost-r2 .venv/bin/python scripts/setup_accounts.py \
  --r2-secret --workspace YOUR_WORKSPACE
```

Use the intended Modal workspace name; the explicit flag prevents copying keys
into the wrong account. The setup script copies without displaying keys or asking
for them in chat. The Secret contains `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
for the worker's read-only bucket mount. Another contributor's Modal Secret is not shared
with your workspace. Inventory reads the release manifest and listing locally;
full videos are read by Modal, not downloaded to the laptop.

## W&B

Authenticate locally using your key from the [W&B authorization page](https://wandb.ai/authorize):

```sh
.venv/bin/python -m wandb login
```

Copy [.env.example](../.env.example) to `.env` if you need shell overrides, then load
the ignored file in the shell that will run the coordinator:

```sh
set -a
. ./.env
set +a
```

Python and Modal do not automatically load this file. `WANDB_API_KEY` in `.env`
is an alternative to the local W&B login; keep it out of Git. The default destination
is `chefOST/vidosc-experiments`; see [W&B configuration](../weights_and_biases/README.md)
for overrides and uploads.

After both logins and Secret setup, verify account access without running a model:

```sh
.venv/bin/python scripts/setup_accounts.py --check
```

This read-only check verifies Modal credentials and the Secret name, plus W&B
account and project access. Successful account checks do not authorize paid
inference; execution still needs the approved
budget described in [the workflow](FULL_RUN_HANDOFF.md#preview-and-execution).
