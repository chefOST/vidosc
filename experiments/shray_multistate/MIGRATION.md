# VidOSC destination verification (2026-10-13)

Shray's active topic branch is `shray/feat/multistate-vidosc` in
`chefOST/vidosc`. The original clone and current origin both point to this
repository. No old remote was removed or replaced. The topic history was
already published; the local clone was unshallowed and its narrow fetch
configuration was extended to track Shray's branch alongside Michelle's.

Destination `main` has a separate, unrelated initialization history. It is
preserved as-is. No merge with `main`, history rewrite, or force-push was
performed. Configuration edits are committed only to Shray's existing topic
branch.

W&B routes to entity `chefOST`:

| Intent | Exact project |
|---|---|
| Development, pilots, and comparisons | `vidosc-experiments` |
| Intended full evaluations | `vidosc - full evals` |

`run_inference.py` and `describe_states.py` default to development.
The MOSCATO evaluation CLI retains opt-in logging: pass `--wandb-project`
with the appropriate project, and its entity defaults to `chefOST`.
Quote the full-evaluation project because it contains spaces.
No global/shared W&B settings or authentication files are modified.

The v3 retrospective logger already targets `chefOST/vidosc-experiments`;
its recorded run ID, historical runs, reports, and artifacts remain unchanged.
Do not reuse that run ID in another project. Fresh runs should receive fresh
IDs. No training, evaluation, run upload, or historical-data deletion is
performed during this migration.

The v3, v4, and v5 experiment folders currently contain local untracked work.
They are preserved locally and excluded from this migration-only commit.
The v5 baseline has no W&B uploader; choose its logging intent explicitly
before a future publication.

Both exact project destinations were accessible through the signed-in browser.
The local W&B SDK has no configured credential, so CLI/cloud upload access
remains unverified until normal W&B authentication is restored. Do not put
credentials in this file, source control, or chat.
