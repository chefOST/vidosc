from moscato.paths import checkpoint_path, state_vocabulary_path


def test_sibling_worktrees_share_assets_without_a_checkout_dependency(tmp_path, monkeypatch):
    monkeypatch.delenv("VIDOSC_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("VIDOSC_STATE_VOCABULARY", raising=False)
    for root in (tmp_path / "cpu", tmp_path / "gpu"):
        assert checkpoint_path("changeit/state_best.ckpt", root) == (
            tmp_path / "data/checkpoints/vidosc/changeit/state_best.ckpt"
        )
        assert state_vocabulary_path(root) == (
            tmp_path / "chefOST-data/MOSCATO/vocabulary/CMU/state_dict.json"
        )


def test_explicit_asset_paths_work_outside_workspace_layout(tmp_path, monkeypatch):
    checkpoint_directory = tmp_path / "weights"
    vocabulary = tmp_path / "labels.json"
    monkeypatch.setenv("VIDOSC_CHECKPOINT_DIR", str(checkpoint_directory))
    monkeypatch.setenv("VIDOSC_STATE_VOCABULARY", str(vocabulary))
    root = tmp_path / "unrelated/checkout"
    assert checkpoint_path("changeit/state_best.ckpt", root) == checkpoint_directory / "changeit/state_best.ckpt"
    assert state_vocabulary_path(root) == vocabulary
