"""Offline checks for checkpoint integrity and safe final-file publication."""
import hashlib
from pathlib import Path
import sys
import types

import pytest

from scripts import setup_checkpoints as setup


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    return {"name": "fixture", "path": "checkpoints/model.ckpt", "bytes": 4,
            "sha256": hashlib.sha256(b"good").hexdigest()}


def test_valid_download_is_promoted_after_verification(checkpoint, monkeypatch):
    destination = setup.ROOT / checkpoint["path"]

    def download(spec, partial):
        assert spec == checkpoint
        assert partial.name == "model.ckpt.part"
        assert not destination.exists()
        partial.write_bytes(b"good")

    monkeypatch.setattr(setup, "download", download)
    assert setup.ensure_checkpoint(checkpoint) == destination
    assert destination.read_bytes() == b"good"
    assert not destination.with_suffix(".ckpt.part").exists()


@pytest.mark.parametrize("content", [b"evil", b"truncated"])
def test_invalid_download_never_creates_final_file(checkpoint, monkeypatch, content):
    monkeypatch.setattr(setup, "download", lambda spec, path: path.write_bytes(content))
    with pytest.raises(ValueError, match="mismatch"):
        setup.ensure_checkpoint(checkpoint)
    assert not (setup.ROOT / checkpoint["path"]).exists()
    assert not list(setup.ROOT.rglob("*.part"))


@pytest.mark.parametrize("content,valid", [(b"good", True), (b"evil", False)])
def test_existing_checkpoint_is_never_overwritten(checkpoint, monkeypatch, content, valid):
    path = setup.ROOT / checkpoint["path"]
    path.parent.mkdir()
    path.write_bytes(content)
    monkeypatch.setattr(setup, "download", lambda *args: pytest.fail("Unexpected download"))
    if valid:
        assert setup.ensure_checkpoint(checkpoint) == path
    else:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            setup.ensure_checkpoint(checkpoint)
    assert path.read_bytes() == content


def test_verify_only_missing_does_not_write_or_download(checkpoint, monkeypatch):
    monkeypatch.setattr(setup, "download", lambda *args: pytest.fail("Unexpected download"))
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        setup.ensure_checkpoint(checkpoint, verify_only=True)
    assert not list(setup.ROOT.iterdir())


def test_download_failure_cleans_partial(checkpoint, monkeypatch):
    def fail(spec, partial):
        partial.write_bytes(b"interrupted")
        raise OSError("connection failed")

    monkeypatch.setattr(setup, "download", fail)
    with pytest.raises(OSError, match="connection failed"):
        setup.ensure_checkpoint(checkpoint)
    assert not list(setup.ROOT.rglob("*.part"))
    assert not (setup.ROOT / checkpoint["path"]).exists()


def test_main_attempts_other_checkpoints_after_failure(monkeypatch):
    attempts = []

    def ensure(spec, **kwargs):
        attempts.append(spec["name"])
        if len(attempts) == 1:
            raise OSError("source unavailable")

    monkeypatch.setattr(setup, "ensure_checkpoint", ensure)
    monkeypatch.setattr("sys.argv", ["setup_checkpoints.py"])
    with pytest.raises(SystemExit) as failure:
        setup.main()
    assert failure.value.code == 1
    assert attempts == [spec["name"] for spec in setup.CHECKPOINTS]


def test_hub_download_stays_temporary_until_hash_verified(checkpoint, monkeypatch):
    checkpoint.update(hf_repo="publisher/model", filename="model.ckpt")
    destination = setup.ROOT / checkpoint["path"]

    def fake_download(**kwargs):
        assert kwargs["repo_id"] == "publisher/model"
        assert kwargs["filename"] == "model.ckpt"
        directory = Path(kwargs["local_dir"])
        assert directory.parent == destination.parent
        assert not destination.exists()
        downloaded = directory / "model.ckpt"
        downloaded.write_bytes(b"evil")
        return str(downloaded)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=fake_download))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors",
                        types.SimpleNamespace(GatedRepoError=type("GatedRepoError", (Exception,), {})))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        setup.ensure_checkpoint(checkpoint)
    assert not destination.exists()
    assert not list(destination.parent.iterdir())


def test_hub_access_error_explains_local_login(checkpoint, monkeypatch):
    checkpoint.update(hf_repo="publisher/model", filename="model.ckpt")
    error = type("GatedRepoError", (Exception,), {})

    def denied(**kwargs):
        raise error("access denied")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=denied))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors",
                        types.SimpleNamespace(GatedRepoError=error))
    with pytest.raises(RuntimeError, match="hf auth login"):
        setup.ensure_checkpoint(checkpoint)
    assert not (setup.ROOT / checkpoint["path"]).exists()
    assert not list((setup.ROOT / "checkpoints").iterdir())
