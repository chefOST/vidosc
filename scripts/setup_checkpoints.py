"""Download the pinned visual encoder and ChangeIt checkpoints without loading models."""
import argparse
import hashlib
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    {
        "name": "InternVideo-MM-L-14",
        "path": "checkpoints/internvideo/InternVideo-MM-L-14.ckpt",
        # The historical official Aliyun URL now returns 404. Use the current
        # publisher release with the user's existing Hugging Face credentials.
        "hf_repo": "OpenGVLab/InternVideo1.0",
        "filename": "InternVideo-MM-L-14.ckpt",
        "sha256": "a508e42229072f84a3fcd5fc3acdb7ebc969b27519ee1f56dac9996999cd40f6",
        "bytes": 2474306363,
    },
    {
        "name": "ChangeIt multitask state_best",
        "path": "checkpoints/changeit/state_best.ckpt",
        "drive_id": "1YpHscdX-ei1j5p5Qlsw-DTjUED5iu3aX",
        "sha256": "e6d1fa0d4708c8dffecba2a37f25149cc80c85fed33e6d4537bcaf98b0d94d36",
        "bytes": 133731419,
    },
)


def verify(path, checkpoint):
    """Validate bytes only; checkpoints are never deserialized."""
    if "bytes" in checkpoint and path.stat().st_size != checkpoint["bytes"]:
        raise ValueError(f"Size mismatch for {path}; existing files are not overwritten")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != checkpoint["sha256"]:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}; expected {checkpoint['sha256']}")
    return actual


def download(checkpoint, partial):
    if "drive_id" in checkpoint:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt in .venv to obtain gdown") from exc
        result = gdown.download(id=checkpoint["drive_id"], output=str(partial),
                                quiet=True, use_cookies=False)
        if result is None:
            raise RuntimeError(f"Google Drive download failed for {checkpoint['name']}")
        return
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt in .venv to obtain huggingface_hub") from exc
    # Keep downloaded bytes away from the final filename until our pinned hash
    # has been checked; never assume a Hub cache entry is already trustworthy.
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=partial.parent) as directory:
        try:
            downloaded = hf_hub_download(repo_id=checkpoint["hf_repo"],
                filename=checkpoint["filename"], local_dir=directory)
        except GatedRepoError:
            raise RuntimeError(
                "Hugging Face access is required. Accept the publisher's access conditions at "
                "https://huggingface.co/OpenGVLab/InternVideo1.0 and run "
                "`.venv/bin/hf auth login` in your terminal, then rerun this script. "
                "Do not send your token in chat."
            ) from None
        Path(downloaded).replace(partial)


def ensure_checkpoint(checkpoint, *, verify_only=False):
    destination = ROOT / checkpoint["path"]
    if destination.exists():
        actual = verify(destination, checkpoint)
        print(f"Verified existing {destination}: SHA256 {actual}", flush=True)
        return destination
    if verify_only:
        raise FileNotFoundError(f"Missing checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {checkpoint['name']} to {destination}", flush=True)
    try:
        download(checkpoint, partial)
        actual = verify(partial, checkpoint)
        if destination.exists():
            raise FileExistsError(f"Destination appeared during download; refusing to overwrite {destination}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"Verified download {destination}: SHA256 {actual}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Check existing checkpoint hashes without downloading or changing files")
    args = parser.parse_args()
    failures = []
    for checkpoint in CHECKPOINTS:
        try:
            ensure_checkpoint(checkpoint, verify_only=args.verify_only)
        except Exception as exc:
            failures.append(f"{checkpoint['name']}: {exc}")
    if failures:
        parser.exit(1, "Checkpoint setup incomplete:\n" + "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
