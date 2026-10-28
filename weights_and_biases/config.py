"""Shared W&B destinations; importing this module does not enable logging."""
import json
import os
from pathlib import Path


_settings = json.loads(Path(__file__).with_name("settings.json").read_text())
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", _settings["entity"])
WANDB_PROJECT = os.environ.get(
    "WANDB_PROJECT", _settings["projects"][_settings["default_project"]]
)
