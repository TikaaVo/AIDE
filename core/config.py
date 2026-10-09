"""Loads config/settings.yaml once, relative to the repo root (not the cwd).

Set AIDE_CONFIG=/path/to/other.yaml to use a different file."""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def load_config() -> dict:
    with open(os.environ.get("AIDE_CONFIG") or ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)
