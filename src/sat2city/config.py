"""Path configuration shared by every script (local laptop or cloud)."""

import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_train_config(path: Path) -> dict:
    """Load a training YAML; a `base:` key (path relative to this file) is loaded first and
    overridden, so experiment configs only list what they change."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_train_config(path.parent / base), cfg)
    return cfg


def load_paths(path: Path = REPO_ROOT / "configs" / "paths.yaml") -> dict[str, Path]:
    """Load configs/paths.yaml, resolving ${key} references.

    SAT2CITY_DATA overrides data_root so the same configs work on Kaggle/Colab.
    """
    raw = yaml.safe_load(path.read_text())
    if os.environ.get("SAT2CITY_DATA"):
        raw["data_root"] = os.environ["SAT2CITY_DATA"]

    def resolve(value: str) -> str:
        return re.sub(r"\$\{(\w+)\}", lambda m: resolve(str(raw[m.group(1)])), value)

    return {key: Path(resolve(str(value))) for key, value in raw.items()}
