"""Path configuration shared by every script (local laptop or cloud)."""

import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


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
