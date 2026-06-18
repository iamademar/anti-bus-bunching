"""Load config.yaml and resolve paths relative to the experiment folder."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The experiment root = parent of this src/ folder.
EXP_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]

    # convenience accessors -------------------------------------------------
    @property
    def od_dir(self) -> Path:
        return (EXP_ROOT / self.raw["paths"]["od_dir"]).resolve()

    @property
    def gtfs_dir(self) -> Path:
        return (EXP_ROOT / self.raw["paths"]["gtfs_dir"]).resolve()

    @property
    def processed_path(self) -> Path:
        return (EXP_ROOT / self.raw["paths"]["processed"]).resolve()

    @property
    def outputs_dir(self) -> Path:
        return (EXP_ROOT / self.raw["paths"]["outputs_dir"]).resolve()

    @property
    def figures_dir(self) -> Path:
        return (EXP_ROOT / self.raw["paths"]["figures_dir"]).resolve()

    @property
    def scope(self) -> dict[str, Any]:
        return self.raw["scope"]

    @property
    def label(self) -> dict[str, Any]:
        return self.raw["label"]

    @property
    def preprocess(self) -> dict[str, Any]:
        return self.raw["preprocess"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def evidence(self) -> dict[str, Any]:
        return self.raw.get("evidence", {})

    @property
    def simulate(self) -> dict[str, Any]:
        return self.raw.get("simulate", {})

    @property
    def seed(self) -> int:
        return int(self.raw["model"]["random_seed"])


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config.yaml (defaults to the one in the experiment root)."""
    cfg_path = Path(path) if path else (EXP_ROOT / "config.yaml")
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)


def ensure_java_home() -> str | None:
    """CapyMOA needs a JVM. On macOS, resolve JAVA_HOME via /usr/libexec/java_home if unset."""
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    import subprocess

    try:
        jh = subprocess.check_output(
            ["/usr/libexec/java_home"], text=True
        ).strip()
        if jh:
            os.environ["JAVA_HOME"] = jh
            return jh
    except Exception:
        pass
    return None
