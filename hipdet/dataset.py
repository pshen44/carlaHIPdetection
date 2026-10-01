"""Reading datasets written by ``hipdet.collect``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Sample:
    root: Path
    record: dict

    @property
    def sample_id(self) -> str:
        return self.record["sample_id"]

    @property
    def scenario(self) -> dict:
        return self.record["scenario"]

    @property
    def gt(self) -> dict:
        return self.record["gt"]

    def frame_paths(self) -> list:
        return [self.root / "frames" / f for f in self.record["frames"]]

    def frames(self) -> list:
        """RGB uint8 arrays for every frame in the burst."""
        return [np.asarray(Image.open(p).convert("RGB")) for p in self.frame_paths()]


def load(root) -> list:
    root = Path(root)
    lines = (root / "samples.jsonl").read_text().splitlines()
    return [Sample(root, json.loads(l)) for l in lines if l.strip()]
