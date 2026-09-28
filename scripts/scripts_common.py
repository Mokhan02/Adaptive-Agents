"""Helpers shared by the analysis scripts."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from regime.analysis import Calibration


def load_cal(agent: str) -> Calibration:
    d = json.loads(Path(f"results/calibration/{agent}.json").read_text())
    return Calibration(**{f.name: d[f.name] for f in fields(Calibration)})
