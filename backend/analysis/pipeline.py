import math
import numpy as np
from .clicks import ClickDetector
from .noise import NoiseDetector
from .robotic import RoboticDetector
from .smoothing import SmoothingDetector
from .buzzing import BuzzingDetector


DETECTORS = [
    ClickDetector(weight=0.25),
    NoiseDetector(weight=0.15),
    RoboticDetector(weight=0.25),
    SmoothingDetector(weight=0.15),
    BuzzingDetector(weight=0.20),
]


def _sanitize(obj):
    """Replace NaN/Inf with None so JSON serialization doesn't fail."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def run_pipeline(y: np.ndarray, sr: int) -> dict:
    sub_scores = {}
    for detector in DETECTORS:
        result = detector.analyze(y, sr)
        sub_scores[detector.name] = detector.to_dict(result)

    composite = sum(
        sub_scores[d.name]["score"] * d.weight for d in DETECTORS
    )

    return _sanitize({
        "score": round(composite, 1),
        "duration": round(len(y) / sr, 2),
        "sample_rate": sr,
        "sub_scores": sub_scores,
    })
