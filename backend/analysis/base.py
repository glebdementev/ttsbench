from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ArtifactRegion:
    start: float       # seconds
    end: float         # seconds
    severity: str      # "low" | "medium" | "high"
    label: str
    type: str          # detector name


@dataclass
class DetectorResult:
    score: float                # 0-100, higher = cleaner
    regions: list[ArtifactRegion]
    raw_metrics: dict


class BaseDetector(ABC):
    name: str
    weight: float

    def __init__(self, weight: float):
        self.weight = weight

    @abstractmethod
    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        ...

    def to_dict(self, result: DetectorResult) -> dict:
        return {
            "score": round(result.score, 1),
            "weight": self.weight,
            "regions": [asdict(r) for r in result.regions],
            "raw_metrics": result.raw_metrics,
        }
