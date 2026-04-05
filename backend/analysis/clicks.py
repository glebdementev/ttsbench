import numpy as np
from scipy.ndimage import percentile_filter
import librosa
from .base import BaseDetector, DetectorResult, ArtifactRegion


class ClickDetector(BaseDetector):
    name = "clicks"

    def __init__(self, weight: float = 0.20):
        super().__init__(weight)

    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        # Detect pops/clicks caused by waveform discontinuities (segment
        # stitching, phase jumps).  A pop produces a single-sample spike in
        # |diff(y)| that far exceeds anything in its local neighbourhood.
        diff = np.abs(np.diff(y))

        # Local 95th-percentile of |diff| over a 10 ms window — represents
        # the loudest "normal" derivative in the neighbourhood.
        win = int(sr * 0.010)
        if win % 2 == 0:
            win += 1
        local_p95 = percentile_filter(diff, percentile=95, size=win)

        ratio = diff / (local_p95 + 1e-10)

        # Voice-activity mask: ignore silence
        hop = 256
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
        s2f = np.minimum(np.arange(len(diff)) // hop, len(rms) - 1)
        voiced = rms[s2f] > 0.01

        # Detection: ratio > 6  AND  |diff| significant (> 0.05)
        pop_mask = (ratio > 6.0) & (diff > 0.05) & voiced

        # Merge nearby detections (within 5 ms)
        regions = self._merge(pop_mask, ratio, diff, sr)

        num_clicks = len(regions)
        score = max(0, 100 - num_clicks * 5)

        return DetectorResult(
            score=float(score),
            regions=regions,
            raw_metrics={"num_clicks": num_clicks},
        )

    @staticmethod
    def _merge(
        mask: np.ndarray, ratio: np.ndarray, diff: np.ndarray, sr: int
    ) -> list[ArtifactRegion]:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []

        merge_gap = int(sr * 0.005)  # 5 ms
        regions: list[ArtifactRegion] = []
        start = indices[0]
        end = indices[0]

        for idx in indices[1:]:
            if idx - end <= merge_gap:
                end = idx
            else:
                regions.append(_make_region(start, end, ratio, diff, sr))
                start = idx
                end = idx
        regions.append(_make_region(start, end, ratio, diff, sr))
        return regions


def _make_region(
    start: int, end: int, ratio: np.ndarray, diff: np.ndarray, sr: int
) -> ArtifactRegion:
    peak_r = float(np.max(ratio[start : end + 1]))
    severity = "high" if peak_r > 10 else "medium" if peak_r > 7 else "low"
    return ArtifactRegion(
        start=round(start / sr, 4),
        end=round((end + 1) / sr, 4),
        severity=severity,
        label=f"pop ({peak_r:.1f}x)",
        type="clicks",
    )
