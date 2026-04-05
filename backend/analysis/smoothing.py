import numpy as np
import librosa
from .base import BaseDetector, DetectorResult, ArtifactRegion


# Reference GV values for natural speech MFCCs (coefficients 1-12)
# These are approximate median values from multiple natural speech datasets.
NATURAL_GV_REFERENCE = np.array([
    120, 60, 40, 30, 25, 20, 18, 15, 13, 12, 11, 10
], dtype=float)


class SmoothingDetector(BaseDetector):
    name = "smoothing"

    def __init__(self, weight: float = 0.15):
        super().__init__(weight)

    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        # MFCCs (skip c0 which is energy)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        mfccs = mfccs[1:]  # 12 coefficients

        # Global Variance per coefficient
        gv = np.var(mfccs, axis=1)
        gv_ratio = gv / (NATURAL_GV_REFERENCE + 1e-10)
        mean_gv_ratio = float(np.mean(gv_ratio))

        # Spectral flux
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux_std = float(np.std(flux))
        flux_mean = float(np.mean(flux))
        flux_cv = flux_std / (flux_mean + 1e-10)  # coefficient of variation

        regions = []
        hop_dur = 512 / sr

        # Detect stretches of low spectral flux (over-smoothed segments)
        if flux_mean > 0:
            low_flux_threshold = flux_mean * 0.3
            low_flux = flux < low_flux_threshold
            in_low = False
            seg_start = 0
            for i, is_low in enumerate(low_flux):
                if is_low and not in_low:
                    seg_start = i
                    in_low = True
                elif not is_low and in_low:
                    if (i - seg_start) > 10:  # at least ~100ms
                        regions.append(ArtifactRegion(
                            start=round(seg_start * hop_dur, 4),
                            end=round(i * hop_dur, 4),
                            severity="medium",
                            label=f"over-smoothed (low spectral flux)",
                            type="smoothing",
                        ))
                    in_low = False

        # Score: combine GV ratio and spectral flux variability
        # GV ratio < 0.5 means over-smoothed
        gv_score = np.clip(mean_gv_ratio / 0.5 * 100, 0, 100) if mean_gv_ratio < 1.0 else 100
        flux_score = np.clip(flux_cv / 0.5 * 100, 0, 100)
        score = 0.6 * gv_score + 0.4 * flux_score

        return DetectorResult(
            score=float(np.clip(score, 0, 100)),
            regions=regions,
            raw_metrics={
                "mean_gv_ratio": round(mean_gv_ratio, 3),
                "spectral_flux_cv": round(flux_cv, 3),
            },
        )
