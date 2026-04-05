import numpy as np
import parselmouth
from parselmouth.praat import call
from .base import BaseDetector, DetectorResult, ArtifactRegion


class RoboticDetector(BaseDetector):
    name = "robotic"

    def __init__(self, weight: float = 0.20):
        super().__init__(weight)

    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        regions = []

        # Jitter and shimmer
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # HNR
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Modulation spectrum: syllable-rate energy
        envelope = np.abs(y)
        # Simple low-pass via moving average (~50Hz cutoff at sr=22050)
        win = max(1, int(sr / 200))
        if win > 1:
            kernel = np.ones(win) / win
            env_smooth = np.convolve(envelope, kernel, mode='same')
        else:
            env_smooth = envelope
        # Downsample to 200Hz
        ds_rate = max(1, sr // 200)
        env_ds = env_smooth[::ds_rate]
        if len(env_ds) > 64:
            mod_fft = np.abs(np.fft.rfft(env_ds - np.mean(env_ds)))
            mod_freqs = np.fft.rfftfreq(len(env_ds), d=ds_rate / sr)
            syllable_mask = (mod_freqs >= 2) & (mod_freqs <= 6)
            total_mask = (mod_freqs >= 0.5) & (mod_freqs <= 30)
            syllable_ratio = float(np.sum(mod_fft[syllable_mask]) / (np.sum(mod_fft[total_mask]) + 1e-10))
        else:
            syllable_ratio = 0.3  # default neutral

        # Score components
        jitter_pct = (jitter or 0) * 100
        shimmer_pct = (shimmer or 0) * 100

        # Too-low jitter/shimmer = robotic; too-high = distorted
        jitter_score = 100 if 0.3 <= jitter_pct <= 2.0 else max(0, 100 - abs(jitter_pct - 1.0) * 50)
        shimmer_score = 100 if 1.5 <= shimmer_pct <= 8.0 else max(0, 100 - abs(shimmer_pct - 4.0) * 15)
        hnr_score = 100 if 8 <= (hnr or 0) <= 22 else max(0, 100 - abs((hnr or 15) - 15) * 5)
        mod_score = min(100, syllable_ratio / 0.3 * 100)

        score = 0.3 * jitter_score + 0.3 * shimmer_score + 0.2 * hnr_score + 0.2 * mod_score

        # Flag if clearly robotic
        if jitter_pct < 0.2 and shimmer_pct < 1.5:
            duration = len(y) / sr
            regions.append(ArtifactRegion(
                start=0, end=round(duration, 4),
                severity="high",
                label=f"robotic (jitter {jitter_pct:.2f}%, shimmer {shimmer_pct:.1f}%)",
                type="robotic",
            ))

        return DetectorResult(
            score=float(np.clip(score, 0, 100)),
            regions=regions,
            raw_metrics={
                "jitter_pct": round(jitter_pct, 3),
                "shimmer_pct": round(shimmer_pct, 2),
                "hnr_db": round(hnr or 0, 1),
                "syllable_ratio": round(syllable_ratio, 3),
            },
        )
