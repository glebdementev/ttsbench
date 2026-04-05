import librosa
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
        duration = len(y) / sr
        regions = []

        # --- Rhythm regularity (segment duration variability) ---
        seg_cv = self._segment_cv(y, sr)

        # --- Pitch contour complexity (direction changes per second) ---
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        num_frames = call(pitch, "Get number of frames")
        f0_values = []
        for i in range(1, num_frames + 1):
            f0 = call(pitch, "Get value in frame", i, "Hertz")
            if f0 and not np.isnan(f0) and f0 > 0:
                f0_values.append(f0)

        if len(f0_values) > 20:
            f0_arr = np.array(f0_values)
            f0_st = 12 * np.log2(f0_arr / np.median(f0_arr))
            deltas = np.diff(f0_st)
            signs = np.sign(deltas)
            dir_changes = int(np.sum(np.abs(np.diff(signs)) > 0))
            dir_chg_per_sec = dir_changes / duration
        else:
            dir_chg_per_sec = 15.0  # neutral default

        # --- Spectral dynamics (timbral variation over time) ---
        spec_flux, centroid_cv = self._spectral_dynamics(y, sr)

        # --- Score components ---
        rhythm_score = _trapezoid(seg_cv, 0.45, 0.70, 1.3, 1.6)
        contour_score = _trapezoid(dir_chg_per_sec, 10, 14, 22, 28)
        flux_score = _sigmoid_score(spec_flux, midpoint=0.080, steepness=200)
        centroid_score = _sigmoid_score(centroid_cv, midpoint=0.72, steepness=25)

        score = (
            0.25 * rhythm_score
            + 0.20 * contour_score
            + 0.30 * flux_score
            + 0.25 * centroid_score
        )

        # --- Flag regions ---
        is_monotone = seg_cv < 0.65 or dir_chg_per_sec < 13
        is_metallic = spec_flux < 0.078 or centroid_cv < 0.70

        if is_monotone or is_metallic:
            parts = []
            if seg_cv < 0.65:
                parts.append(f"seg_cv {seg_cv:.2f}")
            if dir_chg_per_sec < 13:
                parts.append(f"contour {dir_chg_per_sec:.1f}/s")
            if spec_flux < 0.078:
                parts.append(f"flux {spec_flux:.4f}")
            if centroid_cv < 0.70:
                parts.append(f"cent_cv {centroid_cv:.2f}")

            severity = "high" if (is_monotone and is_metallic) else "medium"
            label_prefix = "robotic" if is_metallic else "monotone rhythm"
            regions.append(ArtifactRegion(
                start=0, end=round(duration, 4),
                severity=severity,
                label=f"{label_prefix} ({', '.join(parts)})",
                type="robotic",
            ))

        return DetectorResult(
            score=float(np.clip(score, 0, 100)),
            regions=regions,
            raw_metrics={
                "seg_cv": round(seg_cv, 3),
                "dir_chg_per_sec": round(dir_chg_per_sec, 1),
                "spec_flux": round(spec_flux, 4),
                "centroid_cv": round(centroid_cv, 4),
            },
        )

    @staticmethod
    def _spectral_dynamics(y: np.ndarray, sr: int) -> tuple[float, float]:
        """Spectral flux and centroid CV over voiced frames."""
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        voiced = rms > np.percentile(rms, 30)
        voiced = voiced[:S.shape[1]]

        # Spectral flux on normalized spectra (all frames — silence
        # drag reveals lack of overall spectral dynamism)
        S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
        flux_all = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
        spec_flux = float(np.mean(flux_all))

        # Centroid CV on voiced frames
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
        voiced_cent = voiced[:len(cent)]
        if voiced_cent.sum() > 10:
            c = cent[voiced_cent]
            centroid_cv = float(np.std(c) / (np.mean(c) + 1e-10))
        else:
            centroid_cv = float(np.std(cent) / (np.mean(cent) + 1e-10))

        return spec_flux, centroid_cv

    @staticmethod
    def _segment_cv(y: np.ndarray, sr: int) -> float:
        """Coefficient of variation of voiced-segment durations."""
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        energy = np.array([
            np.sum(y[i : i + frame_len] ** 2)
            for i in range(0, len(y) - frame_len, hop)
        ])
        threshold = np.percentile(energy, 20) * 2
        voiced = energy > threshold

        changes = np.diff(voiced.astype(np.int8))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) < 3 or len(ends) < 3:
            return 0.8  # neutral default

        if ends[0] < starts[0]:
            ends = ends[1:]
        n = min(len(starts), len(ends))
        starts, ends = starts[:n], ends[:n]

        seg_durs = (ends - starts) * 0.010
        seg_durs = seg_durs[seg_durs > 0.05]
        if len(seg_durs) < 5:
            return 0.8

        return float(np.std(seg_durs) / (np.mean(seg_durs) + 1e-10))


def _trapezoid(x: float, low: float, good_low: float, good_high: float, high: float) -> float:
    """Trapezoidal scoring: 0 outside [low, high], 100 inside [good_low, good_high], linear ramps."""
    if x < low or x > high:
        return 0.0
    if good_low <= x <= good_high:
        return 100.0
    if x < good_low:
        return 100.0 * (x - low) / (good_low - low)
    return 100.0 * (high - x) / (high - good_high)


def _sigmoid_score(x: float, midpoint: float, steepness: float) -> float:
    """Sigmoid scoring: values above midpoint → ~100, below → drops toward 0."""
    z = steepness * (x - midpoint)
    z = float(np.clip(z, -20, 20))
    return 100.0 / (1.0 + np.exp(-z))
