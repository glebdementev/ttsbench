import numpy as np
import librosa
from scipy.stats import gmean
from .base import BaseDetector, DetectorResult, ArtifactRegion


class NoiseDetector(BaseDetector):
    name = "noise"

    def __init__(self, weight: float = 0.15):
        super().__init__(weight)

    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        """Detect noise leaked from TTS reference recordings.

        TTS noise leakage manifests during speech only — the model reproduces
        the reference recording's noise floor as part of the synthesis.
        We detect it via spectral flatness in the high-frequency band (6-10 kHz)
        during speech-active frames: clean speech has structured roll-off,
        leaked noise makes the spectrum flatter.
        """
        n_fft = 4096
        hop = 512
        hop_dur = hop / sr

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Speech-active frames via RMS threshold
        rms = np.sqrt(np.mean(S ** 2, axis=0))
        rms_thresh = np.percentile(rms, 30)
        speech_mask = rms > rms_thresh

        if np.sum(speech_mask) < 10:
            return DetectorResult(score=90, regions=[], raw_metrics={
                "hf_flatness": None, "noisy_frame_pct": None,
            })

        # High-frequency band (6-10 kHz) — speech energy is minimal here,
        # but leaked reference noise persists
        hf_band = (freqs >= 6000) & (freqs <= 10000)
        hf_spec = S[hf_band, :]

        # Per-frame spectral flatness in HF band (speech frames only)
        speech_indices = np.where(speech_mask)[0]
        frame_flatness = np.empty(len(speech_indices))
        for j, idx in enumerate(speech_indices):
            col = hf_spec[:, idx] + 1e-10
            frame_flatness[j] = gmean(col) / np.mean(col)

        median_flatness = float(np.median(frame_flatness))

        # Score: low flatness = clean, high flatness = noisy
        # <=0.74 → 100 (чистый речевой спектр с естественным ВЧ-содержанием)
        # >=0.82 → 0   (плоский шумоподобный спектр — утечка шума из референса)
        # Порог 0.74 отделяет движки с широкополосным синтезом (ElevenLabs и т.п.)
        # от реально шумных (SciCom, XTTS), у которых flatness >0.76.
        score = float(np.clip((0.82 - median_flatness) / 0.08 * 100, 0, 100))

        # Region detection: flag speech frames with elevated flatness
        # Порог 0.80 — только явно шумные кадры, чтобы не помечать
        # легитимное ВЧ-содержание как шум
        noisy_threshold = 0.80
        noisy_mask = frame_flatness > noisy_threshold
        noisy_frame_pct = float(np.sum(noisy_mask) / len(frame_flatness) * 100)

        regions = []
        if np.any(noisy_mask):
            noisy_indices = speech_indices[noisy_mask]
            noisy_vals = frame_flatness[noisy_mask]
            groups = self._group_frames(noisy_indices, noisy_vals, max_gap=3)
            for indices, vals in groups:
                if len(indices) < 3:
                    continue
                local_flat = float(np.mean(vals))
                severity = "high" if local_flat > 0.85 else "medium" if local_flat > 0.80 else "low"
                regions.append(ArtifactRegion(
                    start=round(indices[0] * hop_dur, 4),
                    end=round((indices[-1] + 1) * hop_dur, 4),
                    severity=severity,
                    label=f"noise in speech (flatness {local_flat:.2f})",
                    type="noise",
                ))

        return DetectorResult(
            score=score,
            regions=regions,
            raw_metrics={
                "hf_flatness": round(median_flatness, 4),
                "noisy_frame_pct": round(noisy_frame_pct, 1),
            },
        )

    @staticmethod
    def _group_frames(
        indices: np.ndarray, values: np.ndarray, max_gap: int = 3,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        groups: list[tuple[list[int], list[float]]] = [([indices[0]], [values[0]])]
        for i in range(1, len(indices)):
            if indices[i] <= groups[-1][0][-1] + max_gap:
                groups[-1][0].append(indices[i])
                groups[-1][1].append(values[i])
            else:
                groups.append(([indices[i]], [values[i]]))
        return [(np.array(g[0]), np.array(g[1])) for g in groups]
