import numpy as np
import librosa
from .base import BaseDetector, DetectorResult, ArtifactRegion


class BandwidthDetector(BaseDetector):
    """Оценка спектральной ширины полосы аудио.

    Узкая полоса (телефонные кодеки, низкий битрейт) — потеря энергии
    выше 3-4 кГц. Полноценный TTS должен иметь заметную энергию до 8-10 кГц.

    Метрики:
    - Спектральный центроид — среднее взвешенное частот по мощности.
    - Доля энергии выше 6 кГц от общей — устойчива к локальным провалам
      в спектре, отражает реальную ширину полосы.
    """

    name = "bandwidth"

    def __init__(self, weight: float = 0.15):
        super().__init__(weight)

    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        n_fft = 4096
        hop = 512
        hop_dur = hop / sr

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        avg_power = np.mean(S ** 2, axis=1)
        total_power = avg_power.sum() + 1e-20

        # --- Спектральный центроид ---
        centroid = float(np.average(freqs, weights=avg_power + 1e-20))

        # --- Доля энергии выше 6 кГц от общей ---
        hf_energy_frac = float(avg_power[freqs >= 6000].sum() / total_power)

        # --- Оценка ---
        # Центроид: <=400 → 0 (очень узко), >=900 → 100 (полная полоса)
        centroid_score = float(np.clip((centroid - 400) / 500 * 100, 0, 100))

        # Доля ВЧ >6 кГц: <=0.002 → 0, >=0.025 → 100
        hf_score = float(np.clip((hf_energy_frac - 0.002) / 0.023 * 100, 0, 100))

        score = 0.5 * centroid_score + 0.5 * hf_score

        # --- Покадровое обнаружение узкой полосы ---
        frame_power = S ** 2
        frame_total = frame_power.sum(axis=0) + 1e-20
        frame_hf = frame_power[freqs >= 4000].sum(axis=0)
        frame_hf_frac = frame_hf / frame_total

        rms = np.sqrt(np.mean(frame_power, axis=0))
        rms_thresh = np.percentile(rms, 30)
        speech_mask = rms > rms_thresh

        narrow_threshold = 0.01
        narrow_mask = speech_mask & (frame_hf_frac < narrow_threshold)

        regions = []
        if np.any(narrow_mask):
            narrow_indices = np.where(narrow_mask)[0]
            groups = self._group_frames(narrow_indices, max_gap=3)
            for group in groups:
                if len(group) < 5:
                    continue
                local_hf = float(np.mean(frame_hf_frac[group]))
                severity = (
                    "high" if local_hf < 0.005
                    else "medium" if local_hf < 0.01
                    else "low"
                )
                regions.append(ArtifactRegion(
                    start=round(group[0] * hop_dur, 4),
                    end=round((group[-1] + 1) * hop_dur, 4),
                    severity=severity,
                    label=f"узкая полоса (ВЧ энергия {local_hf:.4f})",
                    type="bandwidth",
                ))

        return DetectorResult(
            score=round(score, 1),
            regions=regions,
            raw_metrics={
                "spectral_centroid_hz": round(centroid, 0),
                "hf_energy_fraction_6k": round(hf_energy_frac, 4),
            },
        )

    @staticmethod
    def _group_frames(indices: np.ndarray, max_gap: int = 3) -> list[np.ndarray]:
        groups: list[list[int]] = [[indices[0]]]
        for i in range(1, len(indices)):
            if indices[i] <= groups[-1][-1] + max_gap:
                groups[-1].append(indices[i])
            else:
                groups.append([indices[i]])
        return [np.array(g) for g in groups]
