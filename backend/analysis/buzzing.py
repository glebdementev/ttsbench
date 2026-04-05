import numpy as np
import librosa
from .base import BaseDetector, DetectorResult, ArtifactRegion


class BuzzingDetector(BaseDetector):
    name = "buzzing"

    def __init__(self, weight: float = 0.15):
        super().__init__(weight)

    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        # Estimate F0
        f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=2048, hop_length=512)
        hop_dur = 512 / sr

        # STFT for harmonic analysis
        S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        regions = []
        ratios = []

        for i, f in enumerate(f0):
            if f <= 0 or np.isnan(f) or i >= S.shape[1]:
                continue

            spectrum = S[:, i]
            # Harmonic bins
            harmonics = [int(round(f * h / (sr / 4096))) for h in range(1, 15) if f * h < sr / 2]
            interharmonics = [int(round(f * (h + 0.5) / (sr / 4096))) for h in range(1, 14) if f * (h + 0.5) < sr / 2]

            harmonics = [h for h in harmonics if 0 < h < len(spectrum)]
            interharmonics = [h for h in interharmonics if 0 < h < len(spectrum)]

            if not harmonics or not interharmonics:
                continue

            harm_energy = np.mean(spectrum[harmonics])
            inter_energy = np.mean(spectrum[interharmonics])

            if inter_energy > 1e-10:
                ratio_db = 20 * np.log10(harm_energy / inter_energy)
                ratios.append((i, ratio_db))

        if not ratios:
            return DetectorResult(score=90, regions=[], raw_metrics={"mean_hir_db": None})

        ratio_values = [r[1] for r in ratios]
        mean_hir = float(np.mean(ratio_values))

        # Flag frames with excessive harmonic clarity (>30dB)
        buzzy_frames = [(i, r) for i, r in ratios if r > 30]
        if buzzy_frames:
            # Merge consecutive buzzy frames
            starts = [buzzy_frames[0][0]]
            ends = [buzzy_frames[0][0]]
            for frame_i, _ in buzzy_frames[1:]:
                if frame_i <= ends[-1] + 2:
                    ends[-1] = frame_i
                else:
                    starts.append(frame_i)
                    ends.append(frame_i)

            for s, e in zip(starts, ends):
                local_ratio = np.mean([r for i, r in buzzy_frames if s <= i <= e])
                regions.append(ArtifactRegion(
                    start=round(s * hop_dur, 4),
                    end=round((e + 1) * hop_dur, 4),
                    severity="high" if local_ratio > 40 else "medium",
                    label=f"buzzing (HIR {local_ratio:.0f}dB)",
                    type="buzzing",
                ))

        # Score: penalize if mean HIR is too high (>30dB = buzzy) or too low (<10dB = noisy)
        if mean_hir > 30:
            score = max(0, 100 - (mean_hir - 30) * 5)
        elif mean_hir < 10:
            score = max(0, 100 - (10 - mean_hir) * 5)
        else:
            score = 100.0

        return DetectorResult(
            score=float(score),
            regions=regions,
            raw_metrics={"mean_hir_db": round(mean_hir, 1), "num_buzzy_frames": len(buzzy_frames)},
        )
