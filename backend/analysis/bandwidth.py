import numpy as np
import librosa
from .base import BaseDetector, DetectorResult, ArtifactRegion


class BandwidthDetector(BaseDetector):
    """Detect unnaturally narrow spectral bandwidth.

    Audio that has passed through telephony codecs (G.711, AMR) or
    low-bitrate compression loses most energy above ~3.4-4 kHz,
    producing a characteristic sharp high-frequency rolloff.
    Full-band TTS should have meaningful energy up to 8-10 kHz+.
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

        # --- Global metrics ---
        avg_power = np.mean(S ** 2, axis=1)
        total_power = avg_power.sum() + 1e-20

        # Effective bandwidth: freq below which 95% of energy sits
        cumulative = np.cumsum(avg_power) / total_power
        eff_bw_95 = float(freqs[np.searchsorted(cumulative, 0.95)])

        # HF rolloff sharpness: energy ratio (4-6 kHz) / (3-4 kHz)
        # Telephony has a sharp cutoff → low ratio; full-band → ~1+
        band_3_4k = avg_power[(freqs >= 3000) & (freqs < 4000)].sum() + 1e-20
        band_4_6k = avg_power[(freqs >= 4000) & (freqs < 6000)].sum()
        rolloff_ratio = float(band_4_6k / band_3_4k)

        # Energy above 4 kHz as fraction of total
        hf_energy_frac = float(avg_power[freqs >= 4000].sum() / total_power)

        # --- Score: combine rolloff_ratio and eff_bw_95 ---
        # rolloff_ratio: <=0.5 → 0 (telephone), >=1.2 → 100 (full-band)
        ratio_score = float(np.clip((rolloff_ratio - 0.5) / 0.7 * 100, 0, 100))

        # eff_bw_95: <=1200 → 0 (very narrow), >=2500 → 100 (full-band)
        bw_score = float(np.clip((eff_bw_95 - 1200) / 1300 * 100, 0, 100))

        score = 0.6 * ratio_score + 0.4 * bw_score

        # --- Per-frame narrowband detection ---
        frame_power = S ** 2
        frame_total = frame_power.sum(axis=0) + 1e-20
        frame_hf = frame_power[freqs >= 4000].sum(axis=0)
        frame_hf_frac = frame_hf / frame_total

        # Only flag speech-active frames
        rms = np.sqrt(np.mean(frame_power, axis=0))
        rms_thresh = np.percentile(rms, 30)
        speech_mask = rms > rms_thresh

        narrow_threshold = 0.01  # less than 1% energy above 4 kHz
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
                    label=f"narrow bandwidth (HF energy {local_hf:.4f})",
                    type="bandwidth",
                ))

        return DetectorResult(
            score=round(score, 1),
            regions=regions,
            raw_metrics={
                "eff_bandwidth_95_hz": round(eff_bw_95, 0),
                "rolloff_ratio": round(rolloff_ratio, 3),
                "hf_energy_fraction": round(hf_energy_frac, 4),
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
