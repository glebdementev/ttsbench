import numpy as np
import parselmouth
from parselmouth.praat import call
from .base import BaseDetector, DetectorResult, ArtifactRegion


class PitchContourDetector(BaseDetector):
    """Detect unnatural pitch contour shape.

    Distinct from roboticness (periodicity/regularity): this focuses on
    whether the F0 trajectory itself looks natural.  Measures:
    - Modulation spectrum slope: natural speech F0 follows ~1/f²; TTS is
      often too shallow (insufficient phrase-level intonation)
    - Local variance heterogeneity: natural speech has passages with more/less
      pitch movement; uniform variance signals synthetic averaging
    - Monotonic run length: natural glides produce longer same-direction runs;
      choppy TTS reverses direction frequently
    - Micro-perturbation consistency: natural consonant-induced F0 perturbations
      vary; over-controlled TTS has uniform micro-variation
    """

    name = "pitch_contour"

    def __init__(self, weight: float = 0.10):
        super().__init__(weight)

    def analyze(self, y: np.ndarray, sr: int) -> DetectorResult:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        duration = len(y) / sr
        regions = []

        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        dt = call(pitch, "Get time step")
        num_frames = call(pitch, "Get number of frames")

        f0_times = []
        f0_values = []
        for i in range(1, num_frames + 1):
            t = call(pitch, "Get time from frame number", i)
            f0 = call(pitch, "Get value in frame", i, "Hertz")
            if f0 and not np.isnan(f0) and f0 > 0:
                f0_times.append(t)
                f0_values.append(f0)

        if len(f0_values) < 30:
            return DetectorResult(score=50.0, regions=[], raw_metrics={
                "voiced_frames": len(f0_values),
                "note": "too few voiced frames for reliable analysis",
            })

        f0_arr = np.array(f0_values)
        f0_t = np.array(f0_times)
        median_f0 = np.median(f0_arr)
        f0_st = 12.0 * np.log2(f0_arr / median_f0)
        deltas_raw = np.diff(f0_st)
        deltas = deltas_raw[np.abs(deltas_raw) < 4.0]
        if len(deltas) < 20:
            deltas = deltas_raw

        # --- 1. Modulation spectrum slope ---
        # Treat F0 contour as a signal, compute power spectrum slope.
        # Natural speech: slope ~-1.5 to -2.5 (strong phrase-level structure).
        # TTS with weak intonation: slope ~-0.5 to -1.0 (too flat spectrally).
        mod_slope = 0.0
        if len(f0_st) > 64:
            f0_centered = f0_st - np.mean(f0_st)
            win = np.hanning(len(f0_centered))
            spectrum = np.abs(np.fft.rfft(f0_centered * win))
            freqs = np.fft.rfftfreq(len(f0_centered), d=dt)
            mask = (freqs >= 0.5) & (freqs <= 15) & (spectrum > 0)
            if np.sum(mask) > 5:
                log_f = np.log10(freqs[mask])
                log_s = np.log10(spectrum[mask])
                mod_slope = float(np.polyfit(log_f, log_s, 1)[0])

        # --- 2. Local variance heterogeneity ---
        # CV of local pitch variances. High = natural variation in prosodic
        # activity (questions, emphasis vs calm). Low = uniform synthetic pitch.
        win_size = max(20, int(0.3 / dt))
        local_vars = []
        for start in range(0, len(f0_st) - win_size, win_size // 2):
            seg = f0_st[start:start + win_size]
            local_vars.append(np.var(seg))
        local_vars = np.array(local_vars)
        var_cv = float(np.std(local_vars) / (np.mean(local_vars) + 1e-10)) if len(local_vars) > 2 else 0.0

        # --- 3. Monotonic run statistics ---
        # Natural speech has smooth glides (longer runs in same direction).
        # Choppy/noisy TTS flips direction frequently.
        signs = np.sign(deltas)
        runs = []
        current_run = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1] and signs[i] != 0:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        runs = np.array(runs)
        mean_run = float(np.mean(runs)) if len(runs) > 0 else 1.0

        # --- 4. Micro-perturbation consistency ---
        # Std of F0 deltas within 50ms windows. High CV = natural variation
        # in perturbation level; low CV = over-controlled uniform micro-variation.
        micro_win = max(5, int(0.05 / dt))
        micro_stds = []
        for start in range(0, len(deltas) - micro_win, micro_win):
            seg = deltas[start:start + micro_win]
            micro_stds.append(np.std(seg))
        micro_stds = np.array(micro_stds)
        micro_cv = float(np.std(micro_stds) / (np.mean(micro_stds) + 1e-10)) if len(micro_stds) > 2 else 0.0

        # --- 5. Direction reversal rate ---
        reversal_rate = float(np.mean(np.abs(np.diff(signs)) > 0)) if len(signs) > 1 else 0.5

        # --- Scoring ---
        # Modulation slope: natural -1.5 to -2.5; TTS often -0.5 to -1.0.
        # More negative = better phrase structure.
        slope_score = _trapezoid(-mod_slope, 0.3, 1.3, 2.5, 3.5)

        # Var CV: natural ~1.3-2.0; uniform TTS <1.0.
        var_cv_score = _trapezoid(var_cv, 0.3, 1.2, 2.0, 3.0)

        # Mean run: natural ~5-7; choppy <4.
        run_score = _trapezoid(mean_run, 2.0, 4.5, 7.0, 10.0)

        # Micro CV: natural ~1.1-1.4; over-controlled <0.9.
        micro_score = _trapezoid(micro_cv, 0.3, 1.05, 1.4, 2.5)

        # Reversal rate: natural ~0.20-0.25; choppy >0.32.
        reversal_score = _trapezoid(reversal_rate, 0.10, 0.20, 0.25, 0.38)

        score = (
            0.30 * slope_score
            + 0.25 * var_cv_score
            + 0.15 * run_score
            + 0.15 * micro_score
            + 0.15 * reversal_score
        )

        # --- Detect problem regions ---
        win_frames = max(20, int(0.5 / dt))
        step = max(1, win_frames // 2)

        for start_idx in range(0, len(f0_st) - win_frames, step):
            end_idx = start_idx + win_frames
            seg = f0_st[start_idx:end_idx]
            seg_deltas = np.diff(seg)

            seg_range = float(np.ptp(seg))
            seg_var = float(np.var(seg))

            # Flat segment: very low pitch range
            if seg_range < 0.5:
                t_start = f0_t[start_idx] if start_idx < len(f0_t) else 0
                t_end = f0_t[min(end_idx, len(f0_t) - 1)] if end_idx < len(f0_t) else duration
                regions.append(ArtifactRegion(
                    start=round(t_start, 4),
                    end=round(t_end, 4),
                    severity="high",
                    label=f"pitch: flat ({seg_range:.1f}st)",
                    type="pitch_contour",
                ))
            elif seg_range < 1.0:
                t_start = f0_t[start_idx] if start_idx < len(f0_t) else 0
                t_end = f0_t[min(end_idx, len(f0_t) - 1)] if end_idx < len(f0_t) else duration
                regions.append(ArtifactRegion(
                    start=round(t_start, 4),
                    end=round(t_end, 4),
                    severity="medium",
                    label=f"pitch: narrow ({seg_range:.1f}st)",
                    type="pitch_contour",
                ))

        regions = _merge_regions(regions)

        return DetectorResult(
            score=float(np.clip(score, 0, 100)),
            regions=regions,
            raw_metrics={
                "median_f0_hz": round(float(median_f0), 1),
                "mod_slope": round(mod_slope, 3),
                "var_cv": round(var_cv, 3),
                "mean_run": round(mean_run, 2),
                "micro_cv": round(micro_cv, 3),
                "reversal_rate": round(reversal_rate, 3),
                "sub_scores": {
                    "slope": round(slope_score, 1),
                    "var_cv": round(var_cv_score, 1),
                    "run": round(run_score, 1),
                    "micro": round(micro_score, 1),
                    "reversal": round(reversal_score, 1),
                },
            },
        )


def _trapezoid(x: float, low: float, good_low: float, good_high: float, high: float) -> float:
    if x < low or x > high:
        return 0.0
    if good_low <= x <= good_high:
        return 100.0
    if x < good_low:
        return 100.0 * (x - low) / (good_low - low)
    return 100.0 * (high - x) / (high - good_high)


def _merge_regions(regions: list[ArtifactRegion]) -> list[ArtifactRegion]:
    if not regions:
        return []
    regions.sort(key=lambda r: r.start)
    merged = [regions[0]]
    for r in regions[1:]:
        prev = merged[-1]
        if r.start <= prev.end + 0.05:
            sev_order = {"low": 0, "medium": 1, "high": 2}
            worse = max(prev.severity, r.severity, key=lambda s: sev_order[s])
            merged[-1] = ArtifactRegion(
                start=prev.start, end=max(prev.end, r.end),
                severity=worse, label=prev.label, type=prev.type,
            )
        else:
            merged.append(r)
    return merged
