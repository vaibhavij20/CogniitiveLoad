"""
AWR-CLF: Adaptive Weighted-Reliability Cognitive Load Fusion  (Week 4)

    CLI_raw = 50 * sum_i( w_i * q_i * r_i ) / sum_i( w_i * q_i )

    r_i  = personal-baseline ratio of signal i (fatigue, distraction, stress), clipped to SIGNAL_CLIP
    w_i  = calibrated weight (inverse-variability, learnt from the dataset by calibrate_fusion.py)
           multiplied by the monitoring-mode bias (exam / interview)
    q_i  = per-frame reliability (0..1). E.g. gaze is less trustworthy when the head is turned,
           blink detection is less trustworthy when the face is very small.

At baseline every r_i = 1, so CLI_raw = 50. CLI is then smoothed with an EMA.
"""
import numpy as np

import config

SIGNALS = ("fatigue", "distraction", "stress")


def frame_reliability(yaw, pitch, face_size_px):
    """Per-frame reliability for each signal, in [0.2, 1]."""
    turn = max(abs(yaw) / 45.0, abs(pitch) / 40.0)       # 1 = strongly turned
    gaze_q = float(np.clip(1.0 - turn, 0.2, 1.0))
    size_q = float(np.clip(face_size_px / 120.0, 0.2, 1.0))  # small/far face -> noisy EAR
    return {"fatigue": size_q, "distraction": gaze_q * size_q, "stress": 1.0}


def fuse(ratios, weights, reliability=None, clip=config.SIGNAL_CLIP):
    """Return raw CLI from baseline ratios."""
    reliability = reliability or {k: 1.0 for k in SIGNALS}
    num = den = 0.0
    for k in SIGNALS:
        r = float(np.clip(ratios[k], 0.0, clip))
        wq = weights[k] * reliability[k]
        num += wq * r
        den += wq
    return 50.0 * num / den if den > 0 else 50.0


def calibrate_weights(fatigue, distraction, stress, prior=None):
    """Data-driven weights: w_i proportional to prior_i / std(r_i).
    Signals that swing wildly (e.g. head jitter) are down-weighted so no single
    signal dominates the index. Returns normalised dict."""
    prior = prior or config.DEFAULT_WEIGHTS
    series = {"fatigue": fatigue, "distraction": distraction, "stress": stress}
    raw = {}
    for k, v in series.items():
        v = np.clip(np.asarray(v, dtype=float), 0, config.SIGNAL_CLIP)
        sd = float(np.std(v))
        raw[k] = prior[k] / max(sd, 1e-3)
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}
