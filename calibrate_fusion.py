"""
Week 4: calibrate the AWR-CLF fusion weights on the recorded signal dataset.

Reads cognitive_data.csv, learns inverse-variability weights so no single signal dominates,
compares old vs calibrated fusion, and saves weights.json (used automatically by vision_core.py).

    python calibrate_fusion.py
"""
import json
import sys

import numpy as np
import pandas as pd

import config
import fusion


def ratios_from_csv(df):
    """Baseline ratios for each signal. New CSVs store them directly; the original
    CSV stores BlinkRate raw, so it is divided by its median as the baseline."""
    if "Fatigue" in df.columns:
        fat = df["Fatigue"].to_numpy(float)
    else:
        fat = df["BlinkRate"].to_numpy(float) / max(df["BlinkRate"].median(), 1.0)
    return fat, df["Distraction"].to_numpy(float), df["Stress"].to_numpy(float)


def contribution_share(w, fat, dis, st):
    """Share of CLI variance attributable to each weighted signal."""
    parts = {"fatigue": w["fatigue"] * np.clip(fat, 0, config.SIGNAL_CLIP),
             "distraction": w["distraction"] * np.clip(dis, 0, config.SIGNAL_CLIP),
             "stress": w["stress"] * np.clip(st, 0, config.SIGNAL_CLIP)}
    v = {k: float(np.var(p)) for k, p in parts.items()}
    s = sum(v.values()) or 1.0
    return {k: x / s for k, x in v.items()}


def main(path=config.DATA_FILE):
    df = pd.read_csv(path).dropna(subset=["Distraction", "Stress"])
    if len(df) < 100:
        sys.exit("Need at least 100 rows to calibrate.")
    fat, dis, st = ratios_from_csv(df)

    old = config.DEFAULT_WEIGHTS
    new = fusion.calibrate_weights(fat, dis, st, prior=old)

    def cli(w):
        return np.array([fusion.fuse({"fatigue": a, "distraction": b, "stress": c}, w)
                         for a, b, c in zip(fat, dis, st)])
    cli_old, cli_new = cli(old), cli(new)

    print(f"Calibrated on {len(df)} rows from {path}\n")
    print(f"{'signal':<12}{'std':>8}{'old w':>9}{'new w':>9}{'old share':>11}{'new share':>11}")
    so, sn = contribution_share(old, fat, dis, st), contribution_share(new, fat, dis, st)
    for k, arr in zip(fusion.SIGNALS, (fat, dis, st)):
        print(f"{k:<12}{np.std(np.clip(arr, 0, config.SIGNAL_CLIP)):>8.3f}{old[k]:>9.3f}{new[k]:>9.3f}"
              f"{so[k]*100:>10.1f}%{sn[k]*100:>10.1f}%")
    print(f"\nCLI std  : old {cli_old.std():.2f}  ->  calibrated {cli_new.std():.2f}")
    print(f"CLI peak : old {cli_old.max():.1f}  ->  calibrated {cli_new.max():.1f}")

    with open(config.WEIGHTS_FILE, "w") as f:
        json.dump({"method": "AWR-CLF inverse-variability calibration",
                   "rows": int(len(df)), "source": path,
                   "weights": new, "contribution_share": sn}, f, indent=2)
    print(f"\nSaved {config.WEIGHTS_FILE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.DATA_FILE)
