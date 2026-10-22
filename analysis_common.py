"""Shared loading for the analysis scripts. Handles both the original 5-column CSV
and the new extended CSV written by vision_core.py."""
import numpy as np
import pandas as pd

import calibrate_fusion
import config
import fusion


def load(path=config.DATA_FILE):
    df = pd.read_csv(path)
    df = df.dropna(subset=["CognitiveLoad"]).reset_index(drop=True)
    df["Time"] = df["Time"] - df["Time"].iloc[0]
    if "Mode" not in df.columns:
        df["Mode"] = "legacy"
    extended = "CLI_raw" in df.columns

    if not extended:
        # Original data: logged CLI used the old fixed-weight formula without smoothing.
        # Recompute with calibrated AWR-CLF + EMA so the report reflects the new pipeline.
        df["CLI_original"] = df["CognitiveLoad"]
        fat, dis, st = calibrate_fusion.ratios_from_csv(df)
        w = config.load_weights()
        raw = np.array([fusion.fuse({"fatigue": a, "distraction": b, "stress": c}, w)
                        for a, b, c in zip(fat, dis, st)])
        df["CLI_raw"] = raw
        df["CognitiveLoad"] = pd.Series(raw).ewm(alpha=config.EMA_ALPHA, adjust=False).mean()

    smoothed_phase = df["CognitiveLoad"].rolling(config.PHASE_WINDOW, min_periods=1).mean()
    df["Phase"] = smoothed_phase.apply(config.get_phase)
    return df, extended


def phase_runs(df, phase):
    """Durations (seconds) of continuous runs of a phase, using the Time column."""
    runs, start, last = [], None, None
    for t, p in zip(df["Time"], df["Phase"]):
        if p == phase:
            start = t if start is None else start
            last = t
        elif start is not None:
            runs.append(last - start)
            start = None
    if start is not None:
        runs.append(last - start)
    return runs
