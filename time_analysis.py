import matplotlib.pyplot as plt

import analysis_common

data, extended = analysis_common.load()
t = data["Time"]

fig, axes = plt.subplots(2 if extended else 1, 1, figsize=(12, 8 if extended else 6), sharex=True, squeeze=False)
ax = axes[0, 0]

# Phase shading (contiguous blocks, fast)
colours = {"Focused": "green", "Overload": "orange", "Fatigue": "red"}
block = (data["Phase"] != data["Phase"].shift()).cumsum()
for _, g in data.groupby(block):
    p = g["Phase"].iloc[0]
    if p in colours:
        ax.axvspan(g["Time"].iloc[0], g["Time"].iloc[-1], color=colours[p], alpha=0.12, lw=0)

if "CLI_original" in data:
    ax.plot(t, data["CLI_original"], color="grey", alpha=0.5, lw=1, label="Original CLI (fixed weights)")
ax.plot(t, data["CLI_raw"], color="tab:blue", alpha=0.35, lw=1, label="AWR-CLF raw")
ax.plot(t, data["CognitiveLoad"], color="tab:blue", lw=2.2, label="AWR-CLF smoothed (EMA)")
ax.set_ylabel("Cognitive Load Index")
ax.set_title("Cognitive State Over Time  (green Focused, orange Overload, red Fatigue)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

if extended:
    ax2 = axes[1, 0]
    ax2.plot(t, data["Yaw"], label="Yaw (deg)")
    ax2.plot(t, data["Pitch"], label="Pitch (deg)")
    ax2.plot(t, data["Gaze"] * 100, label="Gaze offset x100")
    away = data["LookingAway"] == 1
    ax2.scatter(t[away], [0] * away.sum(), color="red", s=4, label="Looking away")
    ax2.set_ylabel("Head pose / gaze")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

axes[-1, 0].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()
