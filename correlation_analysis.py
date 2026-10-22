import matplotlib.pyplot as plt

import analysis_common

data, extended = analysis_common.load()

cols = ["BlinkRate", "Stress", "Distraction", "CognitiveLoad"]
if extended:
    cols = ["BlinkRate", "Gaze", "Yaw", "Pitch", "Movement", "Stress", "Distraction", "CognitiveLoad"]
corr = data[cols].corr()

print("Correlation Matrix:\n")
print(corr.round(3))

# Heatmap
plt.figure(figsize=(7, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
plt.yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Between Cognitive Signals")
plt.tight_layout()
plt.show()

# CLI vs each signal
plt.figure(figsize=(12, 4))
for k, (c, label) in enumerate([("BlinkRate", "Blink Rate"), ("Distraction", "Gaze Deviation (ratio)"),
                                ("Stress", "Head Micro-movement (ratio)")]):
    plt.subplot(1, 3, k + 1)
    plt.scatter(data[c], data["CognitiveLoad"], s=4, alpha=0.4)
    plt.xlabel(label)
    if k == 0:
        plt.ylabel("Cognitive Load")
    plt.title(f"{label} vs CLI")
plt.tight_layout()
plt.show()
