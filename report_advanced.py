import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import analysis_common
import config

data, extended = analysis_common.load()
total = len(data)
duration = float(data["Time"].iloc[-1])
fps = total / duration if duration > 0 else 0

# ---------------- SUMMARY ----------------
pct = {p: (data["Phase"] == p).mean() * 100 for p in ["Warm-up", "Focused", "Overload", "Fatigue"]}
avg_cli, max_cli = data["CognitiveLoad"].mean(), data["CognitiveLoad"].max()
stability = max(0.0, 100 - 2 * float(np.std(data["CognitiveLoad"])))
focus_runs = analysis_common.phase_runs(data, "Focused")
max_focus = max(focus_runs) if focus_runs else 0.0
weights = config.load_weights(None if data["Mode"].iloc[0] == "legacy" else data["Mode"].iloc[0])

events = pd.DataFrame()
ev_path = os.path.join(config.BASE_DIR, "session_events.csv")
if extended and os.path.exists(ev_path):
    events = pd.read_csv(ev_path)
attention = (1 - data["LookingAway"].mean()) * 100 if extended else None

# ---------------- GRAPHS ----------------
plt.figure(figsize=(9, 4))
if "CLI_original" in data:
    plt.plot(data["Time"], data["CLI_original"], color="grey", alpha=0.5, lw=1, label="Original fixed-weight CLI")
plt.plot(data["Time"], data["CognitiveLoad"], lw=2, label="AWR-CLF (smoothed)")
for upper, name, _ in config.PHASES[:-1]:
    plt.axhline(upper, ls="--", lw=0.8, color="k", alpha=0.4)
plt.title("Cognitive Load Over Time")
plt.xlabel("Time (s)")
plt.ylabel("CLI")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cli_graph.png", dpi=150)
plt.close()

cols = ["BlinkRate", "Stress", "Distraction", "CognitiveLoad"]
corr = data[cols].corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(4), cols, rotation=45, ha="right")
plt.yticks(range(4), cols)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("corr_graph.png", dpi=150)
plt.close()

if extended:
    plt.figure(figsize=(9, 3.5))
    plt.plot(data["Time"], data["Yaw"], label="Yaw")
    plt.plot(data["Time"], data["Pitch"], label="Pitch")
    plt.ylabel("degrees")
    plt.xlabel("Time (s)")
    plt.title("Head Pose")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pose_graph.png", dpi=150)
    plt.close()

# ---------------- INSIGHTS ----------------
insights = []
if pct["Overload"] > 30:
    insights.append(f"Frequent cognitive overload ({pct['Overload']:.1f}% of session).")
if pct["Fatigue"] > 20:
    insights.append(f"Fatigue detected ({pct['Fatigue']:.1f}% of session).")
if stability < 60:
    insights.append("Low cognitive stability (large swings in load).")
if max_focus < 10:
    insights.append("Short continuous focus periods (under 10 s).")
if extended and attention is not None and attention < 80:
    label = "on-screen attention" if data["Mode"].iloc[0] == "exam" else "eye contact"
    insights.append(f"Low {label}: {attention:.1f}%.")
if not insights:
    insights.append("Session stayed mostly within the focused range.")

# ---------------- PDF ----------------
doc = SimpleDocTemplate("Advanced_Cognitive_Report.pdf")
st = getSampleStyleSheet()
el = [Paragraph("Advanced Cognitive Analysis Report", st["Title"]), Spacer(1, 12)]

mode = data["Mode"].iloc[0]
mode_title = config.MODES[mode]["title"] if mode in config.MODES else "Recorded dataset (pre-mode)"
rows = [["Metric", "Value"],
        ["Monitoring mode", mode_title],
        ["Duration / frames", f"{duration:.1f} s / {total} frames ({fps:.1f} fps)"],
        ["Focused / Overload / Fatigue", f"{pct['Focused']:.1f}% / {pct['Overload']:.1f}% / {pct['Fatigue']:.1f}%"],
        ["Average / Peak CLI", f"{avg_cli:.1f} / {max_cli:.1f}"],
        ["Cognitive stability", f"{stability:.1f}%"],
        ["Longest continuous focus", f"{max_focus:.1f} s"],
        ["AWR-CLF weights (F / D / S)",
         f"{weights['fatigue']:.2f} / {weights['distraction']:.2f} / {weights['stress']:.2f}"]]
if extended:
    rows += [["Attention (on-screen / eye contact)", f"{attention:.1f}%"],
             ["Monitoring events", str(len(events))],
             ["Mean |yaw| / |pitch|", f"{data['Yaw'].abs().mean():.1f} / {data['Pitch'].abs().mean():.1f} deg"]]
tbl = Table(rows, colWidths=[200, 260])
tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                         ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                         ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
el += [Paragraph("Session Summary", st["Heading2"]), tbl, Spacer(1, 12)]

el.append(Paragraph("Insights", st["Heading2"]))
el += [Paragraph(f"- {i}", st["Normal"]) for i in insights]
el.append(Spacer(1, 12))

if not events.empty:
    el.append(Paragraph("Monitoring Events", st["Heading2"]))
    t0 = pd.read_csv(config.DATA_FILE, usecols=["Time"])["Time"].iloc[0]
    erows = [["Time (s)", "Event", "Detail"]] + [[f"{r.Time - t0:.1f}", r.Event, str(r.Detail)]
                                                 for r in events.itertuples()][:25]
    et = Table(erows, colWidths=[70, 220, 170])
    et.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    el += [et, Spacer(1, 12)]

el += [Paragraph("Cognitive Load Trend", st["Heading2"]), Image("cli_graph.png", width=460, height=204)]
el += [PageBreak(), Paragraph("Correlation Analysis", st["Heading2"]), Image("corr_graph.png", width=360, height=300)]
if extended:
    el += [Spacer(1, 12), Paragraph("Head Pose", st["Heading2"]), Image("pose_graph.png", width=460, height=179)]

el += [Spacer(1, 12), Paragraph("Method note", st["Heading2"]), Paragraph(
    "CLI is computed with AWR-CLF: personal-baseline ratios of blink rate, gaze deviation and head "
    "micro-movement are fused with dataset-calibrated inverse-variability weights, scaled by per-frame "
    "reliability and smoothed with an exponential moving average. CLI = 50 at the personal baseline. "
    "It is a heuristic composite index, not a clinically validated measure.", st["Normal"])]

doc.build(el)
print("Advanced PDF Generated: Advanced_Cognitive_Report.pdf")
print(json.dumps({"focused": round(pct["Focused"], 2), "overload": round(pct["Overload"], 2),
                  "fatigue": round(pct["Fatigue"], 2), "avg_cli": round(avg_cli, 2),
                  "peak_cli": round(max_cli, 2), "stability": round(stability, 2),
                  "max_focus_s": round(max_focus, 2)}, indent=2))
