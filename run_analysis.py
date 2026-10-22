"""Runs the full offline analysis on cognitive_data.csv:
calibrate AWR-CLF weights -> correlations -> timeline -> PDF report."""
import subprocess
import sys

for f in ["calibrate_fusion.py", "correlation_analysis.py", "time_analysis.py", "report_advanced.py"]:
    print(f"\n===== Running {f} =====")
    subprocess.run([sys.executable, f], check=True)
print("\nAll done. Open Advanced_Cognitive_Report.pdf")
