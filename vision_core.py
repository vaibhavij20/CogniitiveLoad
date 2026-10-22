"""
Real-time Cognitive Load Monitoring (webcam).

Usage:
    python vision_core.py                      # asks for mode
    python vision_core.py --mode exam
    python vision_core.py --mode interview --process-every 2
    python vision_core.py --video recording.mp4 --mode exam   # run on a recorded video

Keys:  Esc / q = stop and save     p = pause/resume
Outputs: cognitive_data.csv (per-frame signals), session_events.csv, session_summary.json
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import cv2

import config
from landmarks import LandmarkDetector
from pipeline import CognitivePipeline

CSV_COLUMNS = ["Time", "Mode", "EAR", "BlinkRate", "Gaze", "Yaw", "Pitch", "Roll", "Movement",
               "Fatigue", "Distraction", "Stress", "CLI_raw", "CognitiveLoad", "Phase",
               "LookingAway", "FPS"]


def parse_args():
    ap = argparse.ArgumentParser(description="Cognitive load monitor")
    ap.add_argument("--mode", choices=list(config.MODES), help="exam or interview")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 fails)")
    ap.add_argument("--video", help="optional video file instead of webcam")
    ap.add_argument("--calibration", type=int, default=config.CALIBRATION_TIME, help="seconds")
    ap.add_argument("--process-every", type=int, default=config.PROCESS_EVERY,
                    help="run landmark model every N frames (higher = faster)")
    ap.add_argument("--width", type=int, default=config.PROCESS_WIDTH, help="processing width")
    ap.add_argument("--out", default=config.DATA_FILE, help="output CSV")
    a = ap.parse_args()
    if a.mode is None:
        choice = input("Select mode: [1] Examination  [2] Interview  > ").strip()
        a.mode = "interview" if choice == "2" else "exam"
    return a


def draw_text(frame, text, y, colour=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def main():
    a = parse_args()
    src = a.video if a.video else a.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open {'video ' + a.video if a.video else 'camera ' + str(a.camera)}. "
                 "Close other apps using the webcam, check permissions, or try --camera 1.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = LandmarkDetector()
    pipe = CognitivePipeline(a.mode, a.calibration)
    print(f"Mode: {config.MODES[a.mode]['title']}  |  backend: {detector.backend}  |  "
          f"weights: { {k: round(v, 3) for k, v in pipe.weights.items()} }")

    if os.path.exists(a.out):
        shutil.copy(a.out, os.path.splitext(a.out)[0] + "_backup.csv")
    logfile = open(a.out, "w", newline="")
    writer = csv.DictWriter(logfile, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    frame_i, pts, scale = 0, None, 1.0
    fps, t_prev = 0.0, time.time()
    paused = False
    banner, banner_until = "", 0.0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    win = "Cognitive Load Monitor"

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                if not a.video:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = (frame_i / video_fps) if a.video else time.time()

                # ---- real-time optimisation: downscale + frame skipping ----
                if frame_i % max(a.process_every, 1) == 0:
                    scale = min(1.0, a.width / w)
                    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
                    det = detector.detect(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    pts = det / scale if det is not None else None
                frame_i += 1

                t = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(t - t_prev, 1e-6)) if fps else 1.0 / max(t - t_prev, 1e-6)
                t_prev = t

                if pts is None:
                    ev = pipe.no_face(now)
                    if ev:
                        banner, banner_until = ev, time.time() + 3
                    draw_text(frame, "No face detected", 40, (0, 0, 255), 0.9)
                else:
                    r = pipe.process(pts, w, h, now)
                    for i in config.LEFT_EYE + config.RIGHT_EYE + config.POSE_LANDMARKS:
                        cv2.circle(frame, tuple(int(v) for v in pts[i]), 2, (0, 255, 0), -1)

                    if r["calibrating"]:
                        draw_text(frame, f"Calibrating... {int(r['elapsed'])}/{a.calibration}s  look at the screen naturally",
                                  40, (0, 255, 255), 0.7)
                    else:
                        r["FPS"] = fps
                        writer.writerow(r)
                        col = config.phase_colour(pipe.phase)
                        draw_text(frame, f"CLI: {r['CognitiveLoad']:.0f}", 40, col, 1.0, 3)
                        draw_text(frame, f"Phase: {pipe.phase}", 80, col, 0.9)
                        draw_text(frame, f"Stability: {pipe.stability:.0f}%   Max focus: {pipe.max_focus:.0f}s", 115)
                        label = "On-screen" if a.mode == "exam" else "Eye contact"
                        draw_text(frame, f"{label}: {pipe.attention_pct:.0f}%   Events: {len(pipe.events)}", 145)
                        draw_text(frame, f"Yaw {r['Yaw']:+.0f}  Pitch {r['Pitch']:+.0f}  Blinks/min {r['BlinkRate']:.0f}", 175, scale=0.6)
                        if r["event"]:
                            banner, banner_until = r["event"], time.time() + 3

                if time.time() < banner_until:
                    draw_text(frame, f"ALERT: {banner}", h - 50, (0, 0, 255), 0.9)
                draw_text(frame, f"{config.MODES[a.mode]['title']}  |  FPS {fps:.0f}", h - 15, (200, 200, 200), 0.6, 1)
                cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                paused = not paused
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        logfile.close()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        save_session(pipe, a)


def save_session(pipe, a):
    base = os.path.dirname(os.path.abspath(a.out))
    with open(os.path.join(base, "session_events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "Mode", "Event", "Detail"])
        for t, e, d in pipe.events:
            w.writerow([t, pipe.mode, e, d])
    summary = {
        "mode": pipe.mode,
        "calibrated": not pipe.calibrating,
        "baseline": pipe.base,
        "weights": pipe.weights,
        "frames_logged": len(pipe.cli_all),
        "stability": round(pipe.stability, 2),
        "max_focus_seconds": round(pipe.max_focus, 2),
        "attention_percent": round(pipe.attention_pct, 2),
        "events": len(pipe.events),
    }
    with open(os.path.join(base, "session_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if pipe.calibrating:
        print("Stopped during calibration, so no data rows were logged.")
    print("Saved:", a.out, "session_events.csv, session_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
