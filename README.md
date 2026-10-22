# Webcam-based Cognitive Load Monitoring (AWR-CLF)

A real-time system that estimates cognitive load from a webcam using facial landmarks.
It combines blink rate, gaze deviation and head micro-movement into a Cognitive Load Index (CLI).
The fusion method is AWR-CLF. The system has two monitoring modes, Examination and Interview.

## Quick start (VS Code)

1. File > Open Folder > `MV_PROJECT`
2. Open a terminal (Ctrl + `):
   ```
   pip install -r requirements.txt
   ```
   Python 3.9 to 3.13 is supported. The code works with both the old MediaPipe API
   (`mp.solutions`, versions up to 0.10.14) and the new Tasks API. With the new API, the
   face model (`face_landmarker.task`, about 4 MB) downloads automatically on the first run.
3. Run the live monitor:
   ```
   python vision_core.py --mode exam
   python vision_core.py --mode interview
   ```
   Look at the screen normally for the 60 s calibration. Press **Esc** or **q** to stop,
   and **p** to pause.
4. Analyse the session and generate the PDF:
   ```
   python run_analysis.py
   ```
5. Run the tests (no webcam needed):
   ```
   python -m unittest discover -s tests -v
   ```

You can also use the Run and Debug panel (Ctrl+Shift+D). It has ready-made configurations for
the Exam monitor, the Interview monitor, the Analysis + report, and the tests.

### Useful options
| Option | Purpose |
|---|---|
| `--camera 1` | use a different webcam |
| `--process-every 2` | run the landmark model every 2nd frame (about 2x faster on slow laptops) |
| `--width 480` | process smaller frames for more speed |
| `--calibration 30` | shorter calibration |
| `--video file.mp4` | run on a recorded video instead of the webcam |

## Project structure

| File | Week | Role |
|---|---|---|
| `docs/LITERATURE_REVIEW.md`, `docs/REQUIREMENTS.md`, `docs/DATASET_SCHEMA.md` | 1 | Literature review, requirement analysis, dataset schema |
| `landmarks.py` | 2 | MediaPipe 478-point face mesh with iris landmarks (old and new API) |
| `signals.py` | 2, 3, 4 | EAR blink detection, gaze offset, head pose (solvePnP), micro-movement, EMA smoothing |
| `fusion.py` | 4 | AWR-CLF fusion and weight calibration |
| `calibrate_fusion.py` | 4 | Learns the fusion weights from `cognitive_data.csv` and writes `weights.json` |
| `pipeline.py` | 2 to 5 | Per-frame pipeline: baseline, fusion, phase, Exam/Interview events |
| `vision_core.py` | 5 | Real-time app: webcam, overlay, frame skipping, downscaling, FPS, logging |
| `config.py` | all | Every threshold, weight and mode setting in one place |
| `correlation_analysis.py`, `time_analysis.py`, `report_advanced.py`, `run_analysis.py` | 6 | Analysis, plots and the PDF report |
| `tests/test_pipeline.py` | 6 | 18 unit and behavioural validation tests |

## How it works

```
webcam -> face mesh (478 pts) -> EAR -> blinks/min           -> fatigue ratio     \
                              -> iris offset / eye width      -> distraction ratio  > AWR-CLF -> EMA -> CLI -> phase
                              -> solvePnP yaw/pitch/roll      -> look-away events  |
                              -> nose + pose frame-to-frame   -> stress ratio      /
```

1. **Calibration (60 s).** The system records a personal baseline for blink rate, gaze, movement
   and the natural head yaw and pitch.
2. **Ratios.** Each signal is expressed relative to its baseline: `(x + eps) / (baseline + eps)`.
   The ratio is 1 at baseline and is clipped at 4x, so a single glitch cannot explode the index.
3. **AWR-CLF (Adaptive Weighted-Reliability Cognitive Load Fusion).**
   `CLI = 50 * sum(w_i * q_i * r_i) / sum(w_i * q_i)`
   * `w_i` are calibrated weights. `calibrate_fusion.py` sets each signal's weight inversely to
     its variability in the dataset, so no single signal dominates. On the recorded dataset, head
     jitter originally caused 79% of the CLI variance. After calibration, the shares are
     47 / 27 / 27 %.
   * Each mode scales the weights differently. Exam mode up-weights distraction, and Interview
     mode up-weights stress and fatigue.
   * `q_i` is the per-frame reliability. Gaze is trusted less when the head is turned, and blink
     detection is trusted less when the face is small or far away.
4. **Temporal smoothing.** An EMA (alpha 0.15) is applied to the CLI. The phase is decided on a
   30-frame average: Warm-up < 30 <= Focused < 60 <= Overload < 80 <= Fatigue.
5. **Modes.**
   * *Examination:* raises "Looking away from screen" (head turned or eyes sideways for more than
     2 s) and "Candidate left frame" (no face for more than 3 s), and reports the on-screen
     attention %.
   * *Interview:* raises "Broke eye contact" (more than 3 s) and "Face not visible", and reports
     the eye-contact %.

## Outputs
* `cognitive_data.csv` has one row per frame. The schema is in `docs/DATASET_SCHEMA.md`. The
  previous file is copied to `cognitive_data_backup.csv`.
* `session_events.csv` lists the monitoring events, and `session_summary.json` holds the
  baseline, weights and summary metrics.
* `Advanced_Cognitive_Report.pdf`, `cli_graph.png`, `corr_graph.png`, and `pose_graph.png` for
  new sessions.

The analysis scripts read both the original 5-column CSV and the new extended format. For the
original CSV, the CLI is recomputed with calibrated AWR-CLF so you can compare it with the old
fixed-weight index.

## Limitations
* CLI is a heuristic composite index built from behavioural proxies. It has not been clinically
  validated. The tests check that the system responds in the expected direction; they do not
  measure accuracy against a ground-truth cognitive load.
* Head pose uses a generic 3D face model and an approximate camera focal length (the frame
  width), so the angles are approximate. They are good enough for look-away detection.
* Lighting, glasses and a low camera angle can affect EAR and iris tracking.
