# Requirement Analysis (Week 1)

## Functional requirements
| ID | Requirement | Implemented in |
|---|---|---|
| FR1 | Capture live webcam video, or a recorded video | `vision_core.py` (`--camera`, `--video`) |
| FR2 | Detect the face and extract 478 landmarks, including the iris | `landmarks.py` |
| FR3 | Detect blinks using EAR with hysteresis and report blinks per minute | `signals.BlinkDetector` |
| FR4 | Measure gaze deviation, normalised to eye width | `signals.gaze_offset` |
| FR5 | Estimate head pose (yaw, pitch, roll) | `signals.head_pose` |
| FR6 | Measure head micro-movement | `signals.MicroMovement` |
| FR7 | Calibrate a personal baseline at the start | `pipeline.CognitivePipeline` (60 s) |
| FR8 | Fuse the signals into a Cognitive Load Index with calibrated weights | `fusion.py`, `calibrate_fusion.py` |
| FR9 | Smooth the CLI over time and classify the phase | `signals.EMA`, `config.get_phase` |
| FR10 | Examination mode: detect look-away and leaving the frame | `pipeline.py`, `config.MODES["exam"]` |
| FR11 | Interview mode: track eye contact and nervousness | `pipeline.py`, `config.MODES["interview"]` |
| FR12 | Log per-frame data and events to CSV | `vision_core.py` |
| FR13 | Produce correlation and timeline analysis and a PDF report | `correlation_analysis.py`, `time_analysis.py`, `report_advanced.py` |

## Non-functional requirements
| ID | Requirement | How it is met |
|---|---|---|
| NFR1 | Real-time on a normal laptop CPU (at least 15 FPS) | Downscaled inference (`--width`), frame skipping (`--process-every`), single-face mesh, on-screen FPS counter |
| NFR2 | Robust to individual differences | Personal baseline ratios |
| NFR3 | Robust to tracking glitches | Ratio clipping at 4x, noise floors, EMA, per-frame reliability |
| NFR4 | Privacy | No video is stored; only numeric features are logged locally |
| NFR5 | Maintainable and configurable | All thresholds are in `config.py` |
| NFR6 | Testable without hardware | Pure signal functions and 18 synthetic tests |

## Hardware and software
A webcam (720p recommended), Python 3.9 to 3.13, OpenCV, MediaPipe, NumPy, pandas, Matplotlib and ReportLab.
