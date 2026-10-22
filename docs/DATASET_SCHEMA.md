# Dataset Schema (Week 1)

## cognitive_data.csv (one row per processed frame, written after calibration)
| Column | Type | Unit / range | Description |
|---|---|---|---|
| Time | float | Unix seconds (seconds from start for `--video`) | Frame timestamp |
| Mode | str | exam / interview | Monitoring mode |
| EAR | float | about 0.1 to 0.4 | Mean eye aspect ratio of both eyes |
| BlinkRate | float | blinks/min | Blinks in the last 60 s |
| Gaze | float | 0 to about 0.5 | Horizontal iris offset divided by eye width (0 = centred) |
| Yaw, Pitch, Roll | float | degrees | Head pose from solvePnP (0 = facing the camera) |
| Movement | float | arbitrary units | Head micro-movement (nose displacement plus angular change) |
| Fatigue | float | ratio | Blink rate relative to baseline |
| Distraction | float | ratio | Gaze relative to baseline |
| Stress | float | ratio | Movement relative to baseline |
| CLI_raw | float | 0 to 200 | AWR-CLF output before smoothing (50 = baseline) |
| CognitiveLoad | float | 0 to 200 | Smoothed CLI (EMA) |
| Phase | str | Warm-up / Focused / Overload / Fatigue | Phase from the 30-frame CLI average |
| LookingAway | int | 0 / 1 | Head or eyes outside the mode's limits in this frame |
| FPS | float | frames/s | Processing speed |

The original recorded file used only `Time, BlinkRate, Stress, Distraction, CognitiveLoad`.
The analysis scripts accept both formats.

## session_events.csv
`Time, Mode, Event, Detail`. Each row is one event, for example `Looking away from screen, left`.

## session_summary.json
Records the mode, calibration status, personal baseline, fusion weights, stability, longest
focus period, attention % and event count.

## weights.json
Records the calibrated AWR-CLF weights, the number of rows used, and each signal's variance share.
