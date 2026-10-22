"""
Per-frame cognitive pipeline (Weeks 2-5), independent of the camera so it can be unit-tested.

landmarks -> EAR/blinks, gaze, head pose, micro-movement
          -> personal baseline (calibration)
          -> AWR-CLF fusion -> EMA smoothing -> phase
          -> mode-specific events (Examination / Interview)
"""
from collections import deque

import numpy as np

import config
import fusion
import signals


class CognitivePipeline:
    EPS = {"blink": 2.0, "gaze": 0.03, "move": 0.1}   # noise floors for baseline ratios

    def __init__(self, mode="exam", calibration_time=config.CALIBRATION_TIME):
        if mode not in config.MODES:
            raise ValueError(f"mode must be one of {list(config.MODES)}")
        self.mode = mode
        self.cfg = config.MODES[mode]
        self.weights = config.load_weights(mode)
        self.calibration_time = calibration_time

        self.blinks = signals.BlinkDetector()
        self.motion = signals.MicroMovement()
        self.ema = signals.EMA()
        self.phase_hist = deque(maxlen=config.PHASE_WINDOW)

        self.start = None
        self.calibrating = True
        self._cal = {"blink": [], "gaze": [], "move": [], "yaw": [], "pitch": []}
        self.base = {"blink": 1.0, "gaze": 1.0, "move": 1.0, "yaw": 0.0, "pitch": 0.0}

        self.phase = "Calibrating"
        self.cli_all = []
        self.focus_start = None
        self.max_focus = 0.0
        self.away_since = None
        self.away_flagged = False
        self.noface_since = None
        self.noface_flagged = False
        self.events = []          # (time, event, detail)
        self.look_away_frames = 0
        self.tracked_frames = 0

    # ------------------------------------------------------------------
    def no_face(self, now):
        """Call when no face is detected in a frame."""
        if self.start is None:
            return None
        if self.noface_since is None:
            self.noface_since = now
            self.noface_flagged = False
        elif not self.noface_flagged and now - self.noface_since >= self.cfg["no_face_seconds"]:
            self.noface_flagged = True
            msg = "Candidate left frame" if self.mode == "exam" else "Face not visible"
            self._event(now, msg, f"> {self.cfg['no_face_seconds']:.0f}s")
            return msg
        return None

    def _event(self, now, name, detail=""):
        self.events.append((now, name, detail))

    # ------------------------------------------------------------------
    def process(self, pts, frame_w, frame_h, now):
        """pts: (N>=478, 2) pixel landmarks. Returns dict of everything for display/logging."""
        if self.start is None:
            self.start = now
        self.noface_since = None
        elapsed = now - self.start

        le, re_ = pts[config.LEFT_EYE], pts[config.RIGHT_EYE]
        ear = (signals.eye_aspect_ratio(le) + signals.eye_aspect_ratio(re_)) / 2.0
        blink_rate = self.blinks.update(ear, now)

        gaze = (signals.gaze_offset(le, pts[config.LEFT_IRIS].mean(axis=0)) +
                signals.gaze_offset(re_, pts[config.RIGHT_IRIS].mean(axis=0))) / 2.0

        yaw, pitch, roll = signals.head_pose(pts[config.POSE_LANDMARKS], frame_w, frame_h)
        face_size = float(np.linalg.norm(pts[33] - pts[263]))   # outer eye-corner distance
        move = self.motion.update(pts[1], face_size, yaw, pitch, roll)

        out = {"Time": now, "Mode": self.mode, "EAR": ear, "BlinkRate": round(blink_rate, 2),
               "Gaze": gaze, "Yaw": yaw, "Pitch": pitch, "Roll": roll, "Movement": move,
               "calibrating": self.calibrating, "elapsed": elapsed, "event": ""}

        if self.calibrating:
            for k, v in (("blink", blink_rate), ("gaze", gaze), ("move", move), ("yaw", yaw), ("pitch", pitch)):
                self._cal[k].append(v)
            if elapsed >= self.calibration_time:
                self._finish_calibration()
            return out

        # ---- baseline-normalised ratios ----
        rel_yaw, rel_pitch = yaw - self.base["yaw"], pitch - self.base["pitch"]
        # (x + eps) / (baseline + eps): exactly 1 at baseline, and a near-zero baseline can't explode the ratio
        e = self.EPS
        ratios = {"fatigue": (blink_rate + e["blink"]) / (self.base["blink"] + e["blink"]),
                  "distraction": (gaze + e["gaze"]) / (self.base["gaze"] + e["gaze"]),
                  "stress": (move + e["move"]) / (self.base["move"] + e["move"])}
        rel = fusion.frame_reliability(rel_yaw, rel_pitch, face_size)
        cli_raw = fusion.fuse(ratios, self.weights, rel)
        cli = self.ema.update(cli_raw)
        self.cli_all.append(cli)

        # ---- phase (window average of smoothed CLI) ----
        self.phase_hist.append(cli)
        self.phase = config.get_phase(float(np.mean(self.phase_hist)))

        # ---- attention span ----
        if self.phase == "Focused":
            if self.focus_start is None:
                self.focus_start = now
            self.max_focus = max(self.max_focus, now - self.focus_start)
        else:
            self.focus_start = None

        # ---- Week 5: mode-specific monitoring ----
        away = (abs(rel_yaw) > self.cfg["yaw_limit"] or abs(rel_pitch) > self.cfg["pitch_limit"]
                or gaze > self.cfg["gaze_limit"])
        self.tracked_frames += 1
        if away:
            self.look_away_frames += 1
            if self.away_since is None:
                self.away_since, self.away_flagged = now, False
            elif not self.away_flagged and now - self.away_since >= self.cfg["away_seconds"]:
                self.away_flagged = True
                name = "Looking away from screen" if self.mode == "exam" else "Broke eye contact"
                direction = ("left" if rel_yaw < 0 else "right") if abs(rel_yaw) > self.cfg["yaw_limit"] else \
                            ("down" if rel_pitch > 0 else "up") if abs(rel_pitch) > self.cfg["pitch_limit"] else "eyes sideways"
                self._event(now, name, direction)
                out["event"] = name
        else:
            self.away_since = None

        out.update({"Fatigue": ratios["fatigue"], "Distraction": ratios["distraction"],
                    "Stress": ratios["stress"], "CLI_raw": cli_raw, "CognitiveLoad": cli,
                    "Phase": self.phase, "LookingAway": int(away)})
        return out

    def _finish_calibration(self):
        c = self._cal
        self.base = {
            # blink rate at the end of calibration = blinks over the whole calibration period (per minute);
            # averaging the ramp-up would underestimate it.
            "blink": float(c["blink"][-1]),
            "gaze": float(np.mean(c["gaze"])),
            "move": float(np.mean(c["move"])),
            "yaw": float(np.median(c["yaw"])),
            "pitch": float(np.median(c["pitch"])),
        }
        self.calibrating = False
        self.phase = "Focused"

    # ------------------------------------------------------------------
    @property
    def stability(self):
        if len(self.cli_all) < 10:
            return 100.0
        return max(0.0, 100.0 - 2.0 * float(np.std(self.cli_all)))

    @property
    def attention_pct(self):
        """Exam: % of time on screen. Interview: % eye contact."""
        if not self.tracked_frames:
            return 100.0
        return 100.0 * (1 - self.look_away_frames / self.tracked_frames)
