"""
Central configuration for the Cognitive Load Monitoring System.
All tunable constants live here so every script uses the same values.
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cognitive_data.csv")
WEIGHTS_FILE = os.path.join(BASE_DIR, "weights.json")
MODEL_FILE = os.path.join(BASE_DIR, "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# ---------------- LANDMARK INDICES (MediaPipe Face Mesh, 478 points) ----------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]      # p1..p6 for EAR
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# 6 points used for head pose (solvePnP): nose tip, chin, eye corners, mouth corners
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
# Generic 3D face model (mm) matching POSE_LANDMARKS order
POSE_MODEL_3D = [
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye outer corner
    (43.3, 32.7, -26.0),      # right eye outer corner
    (-28.9, -28.9, -24.1),    # left mouth corner
    (28.9, -28.9, -24.1),     # right mouth corner
]

# ---------------- BLINK ----------------
EAR_CLOSE = 0.22      # eye considered closed below this
EAR_OPEN = 0.25       # hysteresis: eye re-opened above this
BLINK_WINDOW = 60     # seconds, blink rate = blinks per minute

# ---------------- TEMPORAL ----------------
CALIBRATION_TIME = 60     # seconds of personal baseline
JITTER_WINDOW = 20        # frames for micro-movement
PHASE_WINDOW = 30         # frames averaged for phase decision
EMA_ALPHA = 0.15          # temporal smoothing factor for CLI (0..1, lower = smoother)
SIGNAL_CLIP = 4.0         # cap each baseline ratio at 4x so one glitch can't explode CLI

# ---------------- PHASES (applied to smoothed CLI, baseline = 50) ----------------
PHASES = [  # (upper bound, name, BGR colour)
    (30, "Warm-up", (0, 255, 0)),
    (60, "Focused", (255, 255, 0)),
    (80, "Overload", (0, 165, 255)),
    (float("inf"), "Fatigue", (0, 0, 255)),
]


def get_phase(cli):
    for upper, name, _ in PHASES:
        if cli < upper:
            return name
    return PHASES[-1][1]


def phase_colour(name):
    for _, n, c in PHASES:
        if n == name:
            return c
    return (255, 255, 255)


# ---------------- MONITORING MODES ----------------
# Default (prior) weights for AWR-CLF. calibrate_fusion.py learns data-driven
# reliability weights and stores them in weights.json; modes then bias them.
DEFAULT_WEIGHTS = {"fatigue": 0.4, "distraction": 0.3, "stress": 0.3}

MODES = {
    "exam": {
        "title": "Examination Monitoring",
        # exam: looking away matters most
        "bias": {"fatigue": 0.8, "distraction": 1.5, "stress": 0.8},
        "yaw_limit": 25.0,       # degrees, head turned away from screen
        "pitch_limit": 20.0,
        "gaze_limit": 0.35,      # normalised iris offset, looking sideways
        "away_seconds": 2.0,     # sustained look-away to raise an event
        "no_face_seconds": 3.0,  # face missing to raise an event
    },
    "interview": {
        "title": "Interview Monitoring",
        # interview: nervousness (micro-movement) and fatigue matter most
        "bias": {"fatigue": 1.2, "distraction": 0.8, "stress": 1.3},
        "yaw_limit": 30.0,
        "pitch_limit": 25.0,
        "gaze_limit": 0.40,
        "away_seconds": 3.0,
        "no_face_seconds": 5.0,
    },
}

# ---------------- REAL-TIME PERFORMANCE ----------------
PROCESS_WIDTH = 640       # frames are resized to this width before inference
PROCESS_EVERY = 1         # run landmark model every N frames (2 = ~2x faster)


def load_weights(mode=None):
    """Return fusion weights: calibrated (weights.json) or default, biased by mode, normalised."""
    w = dict(DEFAULT_WEIGHTS)
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                w.update(json.load(f)["weights"])
        except (OSError, KeyError, ValueError):
            pass
    if mode in MODES:
        for k, b in MODES[mode]["bias"].items():
            w[k] *= b
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}
