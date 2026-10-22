"""
Signal extraction from facial landmarks (pure functions, no camera needed, unit-testable).

Week 2: EAR + blink detection
Week 3: gaze deviation + head pose (solvePnP) + micro-movement
Week 4: temporal smoothing (EMA)
"""
import math
import time
from collections import deque

import numpy as np

import config


# ---------------- WEEK 2: EYE ASPECT RATIO + BLINKS ----------------
def eye_aspect_ratio(eye):
    """EAR (Soukupova & Cech, 2016). eye = 6x2 array p1..p6."""
    eye = np.asarray(eye, dtype=float)
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return float((a + b) / (2.0 * c)) if c > 0 else 0.0


class BlinkDetector:
    """Hysteresis blink counter. update() returns blinks per minute.
    Until a full window has passed, the count is scaled by the time actually observed,
    so the rate is not underestimated at the start (which would distort the baseline)."""

    MIN_SPAN = 10.0   # seconds before the rate is extrapolated

    def __init__(self, close=config.EAR_CLOSE, open_=config.EAR_OPEN, window=config.BLINK_WINDOW):
        self.close, self.open, self.window = close, open_, window
        self.closed = False
        self.times = deque()
        self.t0 = None

    @property
    def count(self):
        return len(self.times)

    def update(self, ear, now=None):
        now = time.time() if now is None else now
        if self.t0 is None:
            self.t0 = now
        if ear < self.close and not self.closed:
            self.times.append(now)
            self.closed = True
        elif ear > self.open:
            self.closed = False
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()
        span = min(max(now - self.t0, self.MIN_SPAN), self.window)
        return len(self.times) * self.window / span


# ---------------- WEEK 3: GAZE ----------------
def gaze_offset(eye, iris_center):
    """Horizontal iris offset normalised by eye width: 0 = centred, ~0.5 = at the corner.
    Normalising makes it independent of distance from the camera."""
    eye = np.asarray(eye, dtype=float)
    corner_a, corner_b = eye[0], eye[3]
    width = np.linalg.norm(corner_b - corner_a)
    if width == 0:
        return 0.0
    mid = (corner_a + corner_b) / 2.0
    return float(abs(iris_center[0] - mid[0]) / width)


# ---------------- WEEK 3: HEAD POSE ----------------
_MODEL_3D = np.array(config.POSE_MODEL_3D, dtype=np.float64)


def head_pose(points_2d, frame_w, frame_h):
    """Estimate (yaw, pitch, roll) in degrees from 6 image points via cv2.solvePnP."""
    import cv2
    pts = np.asarray(points_2d, dtype=np.float64)
    focal = frame_w
    cam = np.array([[focal, 0, frame_w / 2], [0, focal, frame_h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(rmat)


_FLIP = np.diag([1.0, -1.0, -1.0])   # 3D model is y-up, image is y-down: a face looking at the camera = 180 deg about x


def rotation_to_euler(r):
    """Rotation matrix from solvePnP -> (yaw, pitch, roll) in degrees, 0,0,0 = facing the camera.
    Removes the model/image axis flip, then decomposes R = Rz(roll) Ry(yaw) Rx(pitch)."""
    r = np.asarray(r) @ _FLIP
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        roll = 0.0
    return yaw, pitch, roll


class MicroMovement:
    """Head micro-movement = mean frame-to-frame change of nose position (px, normalised by
    face size) plus angular change of head pose (deg). Proxy for restlessness / stress."""

    def __init__(self, window=config.JITTER_WINDOW):
        self.pos = deque(maxlen=window)
        self.ang = deque(maxlen=window)

    def update(self, nose_xy, face_size, yaw, pitch, roll):
        self.pos.append(np.asarray(nose_xy, dtype=float) / max(face_size, 1e-6))
        self.ang.append(np.array([yaw, pitch, roll], dtype=float))
        if len(self.pos) < 2:
            return 0.0
        dp = np.linalg.norm(np.diff(np.array(self.pos), axis=0), axis=1).mean() * 100
        da = np.linalg.norm(np.diff(np.array(self.ang), axis=0), axis=1).mean()
        return float(dp + 0.5 * da)


# ---------------- WEEK 4: TEMPORAL SMOOTHING ----------------
class EMA:
    """Exponential moving average: s_t = a*x_t + (1-a)*s_{t-1}."""

    def __init__(self, alpha=config.EMA_ALPHA):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value
