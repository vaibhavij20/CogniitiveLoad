"""
Facial landmark extraction (Week 2).

Works with BOTH MediaPipe APIs:
  * mediapipe <= 0.10.14 : mp.solutions.face_mesh (legacy)
  * newer mediapipe      : mp.tasks FaceLandmarker (model file auto-downloaded once)
"""
import os
import time
import urllib.request

import numpy as np

import config


class LandmarkDetector:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.backend = "legacy"
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            self.backend = "tasks"
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._ensure_model()
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=config.MODEL_FILE),
                running_mode=vision.RunningMode.VIDEO, num_faces=1)
            self.mesh = vision.FaceLandmarker.create_from_options(opts)
            self._t0 = time.time()
            self._last_ts = -1

    @staticmethod
    def _ensure_model():
        if not os.path.exists(config.MODEL_FILE):
            print("Downloading face landmark model (one time, ~4 MB)...")
            urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_FILE)

    def detect(self, rgb):
        """rgb: HxWx3 uint8. Returns (N,2) array of pixel coords or None if no face."""
        h, w = rgb.shape[:2]
        if self.backend == "legacy":
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
        else:
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts = int((time.time() - self._t0) * 1000)
            ts = max(ts, self._last_ts + 1)          # timestamps must strictly increase
            self._last_ts = ts
            res = self.mesh.detect_for_video(img, ts)
            if not res.face_landmarks:
                return None
            lm = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=float)

    def close(self):
        try:
            self.mesh.close()
        except Exception:
            pass
