import cv2
import numpy as np
import mediapipe as mp

# Basic eye landmark sets (FaceMesh indices)
# (Compact set near pupils; enough for a minimal demo)
LEFT_EYE_IDX  = [33, 133, 159, 145]   # outer/inner + top/bottom-ish
RIGHT_EYE_IDX = [362, 263, 386, 374]

class FeatureExtractor:
    def __init__(self, static_image_mode=False, max_num_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # gives iris landmarks when possible
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def close(self):
        self.mesh.close()

    def _landmarks_to_np(self, landmarks, w, h):
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
        return pts

    def _eye_center(self, all_pts, idxs):
        sel = all_pts[idxs]
        return sel.mean(axis=0)  # (x, y)

    def features_from_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None, None

        face = res.multi_face_landmarks[0]
        pts = self._landmarks_to_np(face.landmark, w, h)

        lc = self._eye_center(pts, LEFT_EYE_IDX)
        rc = self._eye_center(pts, RIGHT_EYE_IDX)

        # Interocular normalization (makes features robust to scale & distance)
        interocular = np.linalg.norm(rc - lc) + 1e-6
        mid = (lc + rc) / 2.0

        # Features: left/right centers relative to midpoint, scaled by interocular
        feat = np.array([
            (lc[0] - mid[0]) / interocular,
            (lc[1] - mid[1]) / interocular,
            (rc[0] - mid[0]) / interocular,
            (rc[1] - mid[1]) / interocular,
        ], dtype=np.float32)

        debug = {"lc": lc, "rc": rc, "mid": mid, "interocular": interocular}
        return feat, debug
