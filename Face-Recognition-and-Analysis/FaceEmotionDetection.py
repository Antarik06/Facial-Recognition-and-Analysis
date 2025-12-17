import cv2
import mediapipe as mp
import numpy as np

class EmotionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        self.LEFT = 61
        self.RIGHT = 291
        self.TOP = 13
        self.BOTTOM = 14

    def detect_emotion(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "No Face"

        h, w, _ = frame.shape
        lm = results.multi_face_landmarks[0].landmark

        left = np.array([lm[self.LEFT].x * w, lm[self.LEFT].y * h])
        right = np.array([lm[self.RIGHT].x * w, lm[self.RIGHT].y * h])
        top = np.array([lm[self.TOP].x * w, lm[self.TOP].y * h])
        bottom = np.array([lm[self.BOTTOM].x * w, lm[self.BOTTOM].y * h])

        width = np.linalg.norm(left - right)
        height = np.linalg.norm(top - bottom)
        ratio = height / width

        if ratio > 0.38:
            return "Surprised"
        elif ratio > 0.30:
            return "Smiling"
        else:
            return "Neutral"
