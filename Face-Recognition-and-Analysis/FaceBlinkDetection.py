import cv2
import mediapipe as mp
import numpy as np

class BlinkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )


        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        self.blink_counter = 0
        self.ear_threshold = 0.25
        self.consecutive_frames = 2

    def _eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def detect_blink(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        h, w, _ = frame.shape

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.LEFT_EYE])
        right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.RIGHT_EYE])

        ear = (self._eye_aspect_ratio(left_eye) +
               self._eye_aspect_ratio(right_eye)) / 2.0

        if ear < self.ear_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.consecutive_frames:
                self.blink_counter = 0
                return True
            self.blink_counter = 0

        return False
