import cv2
from mtcnn import MTCNN
import mediapipe as mp
import numpy as np



face_detector = MTCNN()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(" Webcam not accessible")
    exit()

print("[INFO] Smile detection started. Press ESC to exit.")


LEFT_MOUTH = 61
RIGHT_MOUTH = 291
TOP_LIP = 13
BOTTOM_LIP = 14



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector.detect_faces(rgb)
    mesh_results = face_mesh.process(rgb)

    h, w, _ = frame.shape


    for face in faces:
        x, y, bw, bh = face['box']
        x, y = max(0, x), max(0, y)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

 
    if mesh_results.multi_face_landmarks:
        for landmarks in mesh_results.multi_face_landmarks:
            lm = landmarks.landmark

            left = np.array([lm[LEFT_MOUTH].x * w, lm[LEFT_MOUTH].y * h])
            right = np.array([lm[RIGHT_MOUTH].x * w, lm[RIGHT_MOUTH].y * h])
            top = np.array([lm[TOP_LIP].x * w, lm[TOP_LIP].y * h])
            bottom = np.array([lm[BOTTOM_LIP].x * w, lm[BOTTOM_LIP].y * h])

            mouth_width = np.linalg.norm(left - right)
            mouth_height = np.linalg.norm(top - bottom)

            smile_ratio = mouth_height / mouth_width


            if smile_ratio > 0.35:
                text = "Smiling"
                color = (0, 255, 0)
            else:
                text = "Not Smiling"
                color = (0, 0, 255)

            cv2.putText(frame, text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


            for idx in [LEFT_MOUTH, RIGHT_MOUTH, TOP_LIP, BOTTOM_LIP]:
                x_pt = int(lm[idx].x * w)
                y_pt = int(lm[idx].y * h)
                cv2.circle(frame, (x_pt, y_pt), 3, (255, 255, 0), -1)

    cv2.imshow("Smile Detection (MTCNN + MediaPipe)", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
