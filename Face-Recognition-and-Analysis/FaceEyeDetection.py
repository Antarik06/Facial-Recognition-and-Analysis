import cv2
from mtcnn import MTCNN
import mediapipe as mp



face_detector = MTCNN()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True
)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(" Webcam not accessible")
    exit()

print("[INFO] Face + Eye detection started. Press ESC to exit.")



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

            LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
            RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]

            for idx in LEFT_EYE + RIGHT_EYE:
                lm = landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Face & Eye Detection (MTCNN + MediaPipe)", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
