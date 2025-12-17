import cv2
import torch
import numpy as np
import joblib
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1


print("[INFO] Loading models...")

detector = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()

classifier = joblib.load("trainer/facenet_svm.pkl")

names = {
    1: "Kunal",
    2: "Kaushik",
    3: "Atharv",
    4: "Z",
    5: "W"
}


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cam.isOpened():
    print(" Webcam not accessible")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

print("[INFO] Face recognition started. Press ESC to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_resized = cv2.resize(face_img, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        face_tensor = torch.tensor(face_rgb).permute(2, 0, 1).unsqueeze(0).float()


        with torch.no_grad():
            embedding = facenet(face_tensor).numpy()

        probs = classifier.predict_proba(embedding)[0]
        best_class = np.argmax(probs)
        confidence = probs[best_class]

        if confidence > 0.6:
            name = names.get(best_class, "Unknown")
            label = f"{name} ({confidence*100:.2f}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), font, 0.8, color, 2)

    cv2.imshow("Face Recognition (FaceNet)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


print("\n[INFO] Exiting program and cleaning up")
cam.release()
cv2.destroyAllWindows()
