import cv2
import os
from mtcnn import MTCNN
if not os.path.exists("dataset"):
    os.makedirs("dataset")

detector = MTCNN()
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print(" Webcam not accessible")
    exit()

face_id = input("\nEnter User ID and press <Enter> ==> ")
print("\n[INFO] Look at the camera. Capturing faces...")

count = 0

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
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        count += 1
        file_name = f"dataset/User.{face_id}.{count}.jpg"
        cv2.imwrite(file_name, gray)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(f"[INFO] Saved {file_name}")

    cv2.imshow("MTCNN Face Dataset Capture", frame)

    if cv2.waitKey(100) & 0xFF == 27 or count >= 30:
        break

print("\n[INFO] Dataset creation completed")
cam.release()
cv2.destroyAllWindows()
