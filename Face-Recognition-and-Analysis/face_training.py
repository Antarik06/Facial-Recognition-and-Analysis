import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
import joblib

model = InceptionResnetV1(pretrained='vggface2').eval()

embeddings = []
labels = []

dataset_path = "dataset"

for file in os.listdir(dataset_path):
    path = os.path.join(dataset_path, file)
    img = Image.open(path).convert('RGB').resize((160,160))
    img = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        emb = model(img)

    label = int(file.split(".")[1])
    embeddings.append(emb.squeeze().numpy())
    labels.append(label)

clf = SVC(kernel='linear', probability=True)
clf.fit(embeddings, labels)

joblib.dump(clf, "trainer/facenet_svm.pkl")

print("[INFO] FaceNet model trained successfully")
