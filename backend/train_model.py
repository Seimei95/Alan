import os
import sys
import subprocess
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pickle
import matplotlib.pyplot as plt
import json

# ---------------------------
# Step 1: Check dataset
# ---------------------------
DATASET_DIR = "MICC-F220"
AU_PATH = os.path.join(DATASET_DIR, "Au")
TU_PATH = os.path.join(DATASET_DIR, "Tu")

if not os.path.exists(AU_PATH) or not os.path.exists(TU_PATH):
    print("MICC-F220 dataset not found.")
    print("Attempting to download using download_data.py...")

    if os.path.exists("download_data.py"):
        subprocess.run([sys.executable, "download_data.py"])
    else:
        raise FileNotFoundError(" Dataset missing and no download_data.py found.")

    if not os.path.exists(AU_PATH) or not os.path.exists(TU_PATH):
        raise FileNotFoundError(" Dataset download failed or directory structure incorrect.")

# ---------------------------
# Step 2: Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True).features.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.view(-1).cpu().numpy()

# ---------------------------
# Step 3: Load images and extract features
# ---------------------------
X = []
Y = []

print("🔍 Extracting features from Au (authentic) images...")
for filename in os.listdir(AU_PATH):
    path = os.path.join(AU_PATH, filename)
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        try:
            features = extract_features(path)
            X.append(features)
            Y.append(0)
        except Exception as e:
            print(f" Failed on {filename}: {e}")

print("🔍 Extracting features from Tu (tampered) images...")
for filename in os.listdir(TU_PATH):
    path = os.path.join(TU_PATH, filename)
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        try:
            features = extract_features(path)
            X.append(features)
            Y.append(1)
        except Exception as e:
            print(f" Failed on {filename}: {e}")

X = np.array(X)
Y = np.array(Y)

# ---------------------------
# Step 4: Train model
# ---------------------------
print("🧠 Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X, Y)
y_pred = clf.predict(X)

# ---------------------------
# Step 5: Metrics
# ---------------------------
acc = accuracy_score(Y, y_pred)
prec = precision_score(Y, y_pred)
rec = recall_score(Y, y_pred)
f1 = f1_score(Y, y_pred)
cm = confusion_matrix(Y, y_pred)

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "confusion_matrix": cm.tolist()
}

os.makedirs("backend/model_output", exist_ok=True)
with open("backend/model_output/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------
# Step 6: Save model and features
# ---------------------------
with open("backend/model_output/vgg_forgery_model.pkl", "wb") as f:
    pickle.dump(clf, f)

np.save("backend/model_output/X.npy", X)
np.save("backend/model_output/Y.npy", Y)

# ---------------------------
# Step 7: Plot accuracy
# ---------------------------
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Precision", "Recall", "F1"], [acc, prec, rec, f1], color='skyblue')
plt.title("Model Metrics")
plt.ylim(0, 1)
plt.savefig("backend/static/stats.png")
plt.close()

print("All done. Model and metrics saved.")
