import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import json

# Paths
DATA_DIR = "MICC-F220"
AU_PATH = os.path.join(DATA_DIR, "Au")
TU_PATH = os.path.join(DATA_DIR, "Tu")

# Output
OUTPUT_DIR = "model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image_paths, model):
    features = []
    for path in tqdm(image_paths):
        img = Image.open(path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor).squeeze().numpy()
            features.append(feat)
    return np.array(features)

def get_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def main():
    print("Loading VGG16...")
    vgg = models.vgg16(pretrained=True)
    vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-3])  # Keep feature layers
    vgg.eval()

    print("Collecting image paths...")
    au_imgs = get_image_paths(AU_PATH)
    tu_imgs = get_image_paths(TU_PATH)

    print("Extracting features...")
    X_au = extract_features(au_imgs, vgg)
    X_tu = extract_features(tu_imgs, vgg)

    X = np.concatenate([X_au, X_tu], axis=0)
    Y = np.array([0] * len(X_au) + [1] * len(X_tu))

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, Y)

    print("Evaluating...")
    preds = clf.predict(X)
    acc = accuracy_score(Y, preds)
    prec = precision_score(Y, preds)
    rec = recall_score(Y, preds)
    f1 = f1_score(Y, preds)
    cm = confusion_matrix(Y, preds).tolist()

    # Save model and data
    with open(os.path.join(OUTPUT_DIR, "vgg_forgery_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "Y.npy"), Y)

    # Save training history
    history = {
        "accuracy": [acc],
        "val_accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "f1_score": [f1]
    }
    with open(os.path.join(OUTPUT_DIR, "vgg_history.pckl"), "wb") as f:
        pickle.dump(history, f)

    # Save metrics for web
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("Training complete. All files saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
