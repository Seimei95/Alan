from model_loader import load_vgg16_model
from utils import preprocess_image
from draw import draw_forgery_boxes
from sklearn.cluster import DBSCAN
import numpy as np
import torch
import os

def detect_forgery(image_path):
    model = load_vgg16_model()
    input_tensor, original = preprocess_image(image_path)

    with torch.no_grad():
        features = model(input_tensor).squeeze().numpy()

    reshaped = features.reshape(-1, features.shape[-1])
    clustering = DBSCAN(eps=3, min_samples=2).fit(reshaped)

    mask = clustering.labels_.reshape((7, 7))  # Assuming output was (7,7,N)
    mask = np.kron(mask, np.ones((32, 32)))    # Upsample to 224x224

    output_path = os.path.join("static", "output.jpg")
    draw_forgery_boxes(mask, original, output_path)

    return output_path, True
