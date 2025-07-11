from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_image(path):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, np.array(image)
