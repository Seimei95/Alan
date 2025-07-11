import torchvision.models as models
import torch.nn as nn
import torch

def load_vgg16_model():
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])  # Remove FC layers
    model.eval()
    return model
