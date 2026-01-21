# model/model.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    # Load ResNet18
    model = models.resnet18(weights=None)  # pretrained=False
    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
