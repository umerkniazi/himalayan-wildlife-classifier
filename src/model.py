import ssl
import torch.nn as nn
from torchvision import models

ssl._create_default_https_context = ssl._create_unverified_context


def get_model(num_classes=4, feature_extract=True):
    model = models.resnet18(weights='DEFAULT')

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model