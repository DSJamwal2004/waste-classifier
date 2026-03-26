import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last layer block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model