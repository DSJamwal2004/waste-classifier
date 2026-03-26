import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from data_loader import get_dataloaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader, classes = get_dataloaders()

model = get_model(len(classes))
model.load_state_dict(torch.load("models/waste_classifier.pth"))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(all_labels, all_preds, target_names=classes)
print("\nClassification Report:\n", report)