import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _, classes = get_dataloaders()
model = get_model(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

EPOCHS = 10

best_acc = 0  # 🔥 Track best accuracy

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # 🔍 Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%\n")

    # 🔥 Save BEST model only
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "models/best_model.pth")
        print("🔥 Best model saved!")

    scheduler.step()

print(f"✅ Training complete! Best Accuracy: {best_acc:.2f}%")