# model/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_model
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your NVIDIA GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
model = get_model(num_classes=2).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    # ✅ New validation with F1, Precision, Recall
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
val_f1 = f1_score(all_labels, all_preds)
val_precision = precision_score(all_labels, all_preds)
val_recall = recall_score(all_labels, all_preds)

print(f"Epoch [{epoch+1}/{num_epochs}] - "
      f"Loss: {running_loss/len(train_loader):.4f} - "
      f"Val Acc: {val_acc:.4f} - "
      f"F1: {val_f1:.4f} - "
      f"Precision: {val_precision:.4f} - "
      f"Recall: {val_recall:.4f}")

# Save trained model
torch.save(model.state_dict(), BASE_DIR / "pneumonia.pt")
print("Model saved successfully!")
