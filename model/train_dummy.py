# model/train_dummy.py
import torch
import torch.nn as nn
from torchvision import models

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Small model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Dummy training step (for creating a valid file)
dummy_input = torch.randn(4, 3, 224, 224).to(device)
dummy_target = torch.tensor([0, 1, 0, 1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
output = model(dummy_input)
loss = criterion(output, dummy_target)
loss.backward()
optimizer.step()

# Save only state_dict
torch.save(model.state_dict(), "pneumonia.pt")
print("Dummy model saved successfully!")
