# =========================================================
# Fast CNN for Blood Type Classification (CPU Friendly)
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ───────────────────────────────────────────────
print("started")
TRAIN_DIR = "train"
TEST_DIR  = "test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 42
LR = 0.001
NUM_CLASSES = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ── SAFE LOADER ──────────────────────────────────────────
def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert("L")
    except:
        print("⚠️ Skipped:", path)
        return Image.new("L", IMG_SIZE)

# ── TRANSFORMS ───────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ── DATASET ──────────────────────────────────────────────
train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform, loader=safe_loader)
test_dataset  = ImageFolder(TEST_DIR, transform=test_transform, loader=safe_loader)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

CLASS_NAMES = train_dataset.classes
print("Classes:", CLASS_NAMES)

# ── MODEL ────────────────────────────────────────────────
class FastCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = FastCNN().to(DEVICE)

# ── LOSS & OPTIMIZER ─────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ── TRAINING ─────────────────────────────────────────────
train_accs, val_accs = [], []
best_acc = 0

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    # TRAIN
    model.train()
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    val_acc = 100 * correct / total

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "fastcnn_model.pth")

print("\nBest Accuracy:", best_acc)

# ── EVALUATION ───────────────────────────────────────────
model.load_state_dict(torch.load("fastcnn_model.pth"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        _, pred = outputs.max(1)

        y_true.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── CONFUSION MATRIX ─────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()