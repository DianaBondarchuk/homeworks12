from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")

# ==========================================
# ЗАВДАННЯ 1: Підготовка датасету Chest X-Ray Pneumonia
# ==========================================
data_dir = Path("./chest_xray")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

train_dir = data_dir / "train"
test_dir = data_dir / "test"

if train_dir.exists() and test_dir.exists():
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)
    class_names = train_dataset.classes
else:
    full_train = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_test = datasets.ImageFolder(root=data_dir, transform=eval_transform)
    class_names = full_train.classes

    indices = list(range(len(full_train)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))

    train_dataset = Subset(full_train, indices[:split_idx])
    test_dataset = Subset(full_test, indices[split_idx:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print(f"Класи: {class_names}")
print(f"Розмір train: {len(train_dataset)}, розмір test: {len(test_dataset)}")


# ==========================================
# ЗАВДАННЯ 2: ResNet18, заморозка, новий класифікатор та навчання (5 епох)
# ==========================================
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features  # 512
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)

epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Епоха [{epoch}/{epochs}] — Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")


# ==========================================
# ЗАВДАННЯ 3: Оцінка якості та збереження моделі
# ==========================================
model.eval()
all_preds = []
all_labels = []
raw_test_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        raw_test_images.extend(images.cpu())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("\n--- Метрики на тестовій вибірці ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix — Pneumonia ResNet18")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

model_path = "pneumonia_resnet18.pth"
torch.save(model.state_dict(), model_path)
print(f"Модель збережено у файл: {model_path}")


# ==========================================
# ЗАВДАННЯ 4: Візуалізація 8 випадкових зображень з тесту
# ==========================================
def denormalize(tensor, mean=imagenet_mean, std=imagenet_std):
    img = tensor.clone().numpy().transpose(1, 2, 0)
    mean_arr = np.array(mean)
    std_arr = np.array(std)
    img = std_arr * img + mean_arr
    return np.clip(img, 0.0, 1.0)

num_samples = min(8, len(raw_test_images))
sample_indices = random.sample(range(len(raw_test_images)), num_samples)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, idx in zip(axes, sample_indices):
    img_vis = denormalize(raw_test_images[idx])
    true_cls = class_names[all_labels[idx]]
    pred_cls = class_names[all_preds[idx]]

    ax.imshow(img_vis)
    color = "green" if true_cls == pred_cls else "red"
    ax.set_title(f"Реальний: {true_cls}\nПередбачений: {pred_cls}", color=color, fontsize=10)
    ax.axis("off")

plt.suptitle("Приклади прогнозів на тестовій вибірці (Chest X-Ray Pneumonia)", fontsize=14)
plt.tight_layout()
plt.savefig("pneumonia_predictions.png", dpi=150)
plt.show()
print("Графік збережено під іменем: pneumonia_predictions.png")