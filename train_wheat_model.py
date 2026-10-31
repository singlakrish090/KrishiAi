import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.environ.get("WHEAT_DATA_DIR", "dataset")
OUTPUT_DIR = "ai_model/improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Expected dataset structure:
# dataset/train/<class_name>/*.jpg
# dataset/val/<class_name>/*.jpg
# dataset/test/<class_name>/*.jpg

if not all(os.path.isdir(os.path.join(DATA_DIR, s)) for s in ("train", "val", "test")):
    raise SystemExit(
        f"\nDataset not found at: {os.path.abspath(DATA_DIR)}\n"
        "Dataset must contain train, val, and test folders, each with one subfolder per class.\n"
        "Set WHEAT_DATA_DIR to your dataset path before running training."
    )

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_tf)
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_tf)

if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
    raise SystemExit("Class folders must match across train, val, and test.")

classes = train_ds.classes
counts = np.bincount(train_ds.targets)
weights = 1.0 / np.maximum(counts, 1)
sample_weights = [weights[y] for y in train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

workers = 0
train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=workers)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=workers)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=workers)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

best_f1 = -1.0
best_state = copy.deepcopy(model.state_dict())
patience, stale = 8, 0
epochs = 40

def evaluate(loader):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().tolist())
            ys.extend(y.tolist())
    return accuracy_score(ys, preds), f1_score(
        ys, preds, average="macro", zero_division=0
    ), ys, preds

print(f"Device: {DEVICE}")
print(f"Classes: {classes}")
print(f"Images: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc, val_f1, _, _ = evaluate(val_dl)
    scheduler.step(val_f1)
    print(
        f"Epoch {epoch+1:02d}/{epochs} | loss={total_loss/len(train_dl):.4f} "
        f"| val_accuracy={val_acc:.4f} | val_macro_f1={val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
        torch.save({
            "state_dict": best_state,
            "classes": classes,
            "class_to_idx": train_ds.class_to_idx,
            "architecture": "resnet18",
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        }, os.path.join(OUTPUT_DIR, "wheat_disease_classifier_improved.pth"))
    else:
        stale += 1
        if stale >= patience:
            print("Early stopping.")
            break

model.load_state_dict(best_state)
test_acc, test_f1, ys, preds = evaluate(test_dl)

print("\nFINAL HELD-OUT TEST RESULTS")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Macro F1: {test_f1:.4f}")
print(classification_report(ys, preds, labels=list(range(len(classes))),
                            target_names=classes, zero_division=0))

with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
    json.dump({
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "classes": classes,
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "test_images": len(test_ds),
    }, f, indent=2)

print(f"\nSaved new checkpoint and evaluation in {OUTPUT_DIR}")
print("Original model remains unchanged.")
