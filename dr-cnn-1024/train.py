"""
train.py - DR Pipeline run-08
Modèles : simplecnn | resnet18 | efficientnet_b0 | mobilenet_v2
Corrections run-08 :
  - WeightedRandomSampler : batches équilibrés par classe
  - class_weights dans CrossEntropyLoss (double pression)
  - Métriques complètes : F1_macro, F1_weighted, per_class_recall
  - 4 modèles supportés
"""

import os
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, recall_score, accuracy_score


# ── Modèles ───────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(model_name, num_classes=5):
    if model_name == "simplecnn":
        return SimpleCNN(num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataloaders(data_dir, batch_size, num_workers=4):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Structure attendue : data_dir/{0,1,2,3,4}/image.jpg
    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    val_dataset  = datasets.ImageFolder(data_dir, transform=val_tf)

    # Split 80/20 déterministe
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val   = n - n_train
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = torch.utils.data.random_split(
        range(n), [n_train, n_val], generator=generator
    )
    train_subset = torch.utils.data.Subset(full_dataset, train_idx.indices)
    val_subset   = torch.utils.data.Subset(val_dataset,  val_idx.indices)

    # WeightedRandomSampler sur le train set uniquement
    train_labels = [full_dataset.targets[i] for i in train_idx.indices]
    class_counts = torch.bincount(torch.tensor(train_labels))
    print(f"Distribution classes train : {class_counts.tolist()}")

    sample_weights = [1.0 / class_counts[lbl].item() for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train : {len(train_subset)} images | Val : {len(val_subset)} images")
    return train_loader, val_loader, class_counts


# ── Entraînement ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Modèle  : {args.model}")
    print(f"Epochs  : {args.epochs}")
    print(f"Batch   : {args.batch_size}")
    print(f"LR init : {args.lr}")

    os.makedirs(args.results_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, class_counts = build_dataloaders(
        args.data_dir, args.batch_size
    )

    # Modèle
    model = build_model(args.model).to(device)

    # class_weights pour CrossEntropyLoss
    total = class_counts.sum().float()
    class_weights = (total / (len(class_counts) * class_counts.float())).to(device)
    print(f"class_weights : {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler()

    best_f1_macro   = 0.0
    best_val_acc    = 0.0
    epoch_history   = []
    start_time      = time.time()

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # -- Validation --
        model.eval()
        val_loss    = 0.0
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds     = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        val_acc   = accuracy_score(all_targets, all_preds)
        f1_macro  = f1_score(all_targets, all_preds, average="macro",    zero_division=0)
        f1_w      = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| F1_macro={f1_macro:.4f} F1_w={f1_w:.4f} "
            f"| lr={current_lr:.5e}"
        )

        # Sauvegarde du meilleur modèle sur F1_macro
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_val_acc  = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.results_dir, "best_model.pth"))
            print(f"  -> Nouveau meilleur modèle sauvegardé (F1_macro={f1_macro:.4f})")

        epoch_history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "f1_macro":   round(f1_macro,   4),
            "f1_weighted":round(f1_w,       4),
            "lr":         round(current_lr, 8),
        })

    training_time = time.time() - start_time

    # Métriques finales sur le meilleur modèle
    per_class_names = [
        "class_0_No_DR", "class_1_Mild", "class_2_Moderate",
        "class_3_Severe", "class_4_Proliferative_DR"
    ]

    # Recharger le meilleur modèle pour les métriques finales
    best_model = build_model(args.model).to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(args.results_dir, "best_model.pth"),
                   map_location=device)
    )
    best_model.eval()
    final_preds   = []
    final_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with autocast():
                outputs = best_model(images)
            preds = outputs.argmax(dim=1)
            final_preds.extend(preds.cpu().tolist())
            final_targets.extend(labels.cpu().tolist())

    final_acc      = accuracy_score(final_targets, final_preds)
    final_f1_macro = f1_score(final_targets, final_preds, average="macro",    zero_division=0)
    final_f1_w     = f1_score(final_targets, final_preds, average="weighted", zero_division=0)
    per_class_recall = recall_score(
        final_targets, final_preds, average=None, zero_division=0
    )
    per_class_acc_dict = {
        per_class_names[i]: round(float(per_class_recall[i]), 4)
        for i in range(len(per_class_names))
    }

    metrics = {
        "model":                  args.model,
        "val_accuracy":           round(final_acc,      4),
        "f1_macro":               round(final_f1_macro, 4),
        "f1_weighted":            round(final_f1_w,     4),
        "per_class_accuracy":     per_class_acc_dict,
        "epochs":                 args.epochs,
        "batch_size":             args.batch_size,
        "lr_init":                args.lr,
        "training_time_seconds":  round(training_time, 2),
        "best_model_path":        os.path.join(args.results_dir, "best_model.pth"),
        "epoch_history":          epoch_history,
    }

    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Résultats finaux [{args.model}] ===")
    print(f"  val_accuracy : {final_acc:.4f}")
    print(f"  F1_macro     : {final_f1_macro:.4f}")
    print(f"  F1_weighted  : {final_f1_w:.4f}")
    print(f"  Per-class    : {per_class_acc_dict}")
    print(f"  Temps        : {training_time:.0f}s")
    print(f"  Métriques    : {metrics_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   required=True,
                        choices=["simplecnn", "resnet18", "efficientnet_b0", "mobilenet_v2"])
    parser.add_argument("--data_dir",    type=str,   required=True)
    parser.add_argument("--results_dir", type=str,   default="/tmp/output")
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=0.001)
    args = parser.parse_args()
    train(args)
