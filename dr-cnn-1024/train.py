"""
train.py — Diabetic Retinopathy CNN Training
Pipeline MLOps Kubeflow — Studienarbeit

Modeles supportes : simplecnn | resnet18 | efficientnet_b0 | mobilenet_v2

Parametres CLI (appeles par le composant KFP, cellule 5 du notebook) :
    --model        : architecture CNN
    --data_dir     : repertoire racine du dataset (train/ + val/ OU classes directes)
    --results_dir  : ou sauvegarder best_model.pth et metrics.json
    --epochs       : nombre d'epochs (defaut 25)
    --batch_size   : taille du batch (defaut 32)
    --lr           : learning rate initial (defaut 1e-3)
"""

import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


# Diabetic Retinopathy severity grades
DR_CLASSES = {
    "0": "No DR",
    "1": "Mild",
    "2": "Moderate",
    "3": "Severe",
    "4": "Proliferative DR"
}


class SimpleCNN(nn.Module):
    """Baseline CNN — reference pour comparer avec les modeles pretrained."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(model_name, num_classes):
    """Retourne le modele initialise avec les poids ImageNet (sauf SimpleCNN)."""
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
        raise ValueError(f"Modele inconnu : {model_name}")


def load_datasets(data_dir, transform_train, transform_val):
    """
    Charge train et val depuis data_dir.
    Cas A : data_dir/train/ + data_dir/val/ existent  → on les charge directement.
    Cas B : data_dir/0/ data_dir/1/ ...               → split 80/20 automatique.
    """
    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")

    if os.path.isdir(train_path) and os.path.isdir(val_path):
        print("Structure detectee : train/ + val/ separes.")
        train_dataset = ImageFolder(train_path, transform=transform_train)
        val_dataset   = ImageFolder(val_path,   transform=transform_val)
    else:
        print("Structure detectee : classes directes. Split 80/20 automatique.")
        full_dataset  = ImageFolder(data_dir, transform=transform_train)
        n_total       = len(full_dataset)
        n_val         = int(n_total * 0.2)
        n_train       = n_total - n_val
        train_dataset, val_subset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        # Appliquer les transforms val sur le sous-ensemble val
        val_subset.dataset = ImageFolder(data_dir, transform=transform_val)
        val_dataset = val_subset

    return train_dataset, val_dataset


def compute_class_weights(dataset, num_classes, device):
    """
    Poids par classe = 1 / frequence, normalises.
    Donne plus d'importance aux classes rares (grades 3 et 4 DR).
    """
    counts = torch.zeros(num_classes)

    # Gerer les deux cas : Dataset pur ou Subset (apres random_split)
    if hasattr(dataset, "samples"):
        samples = dataset.samples
    else:
        samples = dataset.dataset.samples

    for _, label in samples:
        counts[label] += 1

    print("Distribution des classes dans le train set :")
    for i, count in enumerate(counts):
        label = DR_CLASSES.get(str(i), str(i))
        print(f"  Classe {i} ({label}) : {int(count)} images")

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    print(f"Poids appliques : {[round(w, 3) for w in weights.tolist()]}")
    return weights.to(device)


def evaluate_per_class(model, loader, device, num_classes):
    """Accuracy par classe — essentiel pour detecter le class imbalance residuel."""
    correct_per_class = torch.zeros(num_classes)
    total_per_class   = torch.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = outputs.max(1)
            for c in range(num_classes):
                mask = labels == c
                correct_per_class[c] += (predicted[mask] == c).sum().item()
                total_per_class[c]   += mask.sum().item()

    per_class_acc = {}
    for c in range(num_classes):
        if total_per_class[c] > 0:
            acc = (correct_per_class[c] / total_per_class[c]).item()
        else:
            acc = 0.0
        label = DR_CLASSES.get(str(c), str(c))
        per_class_acc[f"class_{c}_{label.replace(' ', '_')}"] = round(acc, 4)

    return per_class_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Transforms adaptes aux images de fond d'oeil (retinal fundus images)
    # Augmentations conservatrices : eviter de deformer les lesions DR
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),           # fond d'oeil : symetrie verticale valide
        transforms.RandomRotation(15),             # rotation legere
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.3,                          # contraste important pour les lesions DR
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # moyennes ImageNet
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset, val_dataset = load_datasets(args.data_dir, transform_train, transform_val)
    num_classes = 5  # grades DR fixes : 0-4

    print(f"Train   : {len(train_dataset)} images")
    print(f"Val     : {len(val_dataset)} images")
    print(f"Modele  : {args.model}")
    print(f"Epochs  : {args.epochs}")
    print(f"Batch   : {args.batch_size}")
    print(f"LR init : {args.lr}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = get_model(args.model, num_classes).to(device)

    # Class weights pour corriger le desequilibre (classe 0 >> classes 3 et 4)
    class_weights = compute_class_weights(train_dataset, num_classes, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # CosineAnnealingLR : decroissance douce de lr_init a lr~0 sur toute la duree
    # Stable de 15 a 100 epochs — pas de chute brutale comme StepLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed Precision (AMP) — accelere x1.5-2 sur RTX 5090 sans perte de precision
    scaler = GradScaler()
    use_amp = torch.cuda.is_available()

    os.makedirs(args.results_dir, exist_ok=True)
    best_model_path = os.path.join(args.results_dir, "best_model.pth")
    best_val_acc    = 0.0
    epoch_history   = []
    start_time      = time.time()

    print("\n" + "="*60)
    print(f"Debut entrainement : {args.model.upper()}")
    print("="*60)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # --- Phase entrainement ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # --- Phase validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if use_amp:
                    with autocast():
                        outputs = model(images)
                        loss    = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc  = train_correct / train_total
        val_acc    = val_correct   / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch+1:02d}/{args.epochs}] "
              f"train_loss={train_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} "
              f"val_loss={val_loss/len(val_loader):.4f} "
              f"val_acc={val_acc:.4f} "
              f"lr={current_lr:.2e} "
              f"({epoch_time:.0f}s)")

        epoch_history.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss / len(train_loader), 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss / len(val_loader), 4),
            "val_acc":    round(val_acc, 4),
            "lr":         round(current_lr, 8)
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Meilleur modele sauvegarde (val_acc={best_val_acc:.4f})")

    total_time = time.time() - start_time

    # Accuracy par classe sur le meilleur modele
    print("\nCalcul accuracy par classe (meilleur modele)...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    per_class = evaluate_per_class(model, val_loader, device, num_classes)
    print("Accuracy par classe :")
    for cls, acc in per_class.items():
        print(f"  {cls} : {acc:.4f}")

    # Sauvegarder les metriques completes
    # Cle "val_accuracy" — compatible avec evaluate (cellule 6 du notebook)
    metrics = {
        "model":                 args.model,
        "val_accuracy":          round(best_val_acc, 4),
        "per_class_accuracy":    per_class,
        "epochs":                args.epochs,
        "batch_size":            args.batch_size,
        "lr_init":               args.lr,
        "training_time_seconds": round(total_time, 2),
        "best_model_path":       best_model_path,
        "epoch_history":         epoch_history
    }
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Entrainement termine : {args.model.upper()}")
    print(f"Best val_accuracy    : {best_val_acc:.4f}")
    print(f"Temps total          : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Metriques            -> {metrics_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DR CNN Training — Kubeflow MLOps Pipeline"
    )
    parser.add_argument("--model",       type=str,   required=True,
                        choices=["simplecnn", "resnet18", "efficientnet_b0", "mobilenet_v2"],
                        help="Architecture CNN")
    parser.add_argument("--data_dir",    type=str,   required=True,
                        help="Repertoire dataset (train/+val/ ou classes directes)")
    parser.add_argument("--results_dir", type=str,   required=True,
                        help="Repertoire de sortie pour best_model.pth et metrics.json")
    parser.add_argument("--epochs",      type=int,   default=25,
                        help="Nombre d'epochs (defaut 25)")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Batch size (defaut 32)")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate initial (defaut 1e-3)")
    args = parser.parse_args()
    train(args)
