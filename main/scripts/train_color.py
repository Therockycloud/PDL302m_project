"""Full-fine-tune MobileNetV3-Small for vehicle colour classification.

Class order (must match runtime torch_color.py _CLASSES exactly):
    Index 0: Black
    Index 1: Blue
    Index 2: Brown
    Index 3: Grey
    Index 4: Red
    Index 5: Silver
    Index 6: White
    Index 7: Yellow
Source: main/src/models/torch_color.py line 20 — sorted folder names.

Why this beats the frozen-backbone baseline (~55%):
  • Unfreezes backbone so colour-sensitive features can develop.
  • Removes saturation/hue jitter that corrupts colour labels.
  • Discriminative LR: head 1e-3, backbone 1e-4.
  • Body-crop preprocessing removes sky/road before the network sees the image.

Run on Google Colab T4
-----------------------
# 1. Mount Drive (or upload data manually)
from google.colab import drive
drive.mount('/content/drive')

# 2. Install deps
!pip install torch torchvision scikit-learn

# 3. Copy data to /content/car_colors  (adjust path as needed)
# !cp -r /content/drive/MyDrive/DPL302m/data/raw/car_colors /content/car_colors

# 4. Run
!python train_color.py \\
    --data-dir /content/car_colors \\
    --device cuda \\
    --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image, ImageFilter
import sklearn.metrics as skm

# ---------------------------------------------------------------------------
# Constants — must match runtime torch_color.py _CLASSES exactly
# ---------------------------------------------------------------------------
CLASSES = ["Black", "Blue", "Brown", "Grey", "Red", "Silver", "White", "Yellow"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    """Select compute device: prefer cuda, then mps (Apple GPU), then cpu.

    MPS is ENABLED. Older PyTorch (<=2.0) had an MPS backward-pass bug for
    some MobileNetV3 ops; verified FIXED on torch>=2.2 — measured on an
    M1 Max it trains ~42x faster than CPU with no NaN. Pass --device cpu to
    force CPU only if a future regression reappears.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # auto / None (or cuda requested but unavailable): cuda > mps > cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Body-crop transform
# ---------------------------------------------------------------------------

class BodyCrop:
    """Crop out the top 20 % (sky/windshield) and bottom 15 % (tyres/road).

    Keeps the central body band so the model sees paint, not background.
    Applied as a PIL transform before any resize.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        top    = int(h * 0.20)
        bottom = int(h * 0.85)
        return img.crop((0, top, w, bottom))


class RandomDownscaleUpscale:
    """Mimic CCTV resolution loss: downscale to 50–100%, then restore."""

    def __init__(self, min_scale: float = 0.5):
        self.min_scale = min_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(self.min_scale, 1.0)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                           Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(train: bool) -> transforms.Compose:
    body_crop = BodyCrop()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if train:
        return transforms.Compose([
            body_crop,
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Colour jitter: brightness + contrast ONLY.
            # NO saturation, NO hue — those corrupt the colour label.
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            RandomDownscaleUpscale(min_scale=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            body_crop,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CarColorDataset(Dataset):
    """Flat folder-per-class dataset with explicit file list."""

    def __init__(self, samples: List[Tuple[str, int]], transform: transforms.Compose) -> None:
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Data loading & stratified split
# ---------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_samples(data_dir: str, limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """Collect (path, label_idx) pairs from <data_dir>/<ClassName>/*."""
    samples: List[Tuple[str, int]] = []
    root = pathlib.Path(data_dir)
    for cls_name in CLASSES:
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder not found: {cls_dir}\n"
                f"Data root must contain sub-dirs: {CLASSES}"
            )
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        if limit:
            files = files[:limit]
        label = CLASS_TO_IDX[cls_name]
        samples.extend((str(f), label) for f in files)
    return samples


def stratified_split(
    samples: List[Tuple[str, int]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = SEED,
) -> Tuple[List, List, List]:
    """Per-class stratified train / val / test split."""
    rng = random.Random(seed)
    train_s, val_s, test_s = [], [], []

    # Group by class
    by_class: Dict[int, List] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    for label, items in by_class.items():
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n       = len(shuffled)
        n_train = max(1, int(n * train_frac))
        n_val   = max(1, int(n * val_frac))
        train_s.extend(shuffled[:n_train])
        val_s.extend(  shuffled[n_train : n_train + n_val])
        test_s.extend( shuffled[n_train + n_val :])

    return train_s, val_s, test_s


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """Build MobileNetV3-Small with an 8-class head.

    Architecture mirrors torch_color.py exactly:
        model.classifier[3] = nn.Linear(in_features, 8)
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def make_optimizer(model: nn.Module, head_lr: float, backbone_lr: float) -> torch.optim.Optimizer:
    head_params     = list(model.classifier.parameters())
    head_param_ids  = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return torch.optim.AdamW([
        {"params": head_params,     "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,          # GradScaler or None
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds, all_true = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_true.extend(labels.tolist())
    return all_true, all_preds


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def build_report(y_true: List[int], y_pred: List[int]) -> Dict:
    acc    = skm.accuracy_score(y_true, y_pred)
    f1     = skm.f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec   = skm.precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    recall = skm.recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm     = skm.confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    per_class = {
        CLASSES[i]: {"precision": round(prec[i], 4), "recall": round(recall[i], 4)}
        for i in range(NUM_CLASSES)
    }

    return {
        "classes":          CLASSES,
        "test_accuracy":    round(acc, 4),
        "test_macro_f1":    round(f1, 4),
        "per_class":        per_class,
        "confusion_matrix": cm,
        "note":             "baseline frozen-backbone ≈ 0.55",
    }


def write_json_report(report: Dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] JSON written: {path}")


def write_md_report(report: Dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Color Classifier Fine-Tune Report",
        "",
        f"> baseline frozen-backbone ≈ 0.55",
        "",
        f"**Test Accuracy:** {report['test_accuracy']:.4f}  ",
        f"**Test Macro-F1:** {report['test_macro_f1']:.4f}",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall |",
        "|-------|-----------|--------|",
    ]
    for cls, m in report["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} |")

    lines += [
        "",
        "## Confusion Matrix",
        "",
        "Rows = true class, Columns = predicted class.",
        "",
        "| | " + " | ".join(report["classes"]) + " |",
        "|---" + "|---" * len(report["classes"]) + "|",
    ]
    for cls_name, row in zip(report["classes"], report["confusion_matrix"]):
        lines.append("| " + cls_name + " | " + " | ".join(str(v) for v in row) + " |")

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] Markdown written: {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args) -> None:  # noqa: C901
    set_seed(SEED)

    device = pick_device(args.device)
    print(f"[device] Using: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = args.data_dir
    print(f"[data] Loading from: {data_dir}")
    samples = load_samples(data_dir, limit=args.limit)
    print(f"[data] Total images: {len(samples)}")

    train_s, val_s, test_s = stratified_split(samples)
    print(f"[data] Split — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    train_set = CarColorDataset(train_s, build_transforms(train=True))
    val_set   = CarColorDataset(val_s,   build_transforms(train=False))
    test_set  = CarColorDataset(test_s,  build_transforms(train=False))

    # DataLoader workers parallelize CPU augmentation, but they are only
    # RELIABLE on CUDA/Linux (e.g. Colab). On macOS+MPS they are FLAKY —
    # fork after Metal/OpenMP init crashes intermittently, and persistent
    # workers stall the process at exit. Tested both; not worth a ~12% aug
    # speedup. So enable workers only on cuda; keep 0 on mps/cpu for safety.
    num_workers = 0 if device.type != "cuda" else min(6, os.cpu_count() or 0)
    _pw = num_workers > 0
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              persistent_workers=_pw)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False,
                              num_workers=num_workers, pin_memory=False,
                              persistent_workers=_pw)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False,
                              num_workers=num_workers, pin_memory=False,
                              persistent_workers=_pw)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # AMP only on CUDA
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ------------------------------------------------------------------
    # Phase 1: warm-up — train HEAD only (2-3 epochs)
    # ------------------------------------------------------------------
    warmup_epochs = min(3, args.epochs)
    freeze_backbone(model)
    optimizer = make_optimizer(model, head_lr=1e-3, backbone_lr=1e-4)

    print(f"\n[phase-1] Warming up HEAD for {warmup_epochs} epoch(s)...")
    for ep in range(1, warmup_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        print(f"  epoch {ep:3d}/{warmup_epochs} | "
              f"train loss {t_loss:.4f} acc {t_acc:.4f} | "
              f"val loss {v_loss:.4f} acc {v_acc:.4f}")

    if args.epochs <= warmup_epochs:
        # Smoke / very short run — skip phase 2
        print("[phase-1] Short run, skipping phase-2 unfreeze.")
    else:
        # ------------------------------------------------------------------
        # Phase 2: full fine-tune — unfreeze backbone, discriminative LR
        # ------------------------------------------------------------------
        unfreeze_all(model)
        optimizer = make_optimizer(model, head_lr=1e-3, backbone_lr=1e-4)
        remaining = args.epochs - warmup_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=1e-6
        )

        best_val_acc   = 0.0
        best_state     = None
        patience       = 5
        patience_count = 0

        print(f"\n[phase-2] Full fine-tune for up to {remaining} epoch(s)...")
        for ep in range(1, remaining + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            improved = v_acc > best_val_acc
            if improved:
                best_val_acc   = v_acc
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
                tag = " ← best"
            else:
                patience_count += 1
                tag = f" (patience {patience_count}/{patience})"

            print(f"  epoch {warmup_epochs + ep:3d}/{args.epochs} | "
                  f"train loss {t_loss:.4f} acc {t_acc:.4f} | "
                  f"val loss {v_loss:.4f} acc {v_acc:.4f}{tag}")

            if patience_count >= patience:
                print(f"[early stop] val_acc did not improve for {patience} epochs. Stopping.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[phase-2] Restored best checkpoint (val_acc={best_val_acc:.4f})")

    # ------------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------------
    # Determine project root (script lives at main/scripts/train_color.py)
    script_dir   = pathlib.Path(__file__).resolve().parent          # main/scripts
    project_main = script_dir.parent                                # main/
    weights_dir  = project_main / "data" / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_path = weights_dir / "color_MobileNetV3Small_ft.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"[save] Weights: {weights_path}")

    # ------------------------------------------------------------------
    # Test evaluation (held-out, final only)
    # ------------------------------------------------------------------
    print("\n[test] Evaluating on held-out test set...")
    y_true, y_pred = predict_all(model, test_loader, device)

    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    print(f"[test] Accuracy : {acc:.4f}")

    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"[test] Macro-F1 : {f1:.4f}")

    print("[test] Per-class breakdown:")
    for i, cls in enumerate(CLASSES):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == i and b == i)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != i and b == i)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == i and b != i)
        prec_i   = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {cls:8s}: prec={prec_i:.3f} recall={recall_i:.3f}")

    report = build_report(y_true, y_pred)

    # ------------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------------
    docs_dir = project_main.parent / "docs" / "benchmarks"
    json_path = docs_dir / "color_finetune_report.json"
    md_path   = docs_dir / "color_finetune_report.md"

    write_json_report(report, str(json_path))
    write_md_report(report, str(md_path))

    print("\n[done] Files written:")
    print(f"  Weights : {weights_path}")
    print(f"  JSON    : {json_path}")
    print(f"  Markdown: {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Full-fine-tune MobileNetV3-Small for car colour.")
    p.add_argument(
        "--data-dir",
        default=None,
        help="Path to car_colors/ root (default: auto-detected relative to script).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total training epochs (warmup + backbone fine-tune). Default: 30.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="cuda | cpu | auto (default: auto). MPS is intentionally disabled.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap images per class (for smoke testing).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: equivalent to --epochs 1 --limit 20.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke:
        args.epochs = 1
        if args.limit is None:
            args.limit = 20

    if args.data_dir is None:
        # Auto-detect: script is at main/scripts/, data at main/data/raw/car_colors/
        script_dir   = pathlib.Path(__file__).resolve().parent
        project_main = script_dir.parent
        args.data_dir = str(project_main / "data" / "raw" / "car_colors")

    train(args)
