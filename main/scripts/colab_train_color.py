"""Self-contained Colab fine-tune for vehicle colour classification — 80% push.

This is a STANDALONE script: no imports from this repo. Copy it anywhere
(e.g. paste into a Colab cell or upload the .py and %run it) and it works
on its own, as long as torch / torchvision / scikit-learn / Pillow are
installed (Colab has them pre-installed except scikit-learn, which Colab
also ships by default).

Class order (MUST match runtime torch_color.py _CLASSES exactly — this is
what makes the saved weights drop-in compatible with main/src/models/torch_color.py):
    Index 0: Black
    Index 1: Blue
    Index 2: Brown
    Index 3: Grey
    Index 4: Red
    Index 5: Silver
    Index 6: White
    Index 7: Yellow

Recipe carried over from main/scripts/train_color.py (the 77.6% baseline)
unchanged:
    • Body-crop preprocessing (drop top 20% / bottom 15% — sky/road).
    • Augmentation: brightness/contrast jitter + blur + downscale-upscale.
      NO saturation/hue jitter — that corrupts the colour label itself.
    • Discriminative LR: head 1e-3, backbone 1e-4. Warm up head-only for
      a few epochs, then unfreeze the backbone and fine-tune everything.
    • 70/15/15 stratified split, seed=42.

Three NEW accuracy levers added on top, aimed at the weak neutral classes
(Silver / Grey / White), targeting 77.6% → ≥80% test accuracy:
    1. Class-weighted loss   — nn.CrossEntropyLoss(weight=w), w = inverse
       class frequency (normalized so weights average to 1). Under-
       represented / hard classes (Silver, Grey) get a bigger gradient
       signal instead of being drowned out by the larger classes.
    2. Label smoothing       — label_smoothing=0.1 on the same loss.
       Softens one-hot targets, which helps most exactly on classes that
       are visually ambiguous with their neighbours (Silver vs Grey vs
       White all sit close together in colour space).
    3. Test-time augmentation (TTA) — at FINAL test evaluation only,
       average the softmax of each image with the softmax of its
       horizontal flip. Reported as a separate "TTA accuracy" alongside
       the plain (no-TTA) accuracy so the lift is visible.

===========================================================================
HOW TO RUN THIS ON GOOGLE COLAB (step by step)
===========================================================================
1. Runtime → Change runtime type → Hardware accelerator → GPU (T4 is fine).

2. Get the VCoR dataset onto the Colab VM. Either:
   (a) Upload the VCoR zip via the Colab file browser, then:
         !unzip -q /content/vcor.zip -d /content/vcor
       The script auto-detects the VCoR layout:
         <root>/{train,val,test}/<lowercolor>/*.jpg
   (b) Or mount Google Drive and point --data-dir at the extracted folder:
         from google.colab import drive
         drive.mount('/content/drive')

3. Upload this script (colab_train_color.py) to the Colab session, or
   paste its contents into a cell, then run:
         !python colab_train_color.py --data-dir /content/vcor --epochs 30
   (or, in a notebook cell: `%run colab_train_color.py --data-dir /content/vcor --epochs 30`)

4. When it finishes it prints the path of the saved .pt weights and the
   JSON report, and — because this is running on Colab — automatically
   triggers a browser download of both files via google.colab.files.
   If the auto-download doesn't fire (e.g. pop-up blocked), the files are
   still on disk at the printed paths; download them manually from the
   Colab file browser on the left.

Smoke test (tiny, 1 epoch, CPU, to sanity-check the script end-to-end):
         !python colab_train_color.py --data-dir /content/vcor --smoke
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import sklearn.metrics as skm

# ---------------------------------------------------------------------------
# Constants — must match runtime torch_color.py _CLASSES exactly
# ---------------------------------------------------------------------------
CLASSES = ["Black", "Blue", "Brown", "Grey", "Red", "Silver", "White", "Yellow"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# VCoR dataset folder names (lowercase) -> our class. Colors not in this
# map (beige, gold, green, orange, pink, purple, tan, ...) are ignored.
VCOR_TO_CLASS = {
    "black": "Black",
    "blue": "Blue",
    "brown": "Brown",
    "grey": "Grey",
    "gray": "Grey",  # tolerate US spelling, just in case
    "red": "Red",
    "silver": "Silver",
    "white": "White",
    "yellow": "Yellow",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    """Select compute device.

    NEVER use MPS (Apple Silicon) — PyTorch MPS has a backward-pass bug that
    causes silent NaN gradients for some MobileNetV3 graph patterns. On
    Colab this is moot (no MPS there), but we keep the same guard as
    train_color.py so behaviour is identical if this script is ever run
    locally on a Mac.
    """
    if requested in ("mps",):
        print("[device] MPS requested but DISABLED (backward bug). Using cpu.")
        return torch.device("cpu")
    if requested == "auto" or requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[device] CUDA not available, falling back to cpu.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Body-crop transform (identical to train_color.py)
# ---------------------------------------------------------------------------

class BodyCrop:
    """Crop out the top 20% (sky/windshield) and bottom 15% (tyres/road).

    Keeps the central body band so the model sees paint, not background.
    Applied as a PIL transform before any resize.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        top = int(h * 0.20)
        bottom = int(h * 0.85)
        return img.crop((0, top, w, bottom))


class RandomDownscaleUpscale:
    """Mimic CCTV resolution loss: downscale to 50-100%, then restore."""

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


def build_eval_base_transform() -> transforms.Compose:
    """Eval transform WITHOUT the final ToTensor/Normalize — used by the
    TTA path so we can branch into flip / no-flip on the same PIL image."""
    return transforms.Compose([
        BodyCrop(),
        transforms.Resize((224, 224)),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CarColorDataset(Dataset):
    """Flat or pooled folder-per-class dataset with explicit file list."""

    def __init__(self, samples: List[Tuple[str, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


class CarColorPathDataset(Dataset):
    """Returns (PIL image post body-crop+resize, label) — used for TTA eval
    so we can derive both the plain and the flipped tensor from one PIL
    image without re-decoding the file twice."""

    def __init__(self, samples: List[Tuple[str, int]], base_transform: transforms.Compose) -> None:
        self.samples = samples
        self.base_transform = base_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.base_transform(img)
        return img, label


def tta_collate(batch):
    """Collate for CarColorPathDataset: keep PIL images as a list (can't
    stack PIL objects into a tensor batch)."""
    imgs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, labels


# ---------------------------------------------------------------------------
# Data loading — auto-detect layout
# ---------------------------------------------------------------------------

def _collect_images(folder: pathlib.Path) -> List[pathlib.Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def detect_layout(data_dir: str) -> str:
    """Return 'vcor' if <root>/{train,val,test}/<lowercolor>/ exists,
    else 'flat' if <root>/<Class>/ exists (our canonical class names)."""
    root = pathlib.Path(data_dir)

    vcor_splits = [s for s in ("train", "val", "test") if (root / s).is_dir()]
    if vcor_splits:
        # Confirm at least one split contains a known vcor colour folder.
        for split in vcor_splits:
            split_dir = root / split
            subdirs = {d.name.lower() for d in split_dir.iterdir() if d.is_dir()}
            if subdirs & set(VCOR_TO_CLASS.keys()):
                return "vcor"

    flat_subdirs = {d.name for d in root.iterdir() if d.is_dir()} if root.is_dir() else set()
    if flat_subdirs & set(CLASSES):
        return "flat"

    raise FileNotFoundError(
        f"Could not detect dataset layout under {root}.\n"
        f"Expected EITHER:\n"
        f"  vcor layout: {{train,val,test}}/<lowercolor>/*.jpg\n"
        f"  flat layout: <Class>/*.jpg with Class in {CLASSES}"
    )


def load_samples_flat(data_dir: str, limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """Collect (path, label_idx) pairs from <data_dir>/<ClassName>/*."""
    samples: List[Tuple[str, int]] = []
    root = pathlib.Path(data_dir)
    for cls_name in CLASSES:
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            print(f"[data] WARNING: class folder not found, skipping: {cls_dir}")
            continue
        files = _collect_images(cls_dir)
        if limit:
            files = files[:limit]
        label = CLASS_TO_IDX[cls_name]
        samples.extend((str(f), label) for f in files)
    return samples


def load_samples_vcor(data_dir: str, limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """Pool train+val+test from VCoR layout, map lowercase folder name to
    our class, ignore colours not in VCOR_TO_CLASS (beige/gold/green/
    orange/pink/purple/tan/...). limit, if given, caps images PER OUR
    CLASS across the pooled splits (applied after pooling so each class
    gets a representative mix of the original splits)."""
    root = pathlib.Path(data_dir)
    by_class: Dict[str, List[pathlib.Path]] = {c: [] for c in CLASSES}

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for color_dir in split_dir.iterdir():
            if not color_dir.is_dir():
                continue
            mapped = VCOR_TO_CLASS.get(color_dir.name.lower())
            if mapped is None:
                continue  # ignore beige/gold/green/orange/pink/purple/tan/...
            by_class[mapped].extend(_collect_images(color_dir))

    samples: List[Tuple[str, int]] = []
    for cls_name in CLASSES:
        files = sorted(by_class[cls_name])
        if limit:
            files = files[:limit]
        label = CLASS_TO_IDX[cls_name]
        samples.extend((str(f), label) for f in files)
    return samples


def load_samples(data_dir: str, limit: Optional[int] = None) -> Tuple[List[Tuple[str, int]], str]:
    layout = detect_layout(data_dir)
    print(f"[data] Detected layout: {layout}")
    if layout == "vcor":
        samples = load_samples_vcor(data_dir, limit=limit)
    else:
        samples = load_samples_flat(data_dir, limit=limit)

    missing = [c for c in CLASSES if not any(lbl == CLASS_TO_IDX[c] for _, lbl in samples)]
    if missing:
        raise RuntimeError(f"No images found for class(es): {missing}. Check data-dir contents.")
    return samples, layout


def stratified_split(
    samples: List[Tuple[str, int]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = SEED,
) -> Tuple[List, List, List]:
    """Per-class stratified train / val / test split."""
    rng = random.Random(seed)
    train_s, val_s, test_s = [], [], []

    by_class: Dict[int, List] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    for label, items in by_class.items():
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train_s.extend(shuffled[:n_train])
        val_s.extend(shuffled[n_train: n_train + n_val])
        test_s.extend(shuffled[n_train + n_val:])

    return train_s, val_s, test_s


# ---------------------------------------------------------------------------
# Lever 1: class weights (inverse frequency, normalized to mean 1)
# ---------------------------------------------------------------------------

def compute_class_weights(train_samples: List[Tuple[str, int]]) -> torch.Tensor:
    """Inverse-frequency class weights computed from the TRAIN split only.

    w_i = (1 / count_i) normalized so that mean(w) == 1. This keeps the
    overall loss magnitude in the same ballpark as unweighted CE while
    still boosting gradient signal for under-represented classes (e.g.
    Silver, which has the fewest VCoR images) relative to the large
    classes (e.g. Blue/Yellow).
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, label in train_samples:
        counts[label] += 1
    counts = np.maximum(counts, 1)  # guard against div-by-zero
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model builder (identical architecture to train_color.py / torch_color.py)
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
    head_params = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return torch.optim.AdamW([
        {"params": head_params, "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
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
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
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
# Lever 3: Test-time augmentation (TTA) — flip-averaged softmax
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_all_tta(
    model: nn.Module,
    samples: List[Tuple[str, int]],
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[List[int], List[int], List[int]]:
    """Evaluate with TTA: average softmax(orig) and softmax(hflip).

    Returns (y_true, y_pred_plain, y_pred_tta) so the caller can report
    both the plain and the TTA accuracy from a single pass (no duplicate
    forward work for the "plain" half: it's the first term of the TTA
    average computed once, not run twice).
    """
    base_tf = build_eval_base_transform()
    ds = CarColorPathDataset(samples, base_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=tta_collate)

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    to_tensor = transforms.ToTensor()

    model.eval()
    y_true: List[int] = []
    y_pred_plain: List[int] = []
    y_pred_tta: List[int] = []

    for pil_imgs, labels in loader:
        orig_tensors = torch.stack([normalize(to_tensor(im)) for im in pil_imgs]).to(device)
        flip_tensors = torch.stack(
            [normalize(to_tensor(im.transpose(Image.FLIP_LEFT_RIGHT))) for im in pil_imgs]
        ).to(device)

        probs_orig = F.softmax(model(orig_tensors), dim=1)
        probs_flip = F.softmax(model(flip_tensors), dim=1)
        probs_avg = (probs_orig + probs_flip) / 2.0

        y_true.extend(labels.tolist())
        y_pred_plain.extend(probs_orig.argmax(1).cpu().tolist())
        y_pred_tta.extend(probs_avg.argmax(1).cpu().tolist())

    return y_true, y_pred_plain, y_pred_tta


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def build_report(
    y_true: List[int],
    y_pred_plain: List[int],
    y_pred_tta: List[int],
    class_weights: torch.Tensor,
    layout: str,
    n_samples: int,
) -> Dict:
    acc_plain = skm.accuracy_score(y_true, y_pred_plain)
    acc_tta = skm.accuracy_score(y_true, y_pred_tta)

    f1_plain = skm.f1_score(y_true, y_pred_plain, average="macro", zero_division=0)
    f1_tta = skm.f1_score(y_true, y_pred_tta, average="macro", zero_division=0)

    prec = skm.precision_score(y_true, y_pred_tta, average=None, zero_division=0, labels=list(range(NUM_CLASSES))).tolist()
    recall = skm.recall_score(y_true, y_pred_tta, average=None, zero_division=0, labels=list(range(NUM_CLASSES))).tolist()
    cm = skm.confusion_matrix(y_true, y_pred_tta, labels=list(range(NUM_CLASSES))).tolist()

    per_class = {
        CLASSES[i]: {"precision": round(prec[i], 4), "recall": round(recall[i], 4)}
        for i in range(NUM_CLASSES)
    }

    return {
        "classes": CLASSES,
        "data_layout": layout,
        "n_samples_total": n_samples,
        "class_weights": {CLASSES[i]: round(float(class_weights[i]), 4) for i in range(NUM_CLASSES)},
        "test_accuracy_plain": round(acc_plain, 4),
        "test_accuracy_tta": round(acc_tta, 4),
        "test_macro_f1_plain": round(f1_plain, 4),
        "test_macro_f1_tta": round(f1_tta, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_note": "rows=true, cols=predicted (TTA predictions)",
        "levers": [
            "class_weighted_loss (inverse frequency, mean-normalized)",
            "label_smoothing=0.1",
            "test_time_augmentation (orig + hflip softmax average)",
        ],
        "baseline_note": "prior full fine-tune (no levers) test accuracy ~= 0.776",
    }


def write_json_report(report: Dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] JSON written: {path}")


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
    samples, layout = load_samples(data_dir, limit=args.limit)
    print(f"[data] Total images (pooled, mapped to our {NUM_CLASSES} classes): {len(samples)}")

    counts = np.zeros(NUM_CLASSES, dtype=int)
    for _, lbl in samples:
        counts[lbl] += 1
    print("[data] Per-class counts: " + ", ".join(f"{CLASSES[i]}={counts[i]}" for i in range(NUM_CLASSES)))

    train_s, val_s, test_s = stratified_split(samples)
    print(f"[data] Split — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    train_set = CarColorDataset(train_s, build_transforms(train=True))
    val_set = CarColorDataset(val_s, build_transforms(train=False))

    num_workers = 0 if device.type != "cuda" else 2
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ------------------------------------------------------------------
    # Lever 1: class-weighted loss (computed from TRAIN split)
    # ------------------------------------------------------------------
    class_weights = compute_class_weights(train_s)
    print("[lever-1] Class weights (inverse frequency, mean-normalized):")
    for i, cls in enumerate(CLASSES):
        print(f"           {cls:8s}: {class_weights[i].item():.4f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model()
    model.to(device)

    # Levers 1 + 2 combined in one CrossEntropyLoss: class weights +
    # label smoothing.
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1,
    )
    print("[lever-2] label_smoothing=0.1 enabled on the loss.")

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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

        best_val_acc = 0.0
        best_state = None
        patience = 5
        patience_count = 0

        print(f"\n[phase-2] Full fine-tune for up to {remaining} epoch(s)...")
        for ep in range(1, remaining + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            improved = v_acc > best_val_acc
            if improved:
                best_val_acc = v_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
                tag = " <- best"
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
    weights_path = pathlib.Path(args.out).resolve()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"[save] Weights: {weights_path}")

    # ------------------------------------------------------------------
    # Lever 3: TTA test evaluation (held-out, final only)
    # ------------------------------------------------------------------
    print("\n[test] Evaluating on held-out test set (plain + TTA)...")
    y_true, y_pred_plain, y_pred_tta = predict_all_tta(model, test_s, device)

    acc_plain = sum(1 for a, b in zip(y_true, y_pred_plain) if a == b) / len(y_true)
    acc_tta = sum(1 for a, b in zip(y_true, y_pred_tta) if a == b) / len(y_true)
    print(f"[test] Accuracy (no TTA) : {acc_plain:.4f}")
    print(f"[lever-3] Accuracy (TTA)  : {acc_tta:.4f}  (orig + hflip softmax average)")

    f1_plain = skm.f1_score(y_true, y_pred_plain, average="macro", zero_division=0)
    f1_tta = skm.f1_score(y_true, y_pred_tta, average="macro", zero_division=0)
    print(f"[test] Macro-F1 (no TTA) : {f1_plain:.4f}")
    print(f"[test] Macro-F1 (TTA)    : {f1_tta:.4f}")

    print("[test] Per-class breakdown (TTA predictions):")
    for i, cls in enumerate(CLASSES):
        tp = sum(1 for a, b in zip(y_true, y_pred_tta) if a == i and b == i)
        fp = sum(1 for a, b in zip(y_true, y_pred_tta) if a != i and b == i)
        fn = sum(1 for a, b in zip(y_true, y_pred_tta) if a == i and b != i)
        prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {cls:8s}: prec={prec_i:.3f} recall={recall_i:.3f}")

    cm = skm.confusion_matrix(y_true, y_pred_tta, labels=list(range(NUM_CLASSES)))
    print("[test] Confusion matrix (rows=true, cols=predicted, TTA):")
    header = "          " + " ".join(f"{c[:4]:>5s}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:8s}" + " ".join(f"{v:5d}" for v in row))

    report = build_report(y_true, y_pred_plain, y_pred_tta, class_weights, layout, len(samples))

    # ------------------------------------------------------------------
    # Write JSON report
    # ------------------------------------------------------------------
    json_path = pathlib.Path(args.out).with_suffix("").as_posix() + "_report.json"
    write_json_report(report, json_path)

    print("\n[done] Files written:")
    print(f"  Weights : {weights_path}")
    print(f"  JSON    : {json_path}")
    print(f"  Test accuracy : plain={acc_plain:.4f}  TTA={acc_tta:.4f}")

    # ------------------------------------------------------------------
    # Colab convenience: auto-download weights + report
    # ------------------------------------------------------------------
    if is_colab():
        try:
            from google.colab import files  # type: ignore
            print("[colab] Triggering download of weights + report...")
            files.download(str(weights_path))
            files.download(str(json_path))
        except Exception as exc:  # pragma: no cover - best-effort UX only
            print(f"[colab] Auto-download failed ({exc}); download manually from the file browser.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Self-contained Colab fine-tune for car colour (class-weighted loss + "
                    "label smoothing + TTA, targeting >=80% test accuracy)."
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset root. Either VCoR layout "
             "(<root>/{train,val,test}/<lowercolor>/*.jpg) or flat layout "
             "(<root>/<Class>/*.jpg, Class in " + str(CLASSES) + ").",
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
        help="Cap images per class (use the full dataset by default).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 epoch, --limit 20, forces device=cpu.",
    )
    p.add_argument(
        "--out",
        default="./color_MobileNetV3Small_ft.pt",
        help="Output path for the fine-tuned weights (.pt). "
             "JSON report is written alongside as <out>_report.json. "
             "Default: ./color_MobileNetV3Small_ft.pt",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke:
        args.epochs = 1
        if args.limit is None:
            args.limit = 20
        args.device = "cpu"
        print("[smoke] Smoke mode: epochs=1, limit=20, device=cpu.")

    t0 = time.time()
    train(args)
    print(f"\n[time] Total elapsed: {time.time() - t0:.1f}s")
