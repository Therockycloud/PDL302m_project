"""Domain-randomised fine-tune of the colour classifier for CCTV transfer.

The clean ``car_colors`` photos are a different domain from garage CCTV
(fluorescent light washes colours out). This script fine-tunes the WHOLE
MobileNetV3-Small (unfrozen) with strong augmentation that mimics CCTV
degradation — blur, brightness/contrast/saturation jitter, downscale — to see
whether it transfers. Evaluates on the clean val split AND on the real CCTV
test cars whose true colour is known (black, white). Run from ``main/``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

cv2.setNumThreads(0)
_MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MAIN))
DATA = _MAIN / "data" / "raw" / "car_colors"
CLASSES = ["Black", "Blue", "Brown", "Grey", "Red", "Silver", "White", "Yellow"]
IMG = 224
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _load():
    rng = np.random.default_rng(42)
    Xtr, ytr, Xva, yva = [], [], [], []
    for ci, c in enumerate(CLASSES):
        d = DATA / c
        if not d.exists():
            continue
        imgs = []
        for f in sorted(d.glob("*")):
            im = cv2.imread(str(f))
            if im is None:
                continue
            imgs.append(cv2.cvtColor(cv2.resize(im, (IMG, IMG)), cv2.COLOR_BGR2RGB))
        idx = rng.permutation(len(imgs))
        nval = max(1, int(0.2 * len(imgs)))
        for j, k in enumerate(idx):
            (Xva if j < nval else Xtr).append(imgs[k])
            (yva if j < nval else ytr).append(ci)
    to_t = lambda a: torch.tensor(np.asarray(a, "float32").transpose(0, 3, 1, 2) / 255.0)
    return to_t(Xtr), torch.tensor(ytr), to_t(Xva), torch.tensor(yva)


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xtr, ytr, Xva, yva = _load()
    print(f"device={dev} train={len(ytr)} val={len(yva)}", flush=True)

    norm = transforms.Normalize(_MEAN, _STD)
    # CCTV-mimicking augmentation applied to [0,1] CHW batches on the fly.
    aug = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.7, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.5, hue=0.03),
        transforms.GaussianBlur(5, sigma=(0.1, 2.5)),
    ])

    def downscale(x):  # simulate low-res CCTV: shrink then upsize
        s = int(IMG * float(np.random.default_rng().uniform(0.4, 1.0)))
        x = torch.nn.functional.interpolate(x, size=s, mode="bilinear", align_corners=False)
        return torch.nn.functional.interpolate(x, size=IMG, mode="bilinear", align_corners=False)

    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    for p in m.features.parameters():   # freeze backbone (avoids MPS backward bug)
        p.requires_grad = False
    m = m.to(dev)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    bs, epochs = 32, 14

    for ep in range(epochs):
        m.train()
        t0 = time.perf_counter()
        idx = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            j = idx[i:i + bs]
            xb = downscale(aug(Xtr[j].contiguous()))
            xb = norm(xb).to(dev)
            yb = ytr[j].to(dev)
            opt.zero_grad()
            loss = lossf(m(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        # val (clean)
        m.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(Xva), bs):
                xb = norm(Xva[i:i + bs]).to(dev)
                pv.append(m(xb).argmax(1).cpu())
            acc = (torch.cat(pv) == yva).float().mean().item()
        print(f"  epoch {ep+1}/{epochs} loss={tot/len(Xtr):.3f} val_acc={acc:.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)

    out = _MAIN / "data" / "models" / "color_MobileNetV3Small_cctv.pt"
    torch.save(m.state_dict(), out)
    print(f"saved {out}", flush=True)

    # CCTV reality check on cars with known true colour
    m.eval()
    truth = {"test_authorized.jpg": "Black", "test_unregistered.jpg": "White"}
    for fn, gt in truth.items():
        im = cv2.imread(str(_MAIN / "data" / "test" / fn))
        im = cv2.cvtColor(cv2.resize(im, (IMG, IMG)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        t = norm(torch.tensor(im.transpose(2, 0, 1)[None])).to(dev)
        with torch.no_grad():
            p = torch.softmax(m(t), 1)[0]
        i = int(p.argmax())
        print(f"CCTV {fn}: true={gt} pred={CLASSES[i]} ({float(p[i]):.2f})", flush=True)


if __name__ == "__main__":
    main()
