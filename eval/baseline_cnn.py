#!/usr/bin/env python3
"""A plain image classifier as a baseline for the inspector: "why not a simple CNN?"

ResNet-18 (ImageNet weights) fine-tuned on the same robot-cell photos the inspector was
retrained on (finetune/data/sort_train.jsonl), one network per camera:
  top        the close-up photo on the nest     -> none | wrong_part | defective_part
  underside  the upward camera's photo          -> none | wrong_part | defective_part
Fixed recipe, no choices made on the dev set. Writes generate.py-style answers, so the
per-part scoring is the same as for the inspector.

    ~/cosmos_t4/.venv/bin/python eval/baseline_cnn.py --out finetune/data/gen/gen_cnn_sortdev.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ("none", "wrong_part", "defective_part")
REPO = Path(__file__).resolve().parent.parent


def load(path: Path, view: str):
    rows = [json.loads(l) for l in path.open()]
    rows = [r for r in rows if r["view"] == view]
    # The top example carries overview + close-up; the close-up is the one that shows the part.
    return [(r["seed"], r["images"][-1], CLASSES.index(r["expected"]["failure_type"])) for r in rows]


class Photos(torch.utils.data.Dataset):
    def __init__(self, items, root: Path, tf):
        self.items, self.root, self.tf = items, root, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        seed, img, y = self.items[i]
        return self.tf(Image.open(self.root / img).convert("RGB")), y


def zoom(img: Image.Image) -> Image.Image:
    """The middle half of the 1024x768 photo, where the nut is: 2x the pixels on the part."""
    w, h = img.size
    return img.crop((w // 4, h // 4, 3 * w // 4, h // 4 + h // 2))


def train_view(view: str, args, device) -> list[dict]:
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    first = [transforms.Lambda(zoom)] if args.zoom else []
    tf_train = transforms.Compose(first + [transforms.Resize((240, 320)), transforms.ColorJitter(0.2, 0.2, 0.1),
                                   transforms.RandomAffine(4, translate=(0.03, 0.03)), transforms.ToTensor(), norm])
    tf_eval = transforms.Compose(first + [transforms.Resize((240, 320)), transforms.ToTensor(), norm])
    train = load(args.train, view)
    dev = load(args.dev, view)
    counts = torch.bincount(torch.tensor([y for *_, y in train]), minlength=len(CLASSES)).float()
    weights = (counts.sum() / (len(CLASSES) * counts)).to(device)  # balance the 4:1:1 classes

    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(Photos(train, args.images, tf_train), batch_size=32, shuffle=True,
                                         num_workers=args.workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for ep in range(args.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        print(f"[cnn] {view} epoch {ep + 1}/{args.epochs} loss {total / len(train):.4f} {time.time() - t0:.0f}s",
              flush=True)

    net.eval()
    out = []
    with torch.no_grad():
        for seed, img, _ in dev:
            t0 = time.time()
            x = tf_eval(Image.open(args.images / img).convert("RGB")).unsqueeze(0).to(device)
            probs = torch.softmax(net(x), 1)[0]
            pred = CLASSES[int(probs.argmax())]
            raw = json.dumps({"checkpoint": "CP_PRE_GRASP", "step_ok": pred == "none", "failure_type": pred,
                              "probs": {c: round(float(q), 4) for c, q in zip(CLASSES, probs)}})
            out.append({"seed": seed, "raw": raw, "latency_ms": round((time.time() - t0) * 1000)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=REPO / "finetune/data/sort_train.jsonl")
    ap.add_argument("--dev", type=Path, default=REPO / "finetune/data/sort_dev.jsonl")
    ap.add_argument("--images", type=Path, default=REPO / "runs/sorting")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (leave the rest to Isaac Sim)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", action="store_true", help="crop to the middle of the photo (the nut) first")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rows = train_view("top", args, args.device) + train_view("underside", args, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[cnn] DONE {len(rows)} answers -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
