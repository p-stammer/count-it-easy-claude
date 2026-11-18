"""Train the digit classifier from full-board photos.

    python ml/scripts/train.py --out ml/runs/v2 --epochs 60

Every epoch re-renders each training board several times with a random
misalignment (shift / scale / rotation of the whole board, as happens when the
camera overlay is not matched perfectly) and random photometric damage
(reflections, shadows, white balance, blur, noise, JPEG), then cuts the 57
digit crops with exactly the same code the evaluation uses.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cie.augment import augment_board, degrade
from cie.data import CLASS_OF_VALUE, crop_digits, load_boards, load_extra_crops, to_tensor_batch
from cie.evaluate import evaluate, format_table, summarize
from cie.geometry import DEFAULT_GEOMETRY
from cie.model import DigitNet, direction_logits

INTERPS = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC]


class BoardVariants(Dataset):
    """Item = one augmented rendering of one board -> 57 crops + labels."""

    def __init__(self, boards, variants_per_board, extra, seed=0):
        self.boards, self.k, self.extra, self.seed = boards, variants_per_board, extra, seed
        self.epoch = 0
        self.n_board_items = len(boards) * variants_per_board
        self.extra_chunks = [list(range(i, len(extra), 57)) for i in range(57)] if extra else []

    def __len__(self):
        return self.n_board_items + (3 if self.extra else 0)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        if i >= self.n_board_items:  # a batch of loose extra crops
            idx = rng.choice(len(self.extra), 57, replace=False)
            xs, ys = [], []
            for j in idx:
                img, cls = self.extra[j]
                img = cv2.resize(img, (128, 128), interpolation=INTERPS[rng.integers(2)])
                img = augment_board(img, rng, p_glare=0.4)
                xs.append(img); ys.append(cls)
            return to_tensor_batch(np.stack(xs)), np.array(ys) // 3, np.array(ys) % 3
        b = self.boards[i // self.k]
        img = b.image
        if rng.random() < 0.9:
            img = augment_board(img, rng, p_glare=0.6)
        geom = DEFAULT_GEOMETRY.transformed(
            dx=float(np.clip(rng.normal(0, 0.015), -0.04, 0.04)),
            dy=float(np.clip(rng.normal(0, 0.015), -0.04, 0.04)),
            scale=float(np.clip(rng.normal(1, 0.035), 0.88, 1.12)),
            rotation=float(np.clip(rng.normal(0, 1.5), -4, 4)))
        crops = crop_digits(img, geom, interpolation=INTERPS[rng.integers(len(INTERPS))],
                            rng=rng, box_jitter=0.03)
        lab = b.labels()  # (19, 3)
        direction = np.broadcast_to(np.arange(3), (19, 3)).ravel()
        return to_tensor_batch(crops), direction.copy(), lab.ravel()


def collate(items):
    x = torch.from_numpy(np.concatenate([i[0] for i in items]))
    d = torch.from_numpy(np.concatenate([i[1] for i in items])).long()
    y = torch.from_numpy(np.concatenate([i[2] for i in items])).long()
    return x, d, y


def make_predict(model):
    @torch.no_grad()
    def predict(crops):
        model.eval()
        logits = model(torch.from_numpy(to_tensor_batch(crops))).reshape(19, 3, 9)
        per_dir = torch.stack([logits[:, d, 3 * d:3 * d + 3] for d in range(3)], 1)
        return torch.softmax(per_dir, -1).numpy()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--variants", type=int, default=6, help="renderings per board per epoch")
    ap.add_argument("--batch-boards", type=int, default=3, help="board renderings per batch (x57 crops)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    train_boards = load_boards(splits=("train",), max_side=1024)
    val_boards = load_boards(splits=("val",), max_side=1024)
    extra = load_extra_crops()
    ds = BoardVariants(train_boards, a.variants, extra, a.seed)
    dl = DataLoader(ds, batch_size=a.batch_boards, shuffle=True, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0)
    model = DigitNet(a.width)
    print(f"{len(train_boards)} train boards, {len(val_boards)} val boards, {len(extra)} extra crops, "
          f"{sum(p.numel() for p in model.parameters())} params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    predict = make_predict(model)
    best, log = -1.0, []
    for ep in range(a.epochs):
        ds.epoch = ep
        model.train(); t0 = time.time(); tot = n = correct = 0
        for x, d, y in dl:
            logits = direction_logits(model(x), d)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y); correct += (logits.argmax(1) == y).sum().item()
        entry = {"epoch": ep, "loss": tot / n, "train_acc": correct / n, "sec": time.time() - t0}
        if ep % 5 == 4 or ep == a.epochs - 1:
            res = evaluate(predict, val_boards)
            s = summarize(res)
            entry.update({"val_" + k: v for k, v in s.items()})
            metric = s["board_decoded"] + s["digit_acc"]
            if metric > best:
                best = metric
                torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
                entry["saved"] = True
        log.append(entry); print(json.dumps(entry), flush=True)
        with open(os.path.join(a.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
    torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    print(format_table(evaluate(predict, val_boards)))


if __name__ == "__main__":
    main()
