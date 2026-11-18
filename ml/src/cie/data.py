"""Datasets: full-board photos (data/boards.json) and loose extra crops."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import DEFAULT_GEOMETRY, N_TILES, VALUES, BoardGeometry
from .scoring import parse_tile

INPUT_SIZE = 128
# global class id = 3 * direction + index inside direction
CLASS_OF_VALUE = {v: 3 * d + i for d, vals in enumerate(VALUES) for i, v in enumerate(vals)}


@dataclass
class Board:
    id: str
    image: np.ndarray        # RGB uint8, the canonical square-ish frame
    tiles: list[tuple[int, int, int]]
    split: str
    score: int

    def labels(self) -> np.ndarray:
        """(19, 3) class index inside each direction."""
        return np.array([[VALUES[d].index(t[d]) for d in range(3)] for t in self.tiles])


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_boards(data_dir: str | None = None, splits: tuple[str, ...] | None = None,
                max_side: int = 1280) -> list[Board]:
    data_dir = data_dir or os.path.join(repo_root(), "data")
    with open(os.path.join(data_dir, "boards.json")) as f:
        meta = json.load(f)["boards"]
    out = []
    for b in meta:
        if splits and b["split"] not in splits:
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(data_dir, b["file"])), cv2.COLOR_BGR2RGB)
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append(Board(b["id"], img, [parse_tile(t) for t in b["tiles"]], b["split"], b["score"]))
    return out


def load_extra_crops(data_dir: str | None = None) -> list[tuple[np.ndarray, int]]:
    """Loose crops without board context: (RGB uint8 image, global class id)."""
    root = os.path.join(data_dir or os.path.join(repo_root(), "data"), "extra_crops")
    out = []
    for value in sorted(os.listdir(root)):
        for f in sorted(os.listdir(os.path.join(root, value))):
            img = cv2.cvtColor(cv2.imread(os.path.join(root, value, f)), cv2.COLOR_BGR2RGB)
            out.append((img, CLASS_OF_VALUE[int(value)]))
    return out


def crop_digits(img: np.ndarray, geom: BoardGeometry = DEFAULT_GEOMETRY, size: int = INPUT_SIZE,
                interpolation: int = cv2.INTER_LINEAR, rng: np.random.Generator | None = None,
                box_jitter: float = 0.0) -> np.ndarray:
    """Cut the 57 digit crops -> (19, 3, size, size, 3) uint8.

    The frame is the full image; x is scaled by width and y by height, exactly
    like the app does with the captured square.
    """
    h, w = img.shape[:2]
    boxes = geom.digit_boxes().reshape(-1, 4)
    out = np.empty((len(boxes), size, size, 3), np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if rng is not None and box_jitter > 0:
            bw, bh = x1 - x0, y1 - y0
            dx, dy = rng.normal(0, box_jitter, 2) * [bw, bh]
            sc = 1 + rng.normal(0, box_jitter)
            cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
            x0, x1, y0, y1 = cx - bw * sc / 2, cx + bw * sc / 2, cy - bh * sc / 2, cy + bh * sc / 2
        # affine map box -> size x size (handles boxes partly outside the image)
        src = np.float32([[x0 * w, y0 * h], [x1 * w, y0 * h], [x0 * w, y1 * h]])
        dst = np.float32([[0, 0], [size, 0], [0, size]])
        m = cv2.getAffineTransform(src, dst)
        out[i] = cv2.warpAffine(img, m, (size, size), flags=interpolation, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(N_TILES, 3, size, size, 3)


def to_tensor_batch(crops_u8: np.ndarray) -> np.ndarray:
    """(..., H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1] (model input)."""
    x = crops_u8.reshape(-1, *crops_u8.shape[-3:]).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
