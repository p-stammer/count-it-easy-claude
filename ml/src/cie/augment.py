"""Photometric augmentations that mimic real-world capture problems.

Applied to the *whole board image* before cropping, so that e.g. a specular
reflection spreads realistically over several neighbouring digits.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gauss_blob(h: int, w: int, rng: np.random.Generator, min_sigma: float, max_sigma: float) -> np.ndarray:
    """Random rotated Gaussian blob, computed at low resolution for speed."""
    sh, sw = max(8, h // 8), max(8, w // 8)
    return cv2.resize(_blob_lowres(sh, sw, rng, min_sigma, max_sigma), (w, h), interpolation=cv2.INTER_LINEAR)


def _blob_lowres(h: int, w: int, rng: np.random.Generator, min_sigma: float, max_sigma: float) -> np.ndarray:
    cy, cx = rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w
    sy, sx = rng.uniform(min_sigma, max_sigma, size=2) * min(h, w)
    a = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx - cx, yy - cy
    xr, yr = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)
    return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)).astype(np.float32)


def glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Specular reflections: bright, partly saturated blobs (screen blend)."""
    h, w = img.shape[:2]
    out = img
    for _ in range(rng.integers(1, 4)):
        g = _gauss_blob(h, w, rng, 0.03, 0.18)
        # sharpen the core so the centre saturates like a real reflection
        g = np.clip(g * rng.uniform(1.0, 2.5), 0, 1)[..., None]
        strength = rng.uniform(0.5, 1.0)
        tint = 1.0 - rng.uniform(0, 0.08, size=3).astype(np.float32)
        out = 1 - (1 - out) * (1 - strength * g * tint)
    return out


def sheen(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Broad, low-contrast reflection (e.g. a lamp mirrored in the board)."""
    g = _gauss_blob(*img.shape[:2], rng, 0.25, 0.6)[..., None]
    return 1 - (1 - img) * (1 - rng.uniform(0.15, 0.45) * g)


def shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = _gauss_blob(*img.shape[:2], rng, 0.1, 0.5)[..., None]
    return img * (1 - rng.uniform(0.2, 0.6) * g)


def color_light(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img * rng.uniform(0.85, 1.15, size=3).astype(np.float32)  # white balance
    img = (img - 0.5) * rng.uniform(0.7, 1.3) + 0.5 + rng.uniform(-0.15, 0.15)  # contrast / brightness
    img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.4)  # gamma
    return img


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blur, low resolution, sensor noise, JPEG artefacts (uint8 in/out)."""
    h, w = img.shape[:2]
    big = min(h, w) >= 600  # never destroy already low-resolution photos
    if big and rng.random() < 0.3:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if big and rng.random() < 0.2:  # motion blur
        k = int(rng.integers(5, 13))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(ker, rot, (k, k)))
    if big and rng.random() < 0.3:  # low-resolution camera frame (board >= ~350 px)
        f = rng.uniform(max(0.35, 350 / min(h, w)), 0.8)
        small = cv2.resize(img, (max(32, int(w * f)), max(32, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        noise = rng.normal(0, rng.uniform(2, 10), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        q = int(rng.integers(35, 90))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_UNCHANGED)
    return img


def augment_board(img_u8: np.ndarray, rng: np.random.Generator, p_glare: float = 0.5) -> np.ndarray:
    """Full photometric pipeline for an RGB uint8 board image."""
    img = img_u8.astype(np.float32) / 255
    img = color_light(img, rng)
    if rng.random() < p_glare:
        img = glare(img, rng)
    if rng.random() < 0.25:
        img = sheen(img, rng)
    if rng.random() < 0.25:
        img = shadow(img, rng)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return degrade(img, rng)
