"""Recognition with alignment refinement – mirrors web/src/core/recognize.ts."""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .data import crop_digits
from .decode import Decoded, decode
from .geometry import DEFAULT_GEOMETRY, BoardGeometry


def needs_refine(d: Decoded, unsure: float) -> bool:
    return bool((d.tile_confidence < unsure).any() or d.corrected)


def recognize(img: np.ndarray, predict: Callable[[np.ndarray], np.ndarray],
              geometry: BoardGeometry = DEFAULT_GEOMETRY, refine: bool = True, unsure: float = 0.75,
              interpolation: int = cv2.INTER_LINEAR) -> tuple[Decoded, BoardGeometry, int]:
    """Returns (decoded board, geometry used, number of model passes)."""
    best_g = geometry
    best = decode(predict(crop_digits(img, geometry, interpolation=interpolation)))
    passes = 1
    if refine and needs_refine(best, unsure):
        for step in (0.02, 0.01):
            c = best_g
            for g in (c.transformed(dx=step), c.transformed(dx=-step), c.transformed(dy=step),
                      c.transformed(dy=-step), c.transformed(scale=1 + 2 * step), c.transformed(scale=1 - 2 * step)):
                d = decode(predict(crop_digits(img, g, interpolation=interpolation)))
                passes += 1
                if d.log_likelihood > best.log_likelihood + 0.5:
                    best, best_g = d, g
            if not needs_refine(best, unsure):
                break
    return best, best_g, passes
