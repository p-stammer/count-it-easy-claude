"""Board-level evaluation under realistic capture conditions."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import zlib

import cv2
import numpy as np

from .augment import degrade, glare, sheen
from .data import Board, crop_digits
from .decode import decode
from .recognize import recognize
from .geometry import DEFAULT_GEOMETRY, BoardGeometry
from .scoring import score

# crops (19, 3, 128, 128, 3) uint8 -> probabilities (19, 3, 3)
PredictFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Condition:
    name: str
    geometry: BoardGeometry = DEFAULT_GEOMETRY
    photometric: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None
    seed: int = 0


def _glare_only(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32) / 255
    f = glare(f, rng)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def _strong_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32) / 255
    f = sheen(glare(glare(f, rng), rng), rng)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def _dark_blurry(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = (img.astype(np.float32) / 255) ** 1.6 * 0.7
    img = (f * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return degrade(img, np.random.default_rng(1000 + int(rng.integers(1000))))


def standard_conditions() -> list[Condition]:
    g = DEFAULT_GEOMETRY
    conds = [Condition("clean")]
    for d in (0.02, 0.035):
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            conds.append(Condition(f"shift {d:.3f}", g.transformed(dx=sx * d, dy=sy * d)))
    for s in (0.94, 1.06, 0.9, 1.1):
        conds.append(Condition(f"scale {s}", g.transformed(scale=s)))
    for r in (-3.0, 3.0):
        conds.append(Condition(f"rotate {abs(r)}", g.transformed(rotation=r)))
    for seed in range(4):
        conds.append(Condition("glare", photometric=_glare_only, seed=seed))
        conds.append(Condition("strong glare", photometric=_strong_glare, seed=100 + seed))
    for seed in range(2):
        conds.append(Condition("dark+blurry", photometric=_dark_blurry, seed=200 + seed))
    return conds


def evaluate(predict: PredictFn, boards: list[Board], conditions: list[Condition] | None = None,
             interpolation: int = cv2.INTER_LINEAR, refine: bool = False,
             unsure: float = 0.75) -> dict[str, dict[str, float]]:
    """Per condition: digit accuracy, fully correct boards before / after the
    uniqueness constraint, correct score, and (with refine=True) the app's
    full pipeline including alignment refinement. 'flag_recall' is the share
    of wrongly decoded tiles the app marks as unsure."""
    conditions = conditions or standard_conditions()
    acc: dict[str, dict[str, list[float]]] = {}
    for b in boards:
        truth = b.labels()
        for ci, c in enumerate(conditions):
            img = b.image
            if c.photometric is not None:
                rng = np.random.default_rng(zlib.crc32(f"{b.id}:{c.seed}".encode()))
                img = c.photometric(img, rng)
            probs = predict(crop_digits(img, c.geometry, interpolation=interpolation))
            dec = decode(probs)
            if refine:
                dec, _, _ = recognize(img, predict, c.geometry, unsure=unsure, interpolation=interpolation)
            naive_ok = probs.argmax(-1) == truth
            m = acc.setdefault(c.name, {"digit_acc": [], "board_naive": [], "board_decoded": [],
                                        "score_ok": [], "mean_conf": [], "flagged": [], "wrong_tiles": [],
                                        "wrong_flagged": []})
            wrong = np.array([dec.board[i] != b.tiles[i] for i in range(len(b.tiles))])
            flagged = dec.tile_confidence < unsure
            m["flagged"].append(float(flagged.sum()))
            m["wrong_tiles"].append(float(wrong.sum()))
            m["wrong_flagged"].append(float((wrong & flagged).sum()))
            m["digit_acc"].append(naive_ok.mean())
            m["board_naive"].append(float(naive_ok.all()))
            m["board_decoded"].append(float(dec.board == b.tiles))
            m["score_ok"].append(float(dec.score == score(b.tiles)))
            m["mean_conf"].append(float(dec.tile_confidence.mean()))
    out = {}
    for k, v in acc.items():
        r = {mk: float(np.mean(mv)) for mk, mv in v.items() if mk not in ("wrong_tiles", "wrong_flagged")}
        wt = sum(v["wrong_tiles"])
        r["flag_recall"] = float(sum(v["wrong_flagged"]) / wt) if wt else 1.0
        out[k] = r
    return out


def summarize(results: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = next(iter(results.values())).keys()
    return {k: float(np.mean([r[k] for r in results.values()])) for k in keys}


def format_table(results: dict[str, dict[str, float]]) -> str:
    lines = [f"{'condition':<16}{'digit acc':>10}{'board naive':>13}{'board final':>13}{'score ok':>10}{'flags/board':>12}{'err flagged':>12}"]
    rows = list(results.items()) + [("MEAN", summarize(results))]
    for k, v in rows:
        lines.append(f"{k:<16}{v['digit_acc']:>10.4f}{v['board_naive']:>13.3f}{v['board_decoded']:>13.3f}"
                     f"{v['score_ok']:>10.3f}{v['flagged']:>12.2f}{v['flag_recall']:>12.2f}")
    return "\n".join(lines)
