"""Turn per-digit class probabilities into a consistent board.

Every one of the 27 tiles exists exactly once, so the 19 placed tiles must be
distinct. We pick the assignment of physical tiles to board positions that
maximises the joint log-likelihood (Hungarian algorithm). This repairs single
misread digits whenever the naive reading would produce a duplicate tile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import N_TILES, VALUES
from .scoring import Tile, all_tiles, score

_TILE_IDX = np.array([[VALUES[0].index(a), VALUES[1].index(b), VALUES[2].index(c)]
                      for a, b, c in all_tiles()])  # (27, 3)


@dataclass
class Decoded:
    board: list[Tile]
    naive: list[Tile]
    tile_confidence: np.ndarray  # (19,) probability of the chosen tile
    score: int
    corrected: list[int]          # positions where the constraint changed the reading
    log_likelihood: float


def decode(probs: np.ndarray, eps: float = 1e-6) -> Decoded:
    """probs: (19, 3, 3) – tile, direction, class-in-direction."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (N_TILES, 3, 3):
        raise ValueError(f"expected (19, 3, 3), got {probs.shape}")
    logp = np.log(np.clip(probs, eps, 1.0))
    # cost[position, tile] = -log P(tile at position)
    ll = logp[:, 0, _TILE_IDX[:, 0]] + logp[:, 1, _TILE_IDX[:, 1]] + logp[:, 2, _TILE_IDX[:, 2]]
    rows, cols = linear_sum_assignment(-ll)
    tiles = all_tiles()
    board = [tiles[c] for c in cols[np.argsort(rows)]]
    am = probs.argmax(-1)
    naive = [tuple(VALUES[d][am[i, d]] for d in range(3)) for i in range(N_TILES)]
    conf = np.exp(ll[rows, cols])[np.argsort(rows)]
    corrected = [i for i in range(N_TILES) if board[i] != naive[i]]
    return Decoded(board, naive, conf, score(board), corrected, float(ll[rows, cols].sum()))
