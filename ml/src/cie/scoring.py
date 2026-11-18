"""Scoring rules: a complete line of equal values scores value x line length."""
from __future__ import annotations

from collections.abc import Sequence

from .geometry import N_TILES, VALUES, lines

Tile = tuple[int, int, int]  # printed values (1/5/9, 2/6/7, 3/4/8)


def all_tiles() -> list[Tile]:
    """The 27 physical tiles of the game: every combination exists exactly once."""
    return [(a, b, c) for a in VALUES[0] for b in VALUES[1] for c in VALUES[2]]


def line_scores(board: Sequence[Tile]) -> list[list[int]]:
    """Points of each line, per direction (0 if the line is not uniform)."""
    if len(board) != N_TILES:
        raise ValueError(f"expected {N_TILES} tiles, got {len(board)}")
    out = []
    for d in range(3):
        row = []
        for line in lines(d):
            vals = {board[i][d] for i in line}
            row.append(vals.pop() * len(line) if len(vals) == 1 else 0)
        out.append(row)
    return out


def score(board: Sequence[Tile]) -> int:
    return sum(sum(r) for r in line_scores(board))


def parse_tile(s: str) -> Tile:
    a, b, c = (int(x) for x in s.split("-"))
    if a not in VALUES[0] or b not in VALUES[1] or c not in VALUES[2]:
        raise ValueError(f"invalid tile {s!r}")
    return (a, b, c)
