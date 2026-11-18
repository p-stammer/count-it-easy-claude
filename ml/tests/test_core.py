import json
import os

import numpy as np
import pytest

from cie.data import repo_root
from cie.decode import decode
from cie.geometry import DEFAULT_GEOMETRY, N_TILES, TILES, VALUES, lines
from cie.scoring import all_tiles, line_scores, parse_tile, score


def boards():
    with open(os.path.join(repo_root(), "data", "boards.json")) as f:
        return json.load(f)["boards"]


def test_tile_layout():
    assert N_TILES == 19
    assert [sum(1 for q, _ in TILES if q == c) for c in range(-2, 3)] == [3, 4, 5, 4, 3]
    for d in range(3):
        assert sorted(len(line) for line in lines(d)) == [3, 3, 4, 4, 5]
        assert sorted(i for line in lines(d) for i in line) == list(range(19))


def test_27_unique_tiles():
    assert len(set(all_tiles())) == 27


def test_max_score_board():
    # the theoretical maximum is 307 (known result for Take it Easy!)
    b = next(b for b in boards() if b["id"] == "t5")
    assert b["score"] == 307


@pytest.mark.parametrize("bid,expected", [("t1", 170), ("t2", 181), ("t3", 167), ("t4", 245), ("t5", 307)])
def test_scores_match_original_notebook(bid, expected):
    b = next(b for b in boards() if b["id"] == bid)
    assert score([parse_tile(t) for t in b["tiles"]]) == expected


def test_all_boards_consistent():
    for b in boards():
        tiles = [parse_tile(t) for t in b["tiles"]]
        assert len(tiles) == 19 and len(set(tiles)) == 19, b["id"]
        assert score(tiles) == b["score"], b["id"]


def test_line_scores_uniform_line():
    board = [(1, 2, 3)] * 19
    ls = line_scores(board)
    assert ls[0] == [3, 4, 5, 4, 3]
    assert sum(ls[1]) == 2 * 19 and sum(ls[2]) == 3 * 19


def test_parse_tile_rejects_invalid():
    with pytest.raises(ValueError):
        parse_tile("2-1-3")


def _onehot(board, conf=0.98):
    p = np.full((19, 3, 3), (1 - conf) / 2)
    for i, t in enumerate(board):
        for d in range(3):
            p[i, d, VALUES[d].index(t[d])] = conf
    return p


def test_decode_identity():
    b = next(b for b in boards() if b["id"] == "t5")
    tiles = [parse_tile(t) for t in b["tiles"]]
    dec = decode(_onehot(tiles))
    assert dec.board == tiles and dec.corrected == [] and dec.score == 307


def test_decode_repairs_duplicate_with_uncertain_digit():
    b = next(b for b in boards() if b["id"] == "t5")
    tiles = [parse_tile(t) for t in b["tiles"]]
    # find two placed tiles that differ in exactly one direction
    i, j, d = next((i, j, [k for k in range(3) if tiles[i][k] != tiles[j][k]][0])
                   for i in range(19) for j in range(19)
                   if i != j and sum(tiles[i][k] != tiles[j][k] for k in range(3)) == 1)
    p = _onehot(tiles)
    p[i, d] = 0.0
    p[i, d, VALUES[d].index(tiles[j][d])] = 0.55  # misread: now a duplicate of tile j
    p[i, d, VALUES[d].index(tiles[i][d])] = 0.45
    dec = decode(p)
    assert dec.naive[i] == tiles[j]
    assert dec.board == tiles and dec.corrected == [i]


def test_geometry_inside_frame():
    boxes = DEFAULT_GEOMETRY.digit_boxes()
    assert boxes.shape == (19, 3, 4)
    c = DEFAULT_GEOMETRY.digit_centers()
    assert (c > 0).all() and (c < 1).all()
    poly = DEFAULT_GEOMETRY.tile_polygons()
    assert poly.min() > -0.01 and poly.max() < 1.01
