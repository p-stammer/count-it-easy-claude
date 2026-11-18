"""Board geometry for *Take it Easy!*.

The board is a hexagon of 19 flat-top hex tiles (columns of 3-4-5-4-3).
Everything here is expressed in a square, normalised *frame* (x, y in [0, 1],
y pointing down). The frame is what the user aligns with the camera overlay,
and what training crops are cut from, so camera, overlay, training and
inference all share one definition.

Tiles use axial hex coordinates (q, r) with s = -q - r:
  * q (column)  -> lines of the vertical  direction, values 1/5/9
  * s           -> lines rising to the right,       values 2/6/7
  * r           -> lines falling to the right,      values 3/4/8

Canonical tile order: column-major (q = -2..2), top to bottom inside a column.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

TILES: list[tuple[int, int]] = [
    (q, r) for q in range(-2, 3) for r in range(max(-2, -2 - q), min(2, 2 - q) + 1)
]
N_TILES = len(TILES)  # 19

# Digit values per direction, index = class id inside that direction.
VALUES: tuple[tuple[int, int, int], ...] = ((1, 5, 9), (2, 6, 7), (3, 4, 8))
# Angle (degrees, y down) of each direction's printed digit relative to the tile centre.
DIGIT_ANGLES = (-90.0, 150.0, 30.0)


def line_key(direction: int, q: int, r: int) -> int:
    """Identifier of the straight line a tile belongs to, per direction."""
    return (q, -q - r, r)[direction]


def lines(direction: int) -> list[list[int]]:
    """Tile indices of every line in a direction (5 lines of length 3,4,5,4,3)."""
    groups: dict[int, list[int]] = {}
    for i, (q, r) in enumerate(TILES):
        groups.setdefault(line_key(direction, q, r), []).append(i)
    return [groups[k] for k in sorted(groups)]


@dataclass(frozen=True)
class BoardGeometry:
    """Where the board sits inside the frame.

    Defaults were fitted (least squares, rms error 0.4 % of the frame) to the
    hand-made crop boxes used to build the original training data.
    """

    cx: float = 0.5          # board centre
    cy: float = 0.5
    radius: float = 0.1085   # hex circumradius (= tile centre spacing / sqrt(3))
    rotation: float = 0.0    # degrees, clockwise in image coordinates
    digit_radius: float = 0.60  # distance of printed digits from tile centre, in tile radii
    box_w: float = 1.66      # crop box size, in tile radii
    box_h: float = 1.24

    def transformed(self, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0,
                    rotation: float = 0.0) -> "BoardGeometry":
        d = asdict(self)
        d.update(cx=self.cx + dx, cy=self.cy + dy, radius=self.radius * scale,
                 rotation=self.rotation + rotation)
        return BoardGeometry(**d)

    def _to_frame(self, p: np.ndarray) -> np.ndarray:
        """Board units (tile radius = 1, origin = board centre) -> frame coords."""
        a = math.radians(self.rotation)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return p @ rot.T * self.radius + np.array([self.cx, self.cy])

    def tile_centers(self) -> np.ndarray:
        pts = np.array([[1.5 * q, math.sqrt(3) * (r + q / 2)] for q, r in TILES])
        return self._to_frame(pts)

    def tile_polygons(self) -> np.ndarray:
        """(19, 6, 2) hexagon corners in frame coordinates."""
        base = np.array([[1.5 * q, math.sqrt(3) * (r + q / 2)] for q, r in TILES])
        ang = np.radians(np.arange(6) * 60.0)
        corners = np.stack([np.cos(ang), np.sin(ang)], 1)
        return np.stack([self._to_frame(c + corners) for c in base])

    def digit_centers(self) -> np.ndarray:
        """(19, 3, 2): centre of each printed digit per tile and direction."""
        base = np.array([[1.5 * q, math.sqrt(3) * (r + q / 2)] for q, r in TILES])
        offs = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                         for a in DIGIT_ANGLES]) * self.digit_radius
        pts = base[:, None, :] + offs[None, :, :]
        return self._to_frame(pts.reshape(-1, 2)).reshape(N_TILES, 3, 2)

    def digit_boxes(self) -> np.ndarray:
        """(19, 3, 4) axis-aligned crop boxes (x0, y0, x1, y1) in frame coords."""
        c = self.digit_centers()
        hw, hh = self.box_w * self.radius / 2, self.box_h * self.radius / 2
        return np.concatenate([c - [hw, hh], c + [hw, hh]], axis=-1)

    def to_json(self) -> dict:
        return {
            "geometry": asdict(self),
            "tiles": TILES,
            "values": VALUES,
            "digitAngles": DIGIT_ANGLES,
            "lines": [lines(d) for d in range(3)],
        }


DEFAULT_GEOMETRY = BoardGeometry()


def export_json(path: str, geometry: BoardGeometry = DEFAULT_GEOMETRY) -> None:
    with open(path, "w") as f:
        json.dump(geometry.to_json(), f, indent=2)
