"""Export the shared geometry and cross-language test fixtures for the web app.

    python ml/scripts/export_web_assets.py
"""
import json
import os

import numpy as np

from cie.data import repo_root
from cie.decode import decode
from cie.geometry import DEFAULT_GEOMETRY, export_json
from cie.scoring import parse_tile

root = repo_root()
web = os.path.join(root, "web", "src", "core")
export_json(os.path.join(web, "geometry.json"))

with open(os.path.join(root, "data", "boards.json")) as f:
    boards = json.load(f)["boards"]
rng = np.random.default_rng(42)
cases = []
for i in range(25):
    p = rng.dirichlet([0.6, 0.6, 0.6], size=(19, 3))
    d = decode(p)
    cases.append({"probs": p.round(6).tolist(), "board": ["-".join(map(str, t)) for t in d.board],
                  "score": d.score})
fixtures = {
    "boards": [{"id": b["id"], "tiles": b["tiles"], "score": b["score"]} for b in boards],
    "digitBoxes": DEFAULT_GEOMETRY.digit_boxes().round(6).tolist(),
    "shiftedBoxes": DEFAULT_GEOMETRY.transformed(dx=0.01, dy=-0.02, scale=1.05, rotation=2).digit_boxes().round(6).tolist(),
    "decodeCases": cases,
}
with open(os.path.join(web, "__fixtures__", "python_reference.json"), "w") as f:
    json.dump(fixtures, f)
print("exported geometry.json and fixtures")
