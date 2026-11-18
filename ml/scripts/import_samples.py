"""Turn photos exported from the app (Settings -> "Fotos exportieren") into
training boards: each photo is warped into the canonical frame using the
alignment that was used for counting, and its (user-corrected) tiles become
the ground truth.

    python ml/scripts/import_samples.py cie-samples-2026-11-20.json --split train
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np

from cie.data import repo_root
from cie.geometry import DEFAULT_GEOMETRY, BoardGeometry
from cie.scoring import parse_tile, score

PAD = 0.15  # must match web/src/core/frame.ts
SIZE = 1024

ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("--split", default="train", choices=["train", "val", "test"])
a = ap.parse_args()

root = os.path.join(repo_root(), "data")
meta_path = os.path.join(root, "boards.json")
with open(meta_path) as f:
    meta = json.load(f)
with open(a.export) as f:
    samples = json.load(f)["samples"]
known = {b["id"] for b in meta["boards"]}
added = 0
for s in samples:
    bid = "app-" + s["id"][:8]
    if bid in known:
        continue
    raw = base64.b64decode(s["image"].split(",", 1)[1])
    work = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    g = BoardGeometry(**s["geometry"])
    # work image covers frame [-PAD, 1+PAD]; map the counted geometry onto the default one
    n = work.shape[0]
    to_px = lambda p: (p + PAD) / (1 + 2 * PAD) * n  # noqa: E731
    src = np.float32([to_px(c) for c in g.tile_centers()[[0, 9, 18]]])
    dst = np.float32([c * SIZE for c in DEFAULT_GEOMETRY.tile_centers()[[0, 9, 18]]])
    frame = cv2.warpAffine(work, cv2.getAffineTransform(src, dst), (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(root, "boards", f"{bid}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    tiles = [parse_tile(t) for t in s["tiles"]]
    meta["boards"].append({"id": bid, "file": f"boards/{bid}.jpg", "split": a.split, "tiles": s["tiles"],
                           "score": score(tiles), "source": "app", "predicted": s["predicted"]})
    added += 1
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=1)
print(f"added {added} boards")
