"""Evaluate digit models on full boards under realistic capture conditions.

    python ml/scripts/evaluate.py --model web/public/models/digits.onnx
    python ml/scripts/evaluate.py --legacy ml/models/legacy --interp nearest
"""
import argparse
import json

import cv2

from cie.data import load_boards
from cie.evaluate import evaluate, format_table, summarize
from cie.inference import DigitModel, LegacyModels

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--model", help="9-logit ONNX model")
g.add_argument("--legacy", help="directory with model159/267/348.onnx")
ap.add_argument("--split", default="test")
ap.add_argument("--interp", default="linear", choices=["linear", "nearest", "area"])
ap.add_argument("--json", help="write results to this file")
ap.add_argument("--refine", action="store_true", help="include the app's alignment refinement")
ap.add_argument("--unsure", type=float, default=0.75, help="tile confidence below which the app flags a tile")
a = ap.parse_args()

predict = DigitModel(a.model) if a.model else LegacyModels(a.legacy)
boards = load_boards(splits=(a.split,), max_side=1024)
interp = {"linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA}[a.interp]
res = evaluate(predict, boards, interpolation=interp, refine=a.refine, unsure=a.unsure)
print(f"{len(boards)} {a.split} boards")
print(format_table(res))
if a.json:
    with open(a.json, "w") as f:
        json.dump({"results": res, "mean": summarize(res)}, f, indent=2)
