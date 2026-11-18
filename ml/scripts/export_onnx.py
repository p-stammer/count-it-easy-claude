"""Export a trained checkpoint to ONNX for the web app and verify it.

    python ml/scripts/export_onnx.py runs/v2/best.pt --version 2.0
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch

from cie.data import repo_root
from cie.model import DigitNet

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--version", required=True)
ap.add_argument("--width", type=int, default=1)
ap.add_argument("--out", default=os.path.join(repo_root(), "web", "public", "models"))
ap.add_argument("--metrics", help="evaluation json to embed in model.json")
a = ap.parse_args()

model = DigitNet(a.width)
model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
model.eval()
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "digits.onnx")
x = torch.rand(57, 3, 128, 128)
torch.onnx.export(model, x, path, input_names=["crops"], output_names=["logits"],
                  dynamic_axes={"crops": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
diff = np.abs(sess.run(None, {"crops": x.numpy()})[0] - model(x).detach().numpy()).max()
assert diff < 1e-4, diff
info = {"version": a.version, "input": [3, 128, 128], "outputs": "9 logits: [1,5,9 | 2,6,7 | 3,4,8]",
        "bytes": os.path.getsize(path)}
if a.metrics:
    with open(a.metrics) as f:
        info["test_metrics"] = json.load(f)["mean"]
with open(os.path.join(a.out, "model.json"), "w") as f:
    json.dump(info, f, indent=2)
print(f"exported {path} ({info['bytes']} bytes), max diff {diff:.2e}")
