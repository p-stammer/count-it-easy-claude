"""ONNX inference wrappers (same models the web app runs)."""
from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort

from .data import to_tensor_batch


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class DigitModel:
    """Single 9-logit network (cie.model.DigitNet exported to ONNX)."""

    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        """crops (19, 3, H, W, 3) uint8 -> probs (19, 3, 3)."""
        logits = self.sess.run(None, {self.input: to_tensor_batch(crops)})[0].reshape(19, 3, 9)
        per_dir = np.stack([logits[:, d, 3 * d:3 * d + 3] for d in range(3)], axis=1)
        return _softmax(per_dir)


class LegacyModels:
    """The three original 3-class Keras/TFLite models, converted to ONNX."""

    def __init__(self, directory: str):
        self.sess = []
        for g in ("159", "267", "348"):
            s = ort.InferenceSession(os.path.join(directory, f"model{g}.onnx"), providers=["CPUExecutionProvider"])
            self.sess.append((s, s.get_inputs()[0].name))

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        out = []
        for d, (s, name) in enumerate(self.sess):
            x = to_tensor_batch(crops[:, d])
            out.append(np.concatenate([s.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))]))
        return np.stack(out, axis=1)  # already softmax
