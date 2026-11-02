"""Batched, unit-normalized CLIP image and text embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class CLIPEmbedder:
    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION,
                 device: str = "cpu", batch_size: int = 32, threads: int = 4):
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device == "cpu":
            torch.set_num_threads(threads)
        self.model_id = model_id
        self.model_revision = revision
        self.device = device
        self.batch_size = batch_size
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id, revision=revision).to(device).eval()
        actual = getattr(self.model.config, "_commit_hash", None)
        if actual:
            self.model_revision = actual

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().float().cpu().numpy()
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)

    def texts(self, texts: Sequence[str]) -> np.ndarray:
        results = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self.processor(text=list(texts[start:start + self.batch_size]),
                                        return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                results.append(self._normalize(self.model.get_text_features(**inputs)))
        return np.concatenate(results) if results else np.empty((0, self.model.config.projection_dim), dtype=np.float32)

    def images(self, paths: Sequence[str | Path]) -> np.ndarray:
        results = []
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                opened = []
                try:
                    for path in paths[start:start + self.batch_size]:
                        opened.append(Image.open(path).convert("RGB"))
                    inputs = self.processor(images=opened, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    results.append(self._normalize(self.model.get_image_features(**inputs)))
                finally:
                    for image in opened:
                        image.close()
        return np.concatenate(results) if results else np.empty((0, self.model.config.projection_dim), dtype=np.float32)
