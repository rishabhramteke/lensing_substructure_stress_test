"""Loads one generated population (see ../../data/README.md) as a torch Dataset
of (normalized image, per-pixel mask, truth dict) triples."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import make_mask


def load_population(path: Path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    truths = [json.loads(l) for l in open(path / "truth.jsonl")]
    noisy_path = path / "images_full_noisy.npy"
    images = np.load(noisy_path) if noisy_path.exists() else np.load(path / "images_full_noiseless.npy")
    pixel_scale = manifest["kwargs_band"]["pixel_scale"]
    num_pix = manifest["image_shape"][0]
    return images, truths, pixel_scale, num_pix, manifest


def normalize(images: np.ndarray, scale: float) -> np.ndarray:
    """arcsinh stretch, matching the convention already used for the
    web-demo/sanity visualisations elsewhere in this project."""
    return np.arcsinh(np.clip(images, 0, None) / scale).astype(np.float32)


class LensPatchDataset(Dataset):
    def __init__(self, path: Path, scale: float | None = None, mask_radius_px: float = 2.0):
        images, truths, pixel_scale, num_pix, manifest = load_population(path)
        self.truths = truths
        self.pixel_scale = pixel_scale
        self.num_pix = num_pix
        self.manifest = manifest
        self.scale = scale if scale is not None else float(np.percentile(images[images > 0], 90))
        self.images = normalize(images, self.scale)
        self.masks = np.stack([make_mask(t, num_pix, pixel_scale, mask_radius_px) for t in truths]).astype(np.float32)

    def __len__(self):
        return len(self.truths)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).unsqueeze(0)
        mask = torch.from_numpy(self.masks[i]).unsqueeze(0)
        has_sub = float(self.truths[i].get("subhalo") is not None)
        return img, mask, has_sub, i
