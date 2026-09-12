"""World (arcsec) -> pixel coordinates, and the per-pixel training mask.

Coordinate convention verified directly against lenstronomy's own
`ImageData.map_coord2pix` for this num_pix/pixel_scale (2026-09-10): the grid
is centered at pixel ((num_pix-1)/2, (num_pix-1)/2) for world (0,0), with a
simple isotropic scaling of 1/pixel_scale and no rotation or flip. We
replicate that affine map directly here (cheaper than instantiating a data
class per sample) rather than depend on the assumption silently drifting.
"""
from __future__ import annotations

import numpy as np


def world_to_pixel(x_arcsec, y_arcsec, num_pix: int, pixel_scale: float):
    center = (num_pix - 1) / 2.0
    return center + np.asarray(x_arcsec) / pixel_scale, center + np.asarray(y_arcsec) / pixel_scale


def make_mask(truth: dict, num_pix: int, pixel_scale: float, radius_px: float = 2.0) -> np.ndarray:
    """A small disk around the subhalo's true pixel position, radius_px matching
    Ostdiek+2020's own success criterion ("the center of the subhalo can be at
    most 2 pixels off from the true center"). All-zero if no subhalo."""
    mask = np.zeros((num_pix, num_pix), dtype=np.float32)
    sub = truth.get("subhalo")
    if not sub:
        return mask
    px, py = world_to_pixel(sub["x"], sub["y"], num_pix, pixel_scale)
    yy, xx = np.mgrid[0:num_pix, 0:num_pix]
    mask[((xx - px) ** 2 + (yy - py) ** 2) <= radius_px**2] = 1.0
    return mask
