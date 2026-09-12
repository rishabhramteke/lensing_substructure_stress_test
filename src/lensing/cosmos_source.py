"""Tier-1 source: real COSMOS/HST galaxy postage stamps in place of the Tier-0
single-Sersic profile (`../problem_statement.md`'s "Tier 1 -- realistic
source: real COSMOS/HDF galaxies (via paltas/galsim)").

Uses `galsim.COSMOSCatalog` directly (the "23.5" training sample: ~56,000 HST
ACS/F814W galaxies at F814W<23.5, Zenodo record 3242143 -- the same catalog
`paltas` wraps, downloaded here since `paltas` itself does not install on
Python 3.14; see `../fulltext_findings.md`). Each draw is deconvolved from the
COSMOS/ACS PSF and reconvolved with a small regularizing Gaussian, then handed
to lenstronomy as a fine-pixel-scale image via the INTERPOL light profile --
lenstronomy performs the actual ray-tracing, instrument PSF convolution and
noise, so the source's only role is to supply realistic surface-brightness
structure (clumps, arms, asymmetry) in place of a smooth analytic profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_CATALOG = None
_CATALOG_SAMPLE = None


def _catalog(sample: str = "23.5"):
    global _CATALOG, _CATALOG_SAMPLE
    if _CATALOG is None or _CATALOG_SAMPLE != sample:
        import galsim
        _CATALOG = galsim.COSMOSCatalog(sample=sample)
        _CATALOG_SAMPLE = sample
    return _CATALOG


def n_objects(sample: str = "23.5") -> int:
    """Catalog size, needed by `simulate.sample_truth` to draw a valid index
    without duplicating the (cached) catalog load."""
    return _catalog(sample).nobjects


@dataclass
class CosmosStamp:
    image: np.ndarray  # surface brightness per arcsec^2, ready for lenstronomy's INTERPOL
    scale: float  # arcsec/pixel of `image`
    cosmos_index: int
    rotation_deg: float
    raw_flux: float  # sum(image) * scale^2 before any amp rescaling


def draw_cosmos_stamp(
    rng: np.random.Generator | None,
    sample: str = "23.5",
    pixel_scale: float = 0.03,
    stamp_size: int = 150,
    regularizing_psf_fwhm: float = 0.10,
    cosmos_index: int | None = None,
    rotation_deg: float | None = None,
) -> CosmosStamp:
    """Draw one COSMOS galaxy, deconvolved from its native ACS/F814W PSF and
    reconvolved with a Gaussian (`regularizing_psf_fwhm`) -- standard practice
    for "real" `galsim.RealGalaxy` use (GalSim's own RealGalaxy demo), not a
    proxy for our actual instrument PSF, which lenstronomy applies afterwards
    at the survey's own resolution.

    `regularizing_psf_fwhm` MUST be >~ the native ACS/F814W PSF FWHM (~0.1",
    3 native 0.03" pixels): deconvolving the original PSF out and reconvolving
    with anything narrower over-sharpens beyond the data's real resolution and
    amplifies the correlated pixel noise baked into every COSMOS postage stamp
    without bound -- confirmed empirically here (an earlier 0.02" choice, finer
    than native resolution, produced pure noise/moiré with no galaxy structure
    at all; verified by rendering and looking at the actual stamps, not just
    checking the code ran -- see `data/sanity/cosmos_raw_stamps.png`).

    `cosmos_index`/`rotation_deg` are sampled from `rng` if not given, or
    replayed exactly if given (so `simulate.LensRenderer` can reconstruct the
    identical stamp from the JSON-serialisable truth dict without re-running
    the rng, and without storing the pixel array itself in `truth.jsonl`).

    Returns an intrinsic-frame stamp at `pixel_scale` arcsec/pixel, ready to
    pass as the `image`/`scale` kwargs of lenstronomy's INTERPOL light
    profile.
    """
    import galsim

    cat = _catalog(sample)
    idx = int(cosmos_index) if cosmos_index is not None else int(rng.integers(0, cat.nobjects))
    gal = cat.makeGalaxy(idx, gal_type="real", noise_pad_size=stamp_size * pixel_scale * 1.5)

    rot_deg = float(rotation_deg) if rotation_deg is not None else float(rng.uniform(0.0, 360.0))
    gal = gal.rotate(rot_deg * galsim.degrees)

    reg_psf = galsim.Gaussian(fwhm=regularizing_psf_fwhm)
    final = galsim.Convolve([gal, reg_psf])

    img = galsim.ImageF(stamp_size, stamp_size, scale=pixel_scale)
    final.drawImage(image=img, method="no_pixel")
    arr = img.array.astype(np.float64)
    arr[arr < 0] = 0.0  # a handful of small negative pixels survive deconvolution noise; physical surface brightness is >=0

    surface_brightness = arr / pixel_scale**2  # INTERPOL wants flux per arcsec^2, drawImage gave flux per pixel
    raw_flux = float(arr.sum())

    return CosmosStamp(image=surface_brightness, scale=pixel_scale, cosmos_index=idx, rotation_deg=rot_deg, raw_flux=raw_flux)
