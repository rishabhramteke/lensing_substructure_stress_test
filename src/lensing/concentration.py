"""Subhalo concentration models.

The methods survey (`../fulltext_findings.md` §1) shows every published ML
detector trains at a *fixed* concentration — c=60 (Tsang+2024, Campbell+2026,
justified as "consistent with recent observations") or c=15 (Ostdiek+2020/22,
Anau Montel+2022) — while Tsang's own c=15 ablation scores "around 0.1 one
would expect from random guessing." Only Hughes+2024 trains on a c–M relation
with scatter. RQ4 in `../problem_statement.md` measures exactly this gap, so
concentration is a first-class, switchable axis here rather than a constant.
"""
from __future__ import annotations

import numpy as np


def concentration_fixed(log10_m200: np.ndarray, value: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """The literature's usual choice: every subhalo gets the same concentration."""
    return np.full_like(log10_m200, float(value), dtype=float)


def concentration_dutton_maccio14(
    log10_m200: np.ndarray,
    rng: np.random.Generator,
    z: float = 0.5,
    scatter_dex: float = 0.11,
) -> np.ndarray:
    """Dutton & Maccio (2014, MNRAS 441, 3359), Planck-cosmology NFW relation, eq. 7:

        log10 c200 = a + b * log10(M200 / (1e12 h^-1 Msun))
        a = 0.520 + (0.905 - 0.520) * exp(-0.617 * z^1.21)
        b = -0.101 + 0.026 * z

    with a lognormal scatter of 0.11 dex (their quoted intrinsic scatter),
    applied per subhalo. This is the "ΛCDM c–M relation with scatter" branch
    of RQ4 — the realistic alternative to the fixed c=60/c=15 operating points
    every ML detector has been evaluated at.
    """
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    log10_c = a + b * (log10_m200 - 12.0)
    if scatter_dex > 0:
        log10_c = log10_c + rng.normal(0.0, scatter_dex, size=np.shape(log10_m200))
    return 10.0**log10_c


MODES = {
    "fixed60": lambda logm, rng: concentration_fixed(logm, 60.0),
    # c=30: an intermediate, tidally-plausible value. Dutton & Maccio (2014) describe
    # *isolated* field halos; surviving subhalos are tidally stripped and are denser at
    # fixed M200 than field halos of the same mass -- by factors of ~2-3 near the host
    # centre (Moline et al. 2017, MNRAS 466, 4974). c=30 is therefore the boosted
    # counterpart of the c~10-15 the field relation gives over this mass range, and the
    # honest middle point between the literature's c=60 and its c=15 ablation.
    # (Line-of-sight halos, which are field halos, keep the unboosted relation.)
    "fixed30": lambda logm, rng: concentration_fixed(logm, 30.0),
    "fixed15": lambda logm, rng: concentration_fixed(logm, 15.0),
    "cdm": concentration_dutton_maccio14,
}


def sample_concentration(log10_m200: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"unknown concentration mode {mode!r}; choose from {list(MODES)}")
    return MODES[mode](log10_m200, rng)
