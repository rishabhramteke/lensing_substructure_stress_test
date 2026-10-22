"""Sample a lens system and render it, with paired ablations for confounder analysis.

Every image in this module traces back to a fully-specified `truth` dict —
the hidden-truth record the stress test depends on (`../problem_statement.md`
§4.1: "hidden truth per image; every knob a config flag").
"""
from __future__ import annotations

import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Util import param_util

from .concentration import sample_concentration
from .config import SimConfig
from .cosmos_source import draw_cosmos_stamp
from .cosmos_source import n_objects as cosmos_n_objects

_SURVEYS = {}


def _survey_kwargs(cfg) -> dict:
    """Look up the instrument preset lenstronomy ships (HST/Euclid/JWST/LSST/Roman)."""
    if cfg.survey not in _SURVEYS:
        mod = __import__(f"lenstronomy.SimulationAPI.ObservationConfig.{cfg.survey}", fromlist=[cfg.survey])
        cls = getattr(mod, cfg.survey)
        _SURVEYS[cfg.survey] = cls
    cls = _SURVEYS[cfg.survey]
    try:
        obs = cls(band=cfg.band, psf_type=cfg.psf_type)
    except TypeError:
        obs = cls(psf_type=cfg.psf_type)
    kw = obs.kwargs_single_band()
    if cfg.pixel_scale is not None:
        kw["pixel_scale"] = cfg.pixel_scale
    if getattr(cfg, "exposure_time", None) is not None:
        kw["exposure_time"] = float(cfg.exposure_time)
    if getattr(cfg, "psf_fwhm", None) is not None:
        kw["seeing"] = float(cfg.psf_fwhm)
    return kw


def _mean_mass_powerlaw(lo, hi, slope):
    """<M> for dN/dM ~ M^slope on [10^lo, 10^hi]."""
    a, b = 10.0 ** lo, 10.0 ** hi
    k = slope + 1.0
    num = (b ** (k + 1) - a ** (k + 1)) / (k + 1); den = (b ** k - a ** k) / k
    return num / den


def _draw_powerlaw_log10(rng, n, lo, hi, slope):
    a, b = 10.0 ** lo, 10.0 ** hi; k = slope + 1.0
    u = rng.uniform(size=n)
    return np.log10((a ** k + u * (b ** k - a ** k)) ** (1.0 / k))


def sample_population(cfg: SimConfig, truth: dict):
    """The sub-threshold subhalo and line-of-sight population of `PopulationConfig`, or None. Its generator is seeded
    from the lens's own parameters, so drawing it consumes nothing from the population's main stream."""
    P = getattr(cfg, "population", None)
    if P is None or not P.enabled:
        return None
    lm, s = truth["lens_macro"], truth["source"]
    key = int(abs(lm["theta_E"] * 1e9) + abs(lm["gamma"] * 1e7) + abs(s["x"] * 1e8)) % (2 ** 32 - 1)
    prng = np.random.default_rng([key, 26])
    zl, zs = cfg.cosmology.z_lens, cfg.cosmology.z_source
    lc = LensCosmo(z_lens=zl, z_source=zs)
    R = P.disc_frac_thetaE * lm["theta_E"]
    m_sub_total = P.f_sub * 0.5 * lc.sigma_crit_angle * np.pi * R ** 2
    mbar = _mean_mass_powerlaw(*P.log10_mass_range, P.slope)
    out = []
    for kind, mtot in (("subhalo", m_sub_total), ("los", P.los_to_sub_mass_ratio * m_sub_total)):
        n = prng.poisson(mtot / mbar)
        logm = _draw_powerlaw_log10(prng, n, *P.log10_mass_range, P.slope)
        r = R * np.sqrt(prng.uniform(size=n)); a = prng.uniform(0, 2 * np.pi, size=n)
        for k in range(n):
            if kind == "subhalo":
                z, mode = zl, cfg.subhalo.concentration_mode if cfg.subhalo.concentration_mode != "cdm" else "cdm"
                c = float(sample_concentration(np.array(logm[k]), "cdm", prng))
            else:
                z = float(prng.uniform(*P.los_z_frac_range) * zs)
                c = float(sample_concentration(np.array(logm[k]), "cdm", prng))
            out.append({"kind": kind, "log10_M200": float(logm[k]), "concentration": c, "tau": P.tau, "z": z,
                        "x": float(r[k] * np.cos(a[k])), "y": float(r[k] * np.sin(a[k]))})
    return out


def sample_truth(cfg: SimConfig, rng: np.random.Generator) -> dict:
    """Draw one lens system's parameters. Returns a JSON-serialisable dict —
    the hidden truth — with every quantity a metric or a figure will need."""
    L, S, H, M = cfg.lens, cfg.source, cfg.subhalo, cfg.multipole
    LOS = getattr(cfg, "los_halo", None)

    theta_E = rng.uniform(*L.theta_E_range)
    gamma = rng.uniform(*L.gamma_range)
    q_l = rng.uniform(*L.q_range)
    phi_l = np.radians(rng.uniform(*L.phi_range_deg))
    e1, e2 = param_util.phi_q2_ellipticity(phi_l, q_l)
    shear_mag = rng.uniform(*L.shear_mag_range)
    shear_phi = rng.uniform(0, 2 * np.pi)
    g1, g2 = shear_mag * np.cos(2 * shear_phi), shear_mag * np.sin(2 * shear_phi)

    src_r = rng.uniform(*S.offset_frac_thetaE_range) * theta_E
    src_ang = rng.uniform(0, 2 * np.pi)
    src_x, src_y = src_r * np.cos(src_ang), src_r * np.sin(src_ang)

    if S.source_type == "cosmos":
        cosmos_index = int(rng.integers(0, cosmos_n_objects(S.cosmos_sample)))
        rotation_deg = float(rng.uniform(0.0, 360.0))
        amp_scale = rng.uniform(*S.cosmos_amp_range)
        source = {
            "type": "cosmos", "x": src_x, "y": src_y,
            "cosmos_sample": S.cosmos_sample, "cosmos_index": cosmos_index, "rotation_deg": rotation_deg,
            "pixel_scale": S.cosmos_pixel_scale, "stamp_size": S.cosmos_stamp_size, "amp_scale": amp_scale,
        }
    else:
        R_sersic = rng.uniform(*S.R_sersic_range)
        n_sersic = rng.uniform(*S.n_sersic_range)
        q_s = rng.uniform(*S.q_range)
        phi_s = np.radians(rng.uniform(*S.phi_range_deg))
        e1_s, e2_s = param_util.phi_q2_ellipticity(phi_s, q_s)
        amp_s = rng.uniform(*S.amp_range)
        source = {
            "type": "sersic", "x": src_x, "y": src_y, "R_sersic": R_sersic, "n_sersic": n_sersic,
            "q": q_s, "phi_deg": np.degrees(phi_s), "e1": e1_s, "e2": e2_s, "amp": amp_s,
        }
        if getattr(S, "n_clumps", 0):
            # flux of a Sersic ~ amp * R^2 (up to an n-dependent factor, the same for the n=1 clumps and absorbed
            # into the fraction's range): clump amplitude = frac * amp_s * (R_sersic / R_clump)^2
            clumps = []
            for _ in range(S.n_clumps):
                rc = rng.uniform(*S.clump_R_range); frac = rng.uniform(*S.clump_flux_frac_range)
                d = rng.uniform(0, 1) * R_sersic; a = rng.uniform(0, 2 * np.pi)
                clumps.append({"x": src_x + d * np.cos(a), "y": src_y + d * np.sin(a), "R_sersic": rc, "n_sersic": 1.0,
                               "flux_frac": frac, "amp": frac * amp_s * (R_sersic / rc) ** 2})
            source["clumps"] = clumps

    truth = {
        "tier": cfg.tier,
        "cosmology": {"z_lens": cfg.cosmology.z_lens, "z_source": cfg.cosmology.z_source},
        "lens_macro": {
            "theta_E": theta_E, "gamma": gamma, "q": q_l, "phi_deg": np.degrees(phi_l),
            "e1": e1, "e2": e2, "gamma1": g1, "gamma2": g2, "shear_mag": shear_mag,
        },
        "source": source,
        "subhalo": None,
        "multipole": None,
    }

    if H.enabled and rng.uniform() < H.presence_prob:
        log10_m = rng.uniform(*H.log10_mass_range)
        c = float(sample_concentration(np.array(log10_m), H.concentration_mode, rng))
        r = rng.uniform(*H.placement_annulus_frac_thetaE) * theta_E
        ang = rng.uniform(0, 2 * np.pi)
        sub_x, sub_y = r * np.cos(ang), r * np.sin(ang)
        truth["subhalo"] = {
            "log10_M200": log10_m, "concentration": c, "concentration_mode": H.concentration_mode,
            "tau": H.tau, "x": sub_x, "y": sub_y, "r_from_center": r,
        }

    if M.enabled:
        am_rel = abs(rng.normal(0.0, M.am_sigma)) if getattr(M, "am_sigma", 0.0) > 0 else M.am_over_thetaE
        a_m = am_rel * theta_E
        phi_m = np.radians(rng.uniform(*M.phi_m_range_deg))
        truth["multipole"] = {"m": M.m, "a_m": a_m, "am_over_thetaE": am_rel, "phi_m_deg": np.degrees(phi_m)}
        if getattr(M, "elliptical", False):
            truth["multipole"]["elliptical"] = True; truth["multipole"]["q"] = q_l

    if LOS is not None and LOS.enabled and rng.uniform() < LOS.presence_prob:
        log10_m = rng.uniform(*LOS.log10_mass_range)
        c = float(sample_concentration(np.array(log10_m), LOS.concentration_mode, rng))
        r = rng.uniform(*LOS.placement_annulus_frac_thetaE) * theta_E
        ang = rng.uniform(0, 2 * np.pi)
        truth["los_halo"] = {"log10_M200": log10_m, "concentration": c, "concentration_mode": LOS.concentration_mode,
                              "tau": LOS.tau, "z_halo": LOS.z_halo, "x": r * np.cos(ang), "y": r * np.sin(ang), "r_from_center": r}
    else:
        truth["los_halo"] = None

    truth["population"] = sample_population(cfg, truth)

    truth["lens_light"] = None
    LL = getattr(cfg, "lens_light", None)
    if LL is not None and LL.enabled and S.source_type != "cosmos":
        q_light = float(np.clip(q_l + rng.uniform(*LL.q_offset_range), 0.4, 1.0))
        phi_light = phi_l + np.radians(rng.uniform(*LL.phi_offset_deg_range))
        e1_b, e2_b = param_util.phi_q2_ellipticity(phi_light, q_light)
        q_env = float(np.clip(q_light + rng.uniform(-0.1, 0.1), 0.4, 1.0))
        e1_e, e2_e = param_util.phi_q2_ellipticity(phi_light + np.radians(rng.uniform(-10, 10)), q_env)
        amp_b = truth["source"]["amp"] * rng.uniform(*LL.bulge_amp_ratio_range)
        truth["lens_light"] = [
            {"amp": float(amp_b), "R_sersic": float(rng.uniform(*LL.bulge_R_range)), "n_sersic": float(rng.uniform(*LL.bulge_n_range)),
             "e1": float(e1_b), "e2": float(e2_b), "center_x": 0.0, "center_y": 0.0},
            {"amp": float(amp_b * rng.uniform(*LL.envelope_flux_frac_range)), "R_sersic": float(rng.uniform(*LL.envelope_R_range)), "n_sersic": float(LL.envelope_n),
             "e1": float(e1_e), "e2": float(e2_e), "center_x": 0.0, "center_y": 0.0},
        ]
    return truth


class LensRenderer:
    """Builds a lenstronomy SimAPI once per instrument config and renders any
    truth dict against it, with variants for ablation (see `render`)."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.kwargs_band = _survey_kwargs(cfg.instrument)          # nominal: what the manifest records and the fitters use
        self.kwargs_band_render = dict(self.kwargs_band)
        self.kwargs_band_render["seeing"] = self.kwargs_band["seeing"] * getattr(cfg.instrument, "render_psf_scale", 1.0)
        self.render_supersampling = int(getattr(cfg.instrument, "render_supersampling", 1))
        self.num_pix = cfg.instrument.num_pix

    def _sim_api(self, model_list: list[str], n_lens_light: int = 0, z_list: list[float] | None = None, n_clumps: int = 0) -> SimAPI:
        source_model = ["INTERPOL"] if self.cfg.source.source_type == "cosmos" else ["SERSIC_ELLIPSE"] + ["SERSIC"] * n_clumps
        kwargs_model = {"lens_model_list": model_list, "source_light_model_list": source_model}
        if z_list is not None:
            # a redshift list switches lenstronomy to multi-plane ray tracing (LOS halo)
            kwargs_model["lens_redshift_list"] = z_list
            kwargs_model["z_source"] = self.cfg.cosmology.z_source
        if n_lens_light:
            kwargs_model["lens_light_model_list"] = ["SERSIC_ELLIPSE"] * n_lens_light
        return SimAPI(num_pix=self.num_pix, kwargs_single_band=self.kwargs_band_render, kwargs_model=kwargs_model)

    @staticmethod
    def _kwargs_lens(truth: dict, include_subhalo: bool, include_multipole: bool, include_los: bool = True) -> list[dict]:
        lm = truth["lens_macro"]
        kw = [
            {"theta_E": lm["theta_E"], "gamma": lm["gamma"], "e1": lm["e1"], "e2": lm["e2"], "center_x": 0.0, "center_y": 0.0},
            {"gamma1": lm["gamma1"], "gamma2": lm["gamma2"], "ra_0": 0.0, "dec_0": 0.0},
        ]
        if include_multipole and truth.get("multipole"):
            mp = truth["multipole"]
            kmp = {"m": mp["m"], "a_m": mp["a_m"], "phi_m": np.radians(mp["phi_m_deg"]), "center_x": 0.0, "center_y": 0.0, "r_E": lm["theta_E"]}
            if mp.get("elliptical"):
                kmp["q"] = mp["q"]
            kw.append(kmp)
        if include_subhalo and truth.get("subhalo"):
            sh = truth["subhalo"]
            lc = LensCosmo(z_lens=truth["cosmology"]["z_lens"], z_source=truth["cosmology"]["z_source"])
            Rs_angle, alpha_Rs = lc.nfw_physical2angle(M=10 ** sh["log10_M200"], c=sh["concentration"])
            kw.append({"Rs": Rs_angle, "alpha_Rs": alpha_Rs, "r_trunc": sh["tau"] * Rs_angle, "center_x": sh["x"], "center_y": sh["y"]})
        if include_los and truth.get("los_halo"):
            lh = truth["los_halo"]
            lc = LensCosmo(z_lens=lh["z_halo"], z_source=truth["cosmology"]["z_source"])
            Rs_angle, alpha_Rs = lc.nfw_physical2angle(M=10 ** lh["log10_M200"], c=lh["concentration"])
            kw.append({"Rs": Rs_angle, "alpha_Rs": alpha_Rs, "r_trunc": lh["tau"] * Rs_angle, "center_x": lh["x"], "center_y": lh["y"]})
        for h in truth.get("population") or []:
            lc = LensCosmo(z_lens=h["z"], z_source=truth["cosmology"]["z_source"])
            Rs_angle, alpha_Rs = lc.nfw_physical2angle(M=10 ** h["log10_M200"], c=h["concentration"])
            kw.append({"Rs": Rs_angle, "alpha_Rs": alpha_Rs, "r_trunc": h["tau"] * Rs_angle, "center_x": h["x"], "center_y": h["y"]})
        return kw

    @staticmethod
    def _model_list(include_subhalo: bool, include_multipole: bool, include_los: bool = False, elliptical_multipole: bool = False, n_population: int = 0) -> list[str]:
        m = ["EPL", "SHEAR"]
        if include_multipole:
            m.append("MULTIPOLE_ELL" if elliptical_multipole else "MULTIPOLE")
        if include_subhalo:
            m.append("TNFW")
        if include_los:
            m.append("TNFW")
        m += ["TNFW"] * n_population
        return m

    def _z_list(self, truth: dict, model_list: list[str], include_los: bool) -> list[float] | None:
        """Per-profile redshifts, or None when every profile is at the lens plane (single-plane)."""
        pop = truth.get("population") or []
        if not (include_los and truth.get("los_halo")) and not pop:
            return None
        zl = truth["cosmology"]["z_lens"]
        n_main = len(model_list) - len(pop)
        z = [zl] * n_main
        if include_los and truth.get("los_halo"):
            z[-1] = truth["los_halo"]["z_halo"]
        return z + [h["z"] for h in pop]

    def _kwargs_source(self, truth: dict) -> list[dict]:
        s = truth["source"]
        if s.get("type") == "cosmos":
            stamp = draw_cosmos_stamp(
                rng=None, sample=s["cosmos_sample"], pixel_scale=s["pixel_scale"], stamp_size=s["stamp_size"],
                cosmos_index=s["cosmos_index"], rotation_deg=s["rotation_deg"],
            )
            return [{"image": stamp.image, "amp": s["amp_scale"], "center_x": s["x"], "center_y": s["y"], "phi_G": 0.0, "scale": stamp.scale}]
        return [{"amp": s["amp"], "R_sersic": s["R_sersic"], "n_sersic": s["n_sersic"], "e1": s["e1"], "e2": s["e2"], "center_x": s["x"], "center_y": s["y"]}] + \
               [{"amp": c["amp"], "R_sersic": c["R_sersic"], "n_sersic": c["n_sersic"], "center_x": c["x"], "center_y": c["y"]} for c in s.get("clumps", [])]

    def render(self, truth: dict, noiseless_only: bool = False, seed: int | None = None) -> dict:
        """Render the full image plus ablated controls, sharing one noise draw.

        Returns noiseless "full", "no_subhalo", "no_multipole" and "smooth"
        (macro-only) images, plus the noisy "full" image used for training/
        detection and its matching noisy ablations — so `full - no_subhalo`
        (same noise realization) is exactly the perturbation's signature, the
        quantity the confounder-FPR experiment (RQ2) measures.
        """
        has_sub = truth.get("subhalo") is not None
        has_mp = truth.get("multipole") is not None
        kwargs_source = self._kwargs_source(truth)
        kwargs_lens_light = truth.get("lens_light") or None   # list of Sersic dicts, or None (added 2026-09-11)

        has_los = truth.get("los_halo") is not None
        # (include_subhalo, include_multipole, include_los)
        variants = {"smooth": (False, False, False), "no_subhalo": (False, has_mp, has_los),
                    "no_multipole": (has_sub, False, has_los), "full": (has_sub, has_mp, has_los)}
        if has_los:
            variants["no_los"] = (has_sub, has_mp, False)
        noiseless, sim_full = {}, None
        for name, (inc_sub, inc_mp, inc_los) in variants.items():
            model_list = self._model_list(inc_sub, inc_mp, inc_los, elliptical_multipole=bool((truth.get("multipole") or {}).get("elliptical")),
                                          n_population=len(truth.get("population") or []))
            sim = self._sim_api(model_list, n_lens_light=len(kwargs_lens_light) if kwargs_lens_light else 0,
                                 z_list=self._z_list(truth, model_list, inc_los), n_clumps=len(truth["source"].get("clumps", [])))
            im = sim.image_model_class(kwargs_numerics={"supersampling_factor": self.render_supersampling})
            kwargs_lens = self._kwargs_lens(truth, inc_sub, inc_mp, inc_los)
            noiseless[name] = im.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light)
            if name == "full":
                sim_full = sim

        out = {"noiseless": noiseless}
        if not noiseless_only:
            # the per-image seed drawn from the population seed now also seeds the noise (populations generated before
            # 2026-09-29 drew their noise from numpy's unseeded global state: their truths and noiseless images are
            # regenerable from the manifest, their noisy images only as the released files)
            noise = sim_full.noise_for_model(model=noiseless["full"], seed=None if seed is None else int(seed) % (2**32 - 1))
            # reseed identically is not needed: `noise` is one array, reused for every variant
            out["noise_std"] = float(noise.std())
            out["noisy"] = {name: img + noise for name, img in noiseless.items()}
        return out
