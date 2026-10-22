"""Simulation config: every prior range cited against the paper it reproduces.

Where a paper's full text (`../fulltext_findings.md`) does not state a value
(most papers omit their external-shear prior, for instance), the default here
is a documented choice of ours, not a silent invention — see the `note=`
field on each config. This is the "every config ... released" commitment in
`../problem_statement.md` §5.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CosmologyConfig:
    z_lens: float = 0.5
    z_source: float = 1.0
    # Fixed redshifts, matching Ostdiek+2020/22 and Tsang+2024's operating
    # point. Their own limitation, verbatim (Ostdiek): "both our lenses and
    # sources were at fixed redshift. We have not characterized how the
    # network accuracy would change if these assumptions were removed." A
    # redshift-diversity axis is a documented backlog item, not yet built.


@dataclass
class LensConfig:
    """EPL + external shear. Ranges match Tsang+2024's "power-law elliptical
    potential ... Einstein radius [0.8-1.2] arcsec; slope gamma [1.5-2.5];
    axis ratio q [0.5-1]"."""

    theta_E_range: tuple[float, float] = (0.8, 1.2)
    gamma_range: tuple[float, float] = (1.5, 2.5)
    q_range: tuple[float, float] = (0.5, 1.0)
    phi_range_deg: tuple[float, float] = (0.0, 180.0)
    # External shear magnitude: NOT specified in Tsang+2024's text extract.
    # This range (a uniform magnitude up to 0.05, random angle) is our own
    # choice, picked to be a typical galaxy-galaxy-lensing external-shear
    # scale, not a reproduction of any single paper's prior.
    shear_mag_range: tuple[float, float] = (0.0, 0.05)
    note: str = "theta_E/gamma/q from Tsang+2024 (arXiv:2401.16624); shear range is our own default, unspecified in the source paper"


@dataclass
class SourceConfig:
    """Source light. `source_type="sersic"` is the Tier-0 operating point
    shared by Ostdiek+2020/22, Hughes+2024 and Campbell+2026 (none of these
    three used COSMOS cutouts). `source_type="cosmos"` is the Tier-1 upgrade:
    real HST/ACS COSMOS postage stamps via `galsim.COSMOSCatalog` (see
    `cosmos_source.py`), matching Tsang+2024's own baseline, which already
    used COSMOS sources."""

    source_type: str = "sersic"  # "sersic" | "cosmos"

    R_sersic_range: tuple[float, float] = (0.05, 0.3)
    n_sersic_range: tuple[float, float] = (0.7, 4.0)
    q_range: tuple[float, float] = (0.5, 1.0)
    phi_range_deg: tuple[float, float] = (0.0, 180.0)
    # Placed near, not on, the lens centre so images are extended arcs rather
    # than a point at the origin; radius as a fraction of theta_E.
    offset_frac_thetaE_range: tuple[float, float] = (0.05, 0.35)
    amp_range: tuple[float, float] = (60.0, 220.0)
    # source-misspecification arm (referee round 22): `n_clumps` compact round Sersic clumps (n=1) added to the Sersic
    # source, each carrying a fraction of the main component's amplitude-weighted flux and placed within one effective
    # radius of its centre. Every fitter models a single Sersic, so the source model is misspecified by a known amount.
    n_clumps: int = 0
    clump_flux_frac_range: tuple[float, float] = (0.02, 0.06)
    clump_R_range: tuple[float, float] = (0.03, 0.08)

    # COSMOS (Tier 1) knobs. `cosmos_sample="23.5"` is the F814W<23.5 training
    # sample (~56,000 galaxies, Zenodo 3242143) -- the same one `paltas` wraps.
    # `cosmos_amp_range` plays the same role as `amp_range` above (an overall
    # flux rescaling so COSMOS stamps land in the same surface-brightness
    # regime as the Sersic source, since a raw HST-calibrated COSMOS flux and
    # our synthetic instrument's ADU scale have no reason to match).
    cosmos_sample: str = "23.5"
    cosmos_pixel_scale: float = 0.03
    cosmos_stamp_size: int = 150
    cosmos_amp_range: tuple[float, float] = (0.5, 3.0)

    note: str = "Sersic ranges are the Tier-0 default; source_type='cosmos' is the Tier-1 upgrade (real COSMOS postage stamps, galsim.COSMOSCatalog 23.5 sample)"


@dataclass
class SubhaloConfig:
    """Truncated-NFW (TNFW) perturber. Mass range and truncation ratio from
    Tsang+2024: "mass range 10^8-10^11 M_sun ... Fixed truncation/scale
    radius ratio tau=20." Concentration mode is the RQ4 knob — see
    `concentration.py`."""

    enabled: bool = True
    log10_mass_range: tuple[float, float] = (8.0, 11.0)
    concentration_mode: str = "fixed60"  # "fixed60" | "fixed15" | "cdm"
    tau: float = 20.0  # r_trunc / r_s
    # Placed in an annulus around the Einstein radius, where the lensed arc
    # actually is and where a perturbation is detectable at all (Despali+2022
    # sensitivity maps). Radii are fractions of theta_E.
    placement_annulus_frac_thetaE: tuple[float, float] = (0.6, 1.3)
    presence_prob: float = 0.9  # Tsang+2024: "90% of images contain one subhalo"
    note: str = "mass range and tau from Tsang+2024; placement annulus is our own approximation of their pixel-brightness criterion"


@dataclass
class MultipoleConfig:
    """The lens-shape confounder (RQ2). Off by default (am=0); the Tier-2
    sweep sets am/theta_E to {0.01, 0.03}, the two values O'Riordan+2025 use
    to show the population signal is "consistent with zero" at 3%."""

    enabled: bool = False
    m: int = 4  # 3 or 4, per Tsang/Lange/O'Riordan convention
    am_over_thetaE: float = 0.0
    phi_m_range_deg: tuple[float, float] = (0.0, 180.0)
    # >0: draw a_m/theta_E per lens as |N(0, am_sigma)| instead of the fixed am_over_thetaE (a population of amplitudes,
    # referee round 23); the ordinary populations keep am_sigma = 0
    am_sigma: float = 0.0
    # elliptical multipole (Paugnat & Gilman 2025; lenstronomy MULTIPOLE_ELL) defined on ellipses of the lens's own axis
    # ratio: the out-of-family confounder for a macro-model that carries a circular multipole (referee round 22)
    elliptical: bool = False
    note: str = "am/thetaE=0.01 and 0.03 are O'Riordan+2025's two test amplitudes (arXiv:2509.02660)"



@dataclass
class LensLightConfig:
    """Lens-galaxy light (added 2026-09-11, referee point: the suite had none).

    The TRUE lens light is deliberately more complex than what a pipeline fits: a
    two-component Sersic (a de Vaucouleurs-like bulge plus an exponential envelope,
    slightly misaligned and with different flattening), centred on the lens. Family A
    fits a SINGLE Sersic jointly with the mass model, so the subtraction residual is
    a smooth, centrally concentrated pattern -- the situation Nightingale et al.
    (2024) identify as the largest confounder on real data. Amplitudes are set
    relative to the source's Sersic amplitude so the lens is comparable to or
    brighter than the arcs."""

    enabled: bool = False
    bulge_amp_ratio_range: tuple[float, float] = (0.5, 2.0)   # bulge surface brightness at R_bulge / source amp
    bulge_R_range: tuple[float, float] = (0.25, 0.5)          # arcsec
    bulge_n_range: tuple[float, float] = (3.0, 5.0)
    envelope_flux_frac_range: tuple[float, float] = (0.3, 0.7)  # envelope amp = frac * bulge amp
    envelope_R_range: tuple[float, float] = (0.7, 1.3)        # arcsec
    envelope_n: float = 1.0
    q_offset_range: tuple[float, float] = (-0.15, 0.15)       # light q = mass q + offset (clipped to [0.4, 1])
    phi_offset_deg_range: tuple[float, float] = (-15.0, 15.0)  # light PA = mass PA + offset
    note: str = "our own choice; not a reproduction of any paper's prior"

@dataclass
class InstrumentConfig:
    """lenstronomy's own HST WFC3_F160W preset with a Gaussian PSF —
    Tsang+2024's "HST WFC3 F160W ... Gaussian PSF" at their coarsest (80 mas)
    pixel scale, one of three they test (80/20/10 mas)."""

    survey: str = "HST"
    band: str = "WFC3_F160W"
    psf_type: str = "GAUSSIAN"
    pixel_scale: float | None = 0.08  # None -> use the survey preset's own value
    num_pix: int = 64
    # None -> the preset's own exposure (5400 s). Set it to render the SAME lenses at a
    # different depth: the referee point that every result here is at arc S/N ~1.6e3,
    # while a survey visit delivers far less (2026-09-13).
    exposure_time: float | None = None
    # None -> the preset's own Gaussian FWHM (0.08", one pixel). Set it to render the SAME lenses
    # through a wider PSF: the referee point that the preset is about half the real F160W width
    # and one-pixel sampling favours every detector (2026-09-15).
    psf_fwhm: float | None = None
    note: str = "HST/F160W/Gaussian/0.08\" is Tsang+2024's coarsest grid; 0.02\" reproduces their finest"
    # inverse-crime control (referee round 23): render with a finer numerical supersampling and a slightly wider PSF than
    # the fitters assume. The manifest keeps the NOMINAL kwargs_band the fitters read, so the forward model the detectors
    # use differs from the one that made the data. 1 / 1.0 reproduce the ordinary renders exactly.
    render_supersampling: int = 1
    render_psf_scale: float = 1.0


@dataclass
class LOSHaloConfig:
    """A line-of-sight halo on its own lens plane (referee round 7, point M7).

    Sengul et al. (2022) reanalysed the field's first "dark perturber" (in JVAS B1938+666)
    as a line-of-sight halo rather than a subhalo, and Gilman et al. (2020) show the
    line-of-sight population dominates the perturbation signal in some configurations.
    A halo at z != z_lens is a *field* halo, so it keeps the unboosted concentration-mass
    relation (see concentration.py); a detector that assumes the perturber sits at the
    lens redshift will still flag it, but will infer the wrong mass.
    """

    enabled: bool = False
    z_halo: float = 0.25                       # foreground by default; 0.75 is the background case
    log10_mass_range: tuple = (8.0, 11.0)
    concentration_mode: str = "cdm"            # field halo: the unboosted Dutton & Maccio relation
    tau: float = 20.0
    placement_annulus_frac_thetaE: tuple = (0.6, 1.3)
    presence_prob: float = 1.0
    note: str = "line-of-sight halo on a second lens plane; Sengul+2022, Gilman+2020"


@dataclass
class PopulationConfig:
    """A population of low-mass subhalos and line-of-sight halos below the suite's detection range, present in every
    image of a population (including its subhalo-free control), so that a "clean" lens carries the sub-threshold
    structure a real lens does (referee round 26, M7; Coogan et al. 2024, Despali et al. 2018, Amorisco et al. 2022).
    Subhalos: dN/dM ~ M^-1.9 on [10^7, 10^8.5] Msun at the lens redshift, Dutton & Maccio concentrations with the
    tidal boost of the main subhalo population, tau = 20, uniform within 1.5 theta_E, Poisson-distributed with a
    projected mass of `f_sub` times a convergence of 0.5 over that disc (0.4%: the share of a ~1% substructure
    fraction in this mass range). Line-of-sight halos: `los_to_sub_mass_ratio` times the subhalo mass, on their own
    planes with z uniform in [0.05, 0.95] z_s, field concentrations. Drawn from a generator seeded by the lens's own
    parameters, so the lenses, sources and main subhalos remain those of the population without it."""
    enabled: bool = False
    log10_mass_range: tuple = (7.0, 8.5)
    slope: float = -1.9
    f_sub: float = 0.004
    disc_frac_thetaE: float = 1.5
    los_to_sub_mass_ratio: float = 2.0
    los_z_frac_range: tuple = (0.05, 0.95)
    tau: float = 20.0
    note: str = "sub-threshold subhalo + line-of-sight population; referee round 26"


@dataclass
class SimConfig:
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    subhalo: SubhaloConfig = field(default_factory=SubhaloConfig)
    multipole: MultipoleConfig = field(default_factory=MultipoleConfig)
    los_halo: LOSHaloConfig = field(default_factory=LOSHaloConfig)
    lens_light: LensLightConfig = field(default_factory=LensLightConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    tier: str = "tier0"


def tier0_tsang(concentration_mode: str = "fixed60") -> SimConfig:
    """Tier 0, the literature's own operating point (Tsang+2024 macro/subhalo
    priors on a single-Sersic source). Pass concentration_mode="fixed15" to
    reproduce their low-concentration ablation ("around 0.1 one would expect
    from random guessing")."""
    cfg = SimConfig()
    cfg.subhalo.concentration_mode = concentration_mode
    cfg.tier = f"tier0_tsang_{concentration_mode}"
    return cfg


def tier0_shallow(concentration_mode: str = "fixed60", exposure_time: float = 135.0, no_subhalo: bool = False, am_over_thetaE: float = 0.0, m: int = 4) -> SimConfig:
    """Tier 0 at survey-like depth: the identical priors and (given the same seed) the
    identical lenses, sources and subhalos, with only the exposure time reduced. 135 s
    against the preset's 5400 s takes the median arc signal-to-noise from ~1.3e3 to ~2e2
    and the peak pixel from ~240 sigma to ~37 sigma."""
    cfg = tier0_tsang(concentration_mode)
    cfg.instrument.exposure_time = exposure_time
    if no_subhalo:
        cfg.subhalo.enabled = False
        cfg.subhalo.presence_prob = 0.0
    if am_over_thetaE:
        cfg.multipole.enabled = True
        cfg.multipole.am_over_thetaE = am_over_thetaE
        cfg.multipole.m = m
    cfg.tier = f"tier0_shallow{int(exposure_time)}s_{'no_subhalo' if no_subhalo else concentration_mode}" + (f"_m{m}a{am_over_thetaE}" if am_over_thetaE else "")
    return cfg


def tier0_psf(fwhm: float = 0.15, concentration_mode: str = "fixed60", no_subhalo: bool = False, am_over_thetaE: float = 0.0, m: int = 4) -> SimConfig:
    """Tier 0 with the Gaussian PSF widened from the preset's 0.08" (one pixel) to `fwhm`, everything
    else identical, so that given the same seed the lenses, sources, subhalos and noise draws match
    the Tier-0 populations pixel for pixel before convolution. 0.15" is the real WFC3/F160W width."""
    cfg = tier0_tsang(concentration_mode)
    cfg.instrument.psf_fwhm = float(fwhm)
    if no_subhalo:
        cfg.subhalo.enabled = False
        cfg.subhalo.presence_prob = 0.0
    if am_over_thetaE:
        cfg.multipole.enabled = True
        cfg.multipole.am_over_thetaE = am_over_thetaE
        cfg.multipole.m = m
    cfg.tier = f"tier0_psf{fwhm:g}_{'no_subhalo' if no_subhalo else concentration_mode}" + (f"_m{m}a{am_over_thetaE}" if am_over_thetaE else "")
    return cfg


def tier0_realistic(concentration_mode: str = "fixed60", no_subhalo: bool = False, am_over_thetaE: float = 0.0, m: int = 4) -> SimConfig:
    """The three instrument and lens realism axes at once: the 0.15" WFC3/F160W PSF, a 600 s snapshot-length exposure and
    two-component lens light, everything else Tier 0, so that given the same seed the lenses, sources and subhalos are
    those of the Tier-0 populations."""
    cfg = tier0_psf(0.15, concentration_mode, no_subhalo=no_subhalo, am_over_thetaE=am_over_thetaE, m=m)
    cfg.instrument.exposure_time = 600.0
    cfg.lens_light.enabled = True
    cfg.tier = f"tier0_realistic_{'no_subhalo' if no_subhalo else concentration_mode}" + (f"_m{m}a{am_over_thetaE}" if am_over_thetaE else "")
    return cfg


def tier0_clumpy_source(concentration_mode: str = "fixed60", no_subhalo: bool = False, am_over_thetaE: float = 0.0, m: int = 4) -> SimConfig:
    """Tier 0 with two faint compact clumps added to the Sersic source (2-6% of its flux each): a source-model
    misspecification that keeps the smooth fit within the gate, so the arm has a proper null."""
    cfg = tier0_tsang(concentration_mode)
    cfg.source.n_clumps = 2
    if no_subhalo:
        cfg.subhalo.enabled = False
        cfg.subhalo.presence_prob = 0.0
    if am_over_thetaE:
        cfg.multipole.enabled = True
        cfg.multipole.am_over_thetaE = am_over_thetaE
        cfg.multipole.m = m
    cfg.tier = f"tier0_clumpy_{'no_subhalo' if no_subhalo else concentration_mode}" + (f"_m{m}a{am_over_thetaE}" if am_over_thetaE else "")
    return cfg


def tier2_elliptical_multipole(am_over_thetaE: float = 0.03, m: int = 4) -> SimConfig:
    """A subhalo-free population with an ELLIPTICAL multipole (Paugnat & Gilman 2025) on the lens's own axis ratio."""
    cfg = tier2_multipole_confounder(am_over_thetaE=am_over_thetaE, m=m)
    cfg.multipole.elliptical = True
    cfg.tier = f"tier2_ellmultipole_m{m}_a{am_over_thetaE}"
    return cfg


def tier0_inverse_crime(concentration_mode: str = "fixed60", no_subhalo: bool = False, am_over_thetaE: float = 0.0, m: int = 4) -> SimConfig:
    """Tier 0 rendered with 5x supersampling and a PSF 10% wider than the nominal 0.08" the fitters are told."""
    cfg = tier0_tsang(concentration_mode)
    cfg.instrument.render_supersampling = 5
    cfg.instrument.render_psf_scale = 1.1
    if no_subhalo:
        cfg.subhalo.enabled = False
        cfg.subhalo.presence_prob = 0.0
    if am_over_thetaE:
        cfg.multipole.enabled = True
        cfg.multipole.am_over_thetaE = am_over_thetaE
        cfg.multipole.m = m
    cfg.tier = f"tier0_invcrime_{'no_subhalo' if no_subhalo else concentration_mode}" + (f"_m{m}a{am_over_thetaE}" if am_over_thetaE else "")
    return cfg


def tier0_supersampled(concentration_mode: str = "fixed60", no_subhalo: bool = False, am_over_thetaE: float = 0.0, m: int = 4) -> SimConfig:
    """Tier 0 rendered with 5x supersampling only (the nominal PSF), fitted with the 1x model: the numerical half of the
    inverse-crime control."""
    cfg = tier0_inverse_crime(concentration_mode, no_subhalo=no_subhalo, am_over_thetaE=am_over_thetaE, m=m)
    cfg.instrument.render_psf_scale = 1.0
    cfg.tier = cfg.tier.replace("invcrime", "supersampled")
    return cfg


def tier0_multipole_subhalo(concentration_mode: str = "fixed60", no_subhalo: bool = False, am_sigma: float = 0.01, m: int = 4) -> SimConfig:
    """Subhalo AND an m=4 multipole in the same lens, the amplitude drawn per lens as |N(0, am_sigma)| a_m/theta_E; with
    the Tier-0 seed the lenses, sources and subhalos are those of the Tier-0 concentration populations."""
    cfg = tier0_tsang(concentration_mode)
    if no_subhalo:
        cfg.subhalo.enabled = False
        cfg.subhalo.presence_prob = 0.0
    cfg.multipole.enabled = True
    cfg.multipole.am_sigma = am_sigma
    cfg.multipole.am_over_thetaE = am_sigma
    cfg.multipole.m = m
    cfg.tier = f"tier0_mpsub_{'no_subhalo' if no_subhalo else concentration_mode}_m{m}sig{am_sigma}"
    return cfg


def tier0_los_halo(z_halo: float = 0.25) -> SimConfig:
    """Tier 0 with NO subhalo at the lens plane and one line-of-sight halo at `z_halo`,
    drawn from the same mass range and the same annulus in projection. The detectors are
    unchanged: they assume any perturber sits at the lens redshift."""
    cfg = tier0_tsang()
    cfg.subhalo.enabled = False
    cfg.subhalo.presence_prob = 0.0
    cfg.los_halo.enabled = True
    cfg.los_halo.z_halo = z_halo
    cfg.tier = f"tier0_los_halo_z{z_halo}"
    return cfg


def tier0_tidal_track(concentration: float = 15.0, c_ref: float = 60.0, tau_ref: float = 20.0) -> SimConfig:
    """Tier 0 with concentration AND truncation varied together along a tidal track.

    The rest of this paper holds tau = r_t/r_s fixed at 20 while varying c, which is the
    literature's convention but is unphysical in a specific direction (referee round 8):
    at fixed M200 a lower c means a larger r_s, so a fixed tau makes the *truncation radius*
    grow too, and the low-c perturber ends up both more diffuse and more extended. Tidal
    truncation is set by the host's tidal field, so r_t is roughly fixed in physical units
    and tau should scale with c. This factory sets tau = tau_ref * (c / c_ref), i.e. tau = 5
    at c = 15 against tau = 20 at c = 60, so the low-c subhalo is more severely stripped as a
    real one near theta_E would be."""
    cfg = tier0_tsang("fixed15" if concentration == 15.0 else ("fixed30" if concentration == 30.0 else "fixed60"))
    cfg.subhalo.tau = float(tau_ref * concentration / c_ref)
    cfg.tier = f"tier0_tidal_c{concentration:g}_tau{cfg.subhalo.tau:g}"
    return cfg


def tier0_no_subhalo() -> SimConfig:
    """Tier 0 with the subhalo switched off entirely — the zero-subhalo
    control population used for confounder false-positive-rate measurements
    (RQ2): run a detector on these and count how often it still reports
    substructure."""
    cfg = tier0_tsang()
    cfg.subhalo.enabled = False
    cfg.subhalo.presence_prob = 0.0
    cfg.tier = "tier0_no_subhalo"
    return cfg


def tier1_cosmos(concentration_mode: str = "fixed60") -> SimConfig:
    """Tier 1: Tsang+2024's macro/subhalo priors, unchanged, with the source
    swapped from single-Sersic to a real COSMOS postage stamp -- the one
    axis Tier 0 explicitly did not cover (see `SourceConfig`'s docstring)."""
    cfg = tier0_tsang(concentration_mode)
    cfg.source.source_type = "cosmos"
    cfg.tier = f"tier1_cosmos_{concentration_mode}"
    return cfg


def tier2_multipole_confounder(am_over_thetaE: float = 0.03, m: int = 4) -> SimConfig:
    """Tier 2 confounder: no subhalo, a multipole switched on. am/thetaE=0.03
    is O'Riordan+2025's amplitude at which the hidden-subhalo-population
    signal becomes "consistent with zero" — i.e. the amplitude at which a
    detector *should* stay quiet if it is sensitive to the right thing."""
    cfg = tier0_no_subhalo()
    cfg.multipole.enabled = True
    cfg.multipole.am_over_thetaE = am_over_thetaE
    cfg.multipole.m = m
    cfg.tier = f"tier2_multipole_m{m}_a{am_over_thetaE}"
    return cfg


def tier1_cosmos_no_subhalo(concentration_mode: str = "fixed60") -> SimConfig:
    """Tier-1 (COSMOS source) equivalent of `tier0_no_subhalo` — the RQ2
    zero-subhalo control population, needed separately because the Tier-0
    version defaults to a Sersic source."""
    cfg = tier1_cosmos(concentration_mode)
    cfg.subhalo.enabled = False
    cfg.subhalo.presence_prob = 0.0
    cfg.tier = "tier1_cosmos_no_subhalo"
    return cfg


def tier1_cosmos_multipole_confounder(am_over_thetaE: float = 0.03, m: int = 4) -> SimConfig:
    """Tier-1 (COSMOS source) equivalent of `tier2_multipole_confounder` — the
    RQ2 confounder test, now with real source structure, since a COSMOS
    source's own clumpiness could plausibly change how detectable a multipole
    shape-confounder is."""
    cfg = tier1_cosmos_no_subhalo()
    cfg.multipole.enabled = True
    cfg.multipole.am_over_thetaE = am_over_thetaE
    cfg.multipole.m = m
    cfg.tier = f"tier1_cosmos_multipole_m{m}_a{am_over_thetaE}"
    return cfg


def tier2_lens_light(concentration_mode: str = "fixed60") -> SimConfig:
    """Tier 2 realism axis: Tier-0 lens + subhalo priors with two-component lens
    light switched on (see `LensLightConfig`)."""
    cfg = tier0_tsang(concentration_mode)
    cfg.lens_light.enabled = True
    cfg.tier = f"tier2_lenslight_{concentration_mode}"
    return cfg


def tier2_lens_light_no_subhalo() -> SimConfig:
    cfg = tier0_no_subhalo()
    cfg.lens_light.enabled = True
    cfg.tier = "tier2_lenslight_no_subhalo"
    return cfg


def tier2_lens_light_multipole(am_over_thetaE: float = 0.03, m: int = 4) -> SimConfig:
    cfg = tier2_multipole_confounder(am_over_thetaE=am_over_thetaE, m=m)
    cfg.lens_light.enabled = True
    cfg.tier = f"tier2_lenslight_multipole_m{m}_a{am_over_thetaE}"
    return cfg


def tier0_population(concentration_mode: str = "fixed60", no_subhalo: bool = False) -> SimConfig:
    """Tier 0 plus the sub-threshold subhalo and line-of-sight population (`PopulationConfig`), with the same seeds as
    the Tier-0 populations so the lenses pair with them."""
    cfg = tier0_no_subhalo() if no_subhalo else tier0_tsang(concentration_mode)
    cfg.population.enabled = True
    cfg.tier = "tier0_population_" + ("no_subhalo" if no_subhalo else concentration_mode)
    return cfg


def tier0_euclid(concentration_mode: str = "fixed60", no_subhalo: bool = False, am_over_thetaE: float = 0.0, m: int = 4,
                 source_mag_offset: float = 0.0) -> SimConfig:
    """A Euclid-VIS-like survey arm (referee round 27, M4): lenstronomy's Euclid VIS preset (0.101" pixels, Gaussian PSF of
    0.16", 4 x 566 s, its own zero point), the Tier-0 priors and (given the same seed) the same lenses, sources and subhalos.
    Source amplitudes are rescaled from the F160W to the VIS zero point so the sources keep their magnitudes, and then made
    fainter by `source_mag_offset` magnitudes, set so the median arc signal-to-noise is survey-typical."""
    cfg = tier0_tsang(concentration_mode)
    cfg.instrument.survey, cfg.instrument.band, cfg.instrument.pixel_scale = "Euclid", "VIS", 0.101
    zp_scale = 10 ** (-0.4 * (25.96 - 25.72))
    f = zp_scale * 10 ** (-0.4 * source_mag_offset)
    cfg.source.amp_range = (cfg.source.amp_range[0] * f, cfg.source.amp_range[1] * f)
    if no_subhalo:
        cfg.subhalo.enabled = False
        cfg.subhalo.presence_prob = 0.0
    if am_over_thetaE:
        cfg.multipole.enabled = True
        cfg.multipole.am_over_thetaE = am_over_thetaE
        cfg.multipole.m = m
    cfg.tier = f"tier0_euclid_{'no_subhalo' if no_subhalo else concentration_mode}" + (f"_m{m}a{am_over_thetaE}" if am_over_thetaE else "") + (f"_dm{source_mag_offset:g}" if source_mag_offset else "")
    return cfg
