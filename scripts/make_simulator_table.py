"""Emit the simulator's parameter table straight from the config objects.

The Tier-0 setup was written out as ~130 numbers of running prose, which is hard to check and
easy to let drift from the code. This reads the dataclasses the simulator actually uses and the
instrument keywords lenstronomy actually returns, so the table cannot disagree with the runs.

    python scripts/make_simulator_table.py  ->  paper/tables/simulator.tex
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lensing.config import tier0_tsang, tier2_lens_light  # noqa: E402
from lensing.simulate import LensRenderer  # noqa: E402


def rng(pair, unit="", fmt="{:g}"):
    """A range as $lo$--$hi\\,unit$: the unit sits inside the closing maths so nothing breaks out."""
    lo, hi = pair
    return f"${fmt.format(lo)}$--${fmt.format(hi)}{unit}$"


def main():
    cfg = tier0_tsang("fixed60")
    ll = tier2_lens_light("fixed60").lens_light
    kb = LensRenderer(cfg).kwargs_band
    L, S, H, I, C = cfg.lens, cfg.source, cfg.subhalo, cfg.instrument, cfg.cosmology

    rows = [
        ("Macro-model", r"Einstein radius $\thetaE$", rng(L.theta_E_range, r"''"), r"\citet{tsang2024}"),
        ("", r"power-law slope $\gamma$", rng(L.gamma_range), r"\citet{tsang2024}"),
        ("", r"axis ratio $q$", rng(L.q_range), r"\citet{tsang2024}"),
        ("", r"external shear $|\gamma_{\rm ext}|$", f"$\\le{L.shear_mag_range[1]:g}$", "our choice"),
        ("", r"$z_{\rm lens}$, $z_{\rm source}$", f"${C.z_lens:g}$, ${C.z_source:g}$", r"\citet{ostdiek2020}"),
        (r"Source (Tier 0)", r"S\'ersic $R_{\rm e}$", rng(S.R_sersic_range, r"''"), r"\citet{ostdiek2020}"),
        ("", r"S\'ersic index $n$", rng(S.n_sersic_range), r"\citet{ostdiek2020}"),
        ("", r"offset from centre", rng(S.offset_frac_thetaE_range, r"\,\thetaE"), "our choice"),
        ("", r"unlensed magnitude", r"$19.5$--$24.7$", "from amplitudes"),
        ("Subhalo", r"$\log_{10}(M_{200}/\Msun)$", rng(H.log10_mass_range), r"\citet{tsang2024}"),
        ("", r"truncation $\tau=r_{\rm t}/r_{\rm s}$", f"${H.tau:g}$", r"\citet{tsang2024}"),
        ("", r"concentration $c$", r"$60$, $30$, $15$ or $c$--$M$", "switch; Sect.~\\ref{sec:rq4}"),
        ("", r"placement annulus", rng(H.placement_annulus_frac_thetaE, r"\,\thetaE"), "our choice"),
        ("", r"present in", f"${100*H.presence_prob:g}\\%$ of images", r"\citet{tsang2024}"),
        ("Confounder", r"multipole order $m$", r"$4$ (and $3$)", r"\citet{oriordan2025}"),
        ("", r"amplitude $a_m/\thetaE$", r"$0.01$, $0.03$", r"\citet{oriordan2025}"),
        ("Lens light", r"bulge $n$, $R_{\rm e}$", rng(ll.bulge_n_range) + ", " + rng(ll.bulge_R_range, r"''"), "our choice"),
        ("", r"envelope $R_{\rm e}$", rng(ll.envelope_R_range, r"''"), "our choice"),
        ("Instrument", r"pixel scale, grid", f"${I.pixel_scale:g}''$, ${I.num_pix}\\times{I.num_pix}$", r"\citet{tsang2024}"),
        ("", r"exposure time", f"${kb['exposure_time']:g}$~s", "HST preset"),
        ("", r"zero point", f"${kb['magnitude_zero_point']:g}$", "HST preset"),
        ("", r"sky brightness", f"${kb['sky_brightness']:g}$~mag\\,arcsec$^{{-2}}$", "HST preset"),
        ("", r"read noise, gain", f"${kb['read_noise']:g}\\,e^-$, ${kb['ccd_gain']:g}$", "HST preset"),
        ("", r"PSF FWHM (Gaussian)", f"${kb['seeing']:g}''$", "HST preset"),
    ]
    out = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
           r"component & parameter & value & source \\", r"\midrule"]
    prev = None
    for comp, par, val, src in rows:
        if comp and prev is not None:
            out.append(r"\addlinespace[2pt]")
        out.append(f"{comp} & {par} & {val} & {src} \\\\")
        prev = comp or prev
    out += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/simulator.tex").write_text("\n".join(out) + "\n")
    print(f"wrote paper/tables/simulator.tex ({len(rows)} rows)")


if __name__ == "__main__":
    main()
