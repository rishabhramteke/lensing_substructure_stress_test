"""The two strongest learned detectors on every stress population (referee round 27, M1): the U-Net trained on 500 000
c=60 images (the published scale) and the neural ratio estimator trained on 100 000 (scripts/train_image_detector.py).

Confounder populations (multipoles, decoys, line-of-sight halos) are scored at each detector's own Tier-0 10% threshold.
Re-rendered arms (depth, PSF, lens light, combined realism, misspecification, sub-threshold population, Tier 1, Euclid)
are recalibrated on the arm's own clean control, as every family is elsewhere, and give c=60 completeness per mass bin and
the AUC against that control; the detectors are not retrained.

    python scripts/evaluate_strong_detectors.py      -> results/strong_detectors.json, paper/tables/strong_detectors.tex
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from detector.dataset import load_population, normalize  # noqa: E402
from detector.unet import build_model  # noqa: E402
import train_image_detector as TID  # noqa: E402

BINS = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]
CONF = ["multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3", "ellmultipole_m4_a3",
        "decoy_gaussian_s42_a10", "decoy_gaussian_s43_a10", "decoy_dipole_s42_a10", "los_halo_z025", "los_halo_z075"]
ARMS = {  # arm: (clean, c60)
    "shallow600": ("test_shallow600_no_subhalo", "test_shallow600_fixed60"), "shallow135": ("test_shallow_no_subhalo", "test_shallow_fixed60"),
    "psf015": ("psf015_no_subhalo", "psf015_fixed60"), "lenslight": ("lenslight_no_subhalo", "lenslight_fixed60"),
    "realistic": ("realistic_no_subhalo", "realistic_fixed60"), "clumpy": ("clumpy_no_subhalo", "clumpy_fixed60"),
    "supersampled": ("super_no_subhalo", "super_fixed60"), "render_psf": ("invcrime_no_subhalo", "invcrime_fixed60"), "population": ("pop_no_subhalo", "pop_fixed60"),
    "tier1": ("tier1/no_subhalo", "tier1/test_fixed60"), "euclid": ("euclid_no_subhalo", "euclid_fixed60")}


def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class Det:
    def __init__(self, kind):
        self.kind, self.d = kind, dev()
        if kind == "unet500k":
            c = torch.load(ROOT / "checkpoints/unet_30k_c60pub_seed0/model_best.pt", map_location=self.d)
            self.m = build_model(c).to(self.d); self.m.load_state_dict(c["model_state"]); self.m.eval(); self.scale = c["scale"]
            self.thr = json.loads((ROOT / "results/detector_30k_c60pub_seed0/results.json").read_text())["threshold_at_10pct_fpr"]
        else:
            c = torch.load(ROOT / "checkpoints/nre_100k_c60_seed0/model_best.pt", map_location=self.d)
            self.m = TID.NRE().to(self.d); self.m.load_state_dict(c["model_state"]); self.m.eval(); self.scale = c["scale"]
            self.prior = np.load(ROOT / "checkpoints/nre_100k_c60_seed0/prior_theta.npy")
            self.thr = json.loads((ROOT / "results/nre_100k_c60_seed0/results.json").read_text())["threshold_at_10pct_fpr"]

    @torch.no_grad()
    def score(self, pop):
        imgs, truths, *_ = load_population(ROOT / "data" / pop)
        x = normalize(imgs, self.scale)
        if self.kind == "unet500k":
            t = torch.from_numpy(x).unsqueeze(1)
            s = np.concatenate([torch.sigmoid(self.m(t[a:a + 128].to(self.d))).amax(dim=(1, 2, 3)).cpu().numpy() for a in range(0, len(t), 128)])
        else:
            s, _ = TID.score(self.m, "nre", x, self.prior, self.d)
        return s, truths


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def main():
    out = {}
    for kind in ("unet500k", "nre"):
        D = Det(kind); R = {"tier0_threshold": float(D.thr), "confounders": {}, "arms": {}}
        for pop in CONF:
            if (ROOT / "data" / pop / "manifest.json").exists():
                s, _ = D.score(pop); R["confounders"][pop] = {"n": len(s), "flag_rate": float(np.mean(s >= D.thr))}
        for arm, (cl, c60) in ARMS.items():
            if not ((ROOT / "data" / cl / "manifest.json").exists() and (ROOT / "data" / c60 / "manifest.json").exists()):
                continue
            sc, _ = D.score(cl); s6, T = D.score(c60)
            t = float(np.quantile(sc, 0.9)); sub = [(s6[i], tr["subhalo"]["log10_M200"]) for i, tr in enumerate(T) if tr.get("subhalo")]
            R["arms"][arm] = {"n_clean": len(sc), "auc": auc([a for a, _ in sub], sc), "frac_clean_above_tier0_thr": float(np.mean(sc >= D.thr)),
                              "completeness": [float(np.mean([a >= t for a, m in sub if lo <= m < hi])) if any(lo <= m < hi for _, m in sub) else None for lo, hi in BINS]}
        out[kind] = R
        print(kind, json.dumps(R)[:1500], flush=True)
    (ROOT / "results/strong_detectors.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


def table():
    """paper/tables/strong_detectors.tex from results/strong_detectors.json."""
    S = json.loads((ROOT / "results/strong_detectors.json").read_text())
    U, N = S["unet500k"], S["nre"]
    lab = {"multipole_m4_a1": r"multipole $m{=}4$, $a_4/\thetaE=0.01$", "multipole_m4_a3": r"multipole $m{=}4$, 0.03", "multipole_m3_a3": r"multipole $m{=}3$, 0.03",
           "ellmultipole_m4_a3": r"elliptical $m{=}4$, 0.03", "decoy_gaussian_s42_a10": r"decoy, Gaussian $10\sigma$ (draw 1)", "decoy_gaussian_s43_a10": r"decoy, Gaussian $10\sigma$ (draw 2)",
           "decoy_dipole_s42_a10": r"decoy, dipole $10\sigma$", "los_halo_z025": r"line-of-sight halo, $z=0.25$", "los_halo_z075": r"line-of-sight halo, $z=0.75$"}
    alab = {"shallow600": "600~s", "shallow135": "135~s", "psf015": r"$0.15''$ PSF", "lenslight": "lens light, raw", "realistic": "combined realism", "clumpy": "clumpy source",
            "supersampled": r"$5\times$ supersampled", "render_psf": r"$5\times$ supersampled, PSF $+10\%$", "population": "sub-threshold population", "tier1": "Tier~1 (COSMOS)", "euclid": "Euclid-VIS-like"}
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r" & \multicolumn{2}{c}{U-Net, $5\times10^{5}$} & \multicolumn{2}{c}{ratio estimator} \\",
         r"\multicolumn{5}{@{}l}{\emph{confounders: flagged (\%) at the Tier-0 calibration}} \\"]
    for p, t in lab.items():
        if p in U["confounders"]:
            L.append(f"{t} & \\multicolumn{{2}}{{c}}{{{100 * U['confounders'][p]['flag_rate']:.0f}}} & \\multicolumn{{2}}{{c}}{{{100 * N['confounders'][p]['flag_rate']:.0f}}} \\\\")
    L += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{re-rendered variants, recalibrated: AUC and $c=60$ completeness (\%) at $10^{9.5}$--$10^{10}\,\Msun$}} \\",
          r" & AUC & compl. & AUC & compl. \\"]
    for a, t in alab.items():
        if a in U["arms"]:
            u, n = U["arms"][a], N["arms"][a]
            L.append(f"{t} & {u['auc']:.2f} & {100 * u['completeness'][3]:.0f} & {n['auc']:.2f} & {100 * n['completeness'][3]:.0f} \\\\")
    L += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{ratio estimator retrained on 30\,000 images of the variant itself}} \\"]
    for run, t in (("nre_30k_shallow_seed0", "135~s"), ("nre_30k_realistic_seed0", "combined realism")):
        f = ROOT / "results" / run / "results.json"
        if f.exists():
            r = json.loads(f.read_text())
            L.append(f"{t} & -- & -- & {r['auc_test_fixed60']:.2f} & {100 * r['completeness_test_fixed60']['9.5-10.0']['completeness']:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/strong_detectors.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L))
