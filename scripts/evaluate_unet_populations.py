"""Score a U-Net seed set on arbitrary populations: flag rate at each seed's own 10%-FPR threshold
(from results/<run>/results.json), completeness per mass bin where the population carries subhalos,
and AUC against a clean control. Used for the stress populations the per-family evaluators do not
cover for the network: line-of-sight halos, the decoy populations on disk, the PSF-widened tier and
the raw lens-light images (2026-09-15).

    python scripts/evaluate_unet_populations.py                       # primary set (paper_config.UNET_SET) on the default list
    python scripts/evaluate_unet_populations.py --set psf --pops psf015_fixed60 psf015_fixed15 --clean psf015_no_subhalo
-> results/unet_populations[_<set>].json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

from paper_config import ROOT, BINS, wilson, unet_seeds, UNET_SETS, UNET_SET, UNET_SHALLOW, UNET_PSF
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import LensPatchDataset  # noqa: E402
from detector.unet import UNet  # noqa: E402

DEFAULT_POPS = ["los_halo_z025", "los_halo_z075",
                "decoy_gaussian_s42_a3", "decoy_gaussian_s42_a6", "decoy_gaussian_s42_a10",
                "decoy_gaussian_s43_a3", "decoy_gaussian_s43_a6", "decoy_gaussian_s43_a10",
                "decoy_dipole_s42_a3", "decoy_dipole_s42_a6", "decoy_dipole_s42_a10",
                "psf015_no_subhalo", "psf015_fixed60", "psf015_fixed15", "psf015_multipole_m4_a3",
                "lenslight_no_subhalo", "lenslight_fixed60", "lenslight_multipole_m4_a3"]


def scores(model, ds, dev, batch=64):
    out = np.zeros(len(ds), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(ds), batch):
            idx = range(s, min(s + batch, len(ds)))
            x = torch.stack([ds[i][0] for i in idx]).to(dev)
            out[list(idx)] = torch.sigmoid(model(x)).amax(dim=(1, 2, 3)).cpu().numpy()
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) < 5 or len(neg) < 5:
        return None
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=UNET_SET, help="30k | 8k | shallow | psf")
    ap.add_argument("--pops", nargs="+", default=DEFAULT_POPS)
    ap.add_argument("--clean", default="no_subhalo", help="clean control for the AUC (and for a recalibrated threshold on a re-rendered tier)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    seeds = {"shallow": UNET_SHALLOW, "psf": UNET_PSF}.get(args.set) or UNET_SETS[args.set]
    seeds = {r: c for r, c in seeds.items() if (ROOT / c / "model_best.pt").exists() and (ROOT / "results" / r / "results.json").exists()}
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    res = {"set": args.set, "seeds": list(seeds), "clean": args.clean, "populations": {}}
    models = {}
    for run, ck in seeds.items():
        c = torch.load(ROOT / ck / "model_best.pt", map_location=dev)
        m = UNet(in_ch=1, base=c.get("base", 16)).to(dev); m.load_state_dict(c["model_state"]); m.eval()
        models[run] = (m, c["scale"], json.loads((ROOT / "results" / run / "results.json").read_text())["threshold_at_10pct_fpr"])
    clean_scores = {}
    if (ROOT / "data" / args.clean / "manifest.json").exists():
        for run, (m, scale, thr) in models.items():
            clean_scores[run] = scores(m, LensPatchDataset(ROOT / "data" / args.clean, scale=scale), dev)
    for pop in args.pops:
        pdir = ROOT / "data" / pop
        if not (pdir / "manifest.json").exists():
            print("missing", pop); continue
        per = []
        for run, (m, scale, thr) in models.items():
            ds = LensPatchDataset(pdir, scale=scale); sc = scores(m, ds, dev)
            row = {"run": run, "threshold": thr, "n": len(ds), "flag_rate": float((sc >= thr).mean())}
            if run in clean_scores:
                thr_own = float(np.quantile(clean_scores[run], 0.90))
                row["threshold_recal_on_clean"] = thr_own; row["flag_rate_recal"] = float((sc >= thr_own).mean())
            sub_idx = [i for i, t in enumerate(ds.truths) if t.get("subhalo")]
            los_idx = [i for i, t in enumerate(ds.truths) if t.get("los_halo")]
            if sub_idx:
                masses = np.array([ds.truths[i]["subhalo"]["log10_M200"] for i in sub_idx]); s = sc[sub_idx]
                row["completeness"] = {f"{lo}-{hi}": {"n": int(((masses >= lo) & (masses < hi)).sum()), "p": float((s[(masses >= lo) & (masses < hi)] >= thr).mean()) if ((masses >= lo) & (masses < hi)).any() else None} for lo, hi in BINS}
                if run in clean_scores:
                    row["auc_vs_clean"] = auc(s, clean_scores[run])
                    thr_own = row["threshold_recal_on_clean"]
                    row["completeness_recal"] = {f"{lo}-{hi}": float((s[(masses >= lo) & (masses < hi)] >= thr_own).mean()) if ((masses >= lo) & (masses < hi)).any() else None for lo, hi in BINS}
            if los_idx:
                masses = np.array([ds.truths[i]["los_halo"]["log10_M200"] for i in los_idx]); s = sc[los_idx]
                row["flag_by_los_mass"] = {f"{lo}-{hi}": float((s[(masses >= lo) & (masses < hi)] >= thr).mean()) if ((masses >= lo) & (masses < hi)).any() else None for lo, hi in BINS}
            per.append(row)
        agg = {"n_seeds": len(per), "n": per[0]["n"], "flag_rate": {"mean": float(np.mean([r["flag_rate"] for r in per])), "std": float(np.std([r["flag_rate"] for r in per]))}}
        if "flag_rate_recal" in per[0]:
            agg["flag_rate_recal"] = {"mean": float(np.mean([r["flag_rate_recal"] for r in per])), "std": float(np.std([r["flag_rate_recal"] for r in per]))}
        if "completeness" in per[0]:
            agg["completeness"] = {k: {"n": per[0]["completeness"][k]["n"], "mean": float(np.mean([r["completeness"][k]["p"] for r in per])) if per[0]["completeness"][k]["p"] is not None else None,
                                       "std": float(np.std([r["completeness"][k]["p"] for r in per])) if per[0]["completeness"][k]["p"] is not None else None} for k in per[0]["completeness"]}
            if "auc_vs_clean" in per[0] and per[0]["auc_vs_clean"] is not None:
                agg["auc_vs_clean"] = {"mean": float(np.mean([r["auc_vs_clean"] for r in per])), "std": float(np.std([r["auc_vs_clean"] for r in per]))}
                agg["completeness_recal"] = {k: float(np.mean([r["completeness_recal"][k] for r in per])) if per[0]["completeness_recal"][k] is not None else None for k in per[0]["completeness_recal"]}
        if "flag_by_los_mass" in per[0]:
            agg["flag_by_los_mass"] = {k: float(np.mean([r["flag_by_los_mass"][k] for r in per])) if per[0]["flag_by_los_mass"][k] is not None else None for k in per[0]["flag_by_los_mass"]}
        res["populations"][pop] = {"aggregate": agg, "per_seed": per}
        line = f"{pop:28s} flag {100*agg['flag_rate']['mean']:5.1f} ± {100*agg['flag_rate']['std']:.1f}%"
        if "flag_rate_recal" in agg:
            line += f"  (recal {100*agg['flag_rate_recal']['mean']:.1f}%)"
        if "completeness" in agg:
            line += "  compl " + " ".join(f"{100*v['mean']:3.0f}" if v["mean"] is not None else " --" for v in agg["completeness"].values())
            if "auc_vs_clean" in agg:
                line += f"  AUC {agg['auc_vs_clean']['mean']:.3f}"
        if "flag_by_los_mass" in agg:
            line += "  by LOS mass " + " ".join(f"{100*v:3.0f}" if v is not None else " --" for v in agg["flag_by_los_mass"].values())
        print(line, flush=True)
    tag = args.tag if args.tag is not None else ("" if args.set == UNET_SET else f"_{args.set}")
    out = ROOT / f"results/unet_populations{tag}.json"
    out.write_text(json.dumps(res, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main()
