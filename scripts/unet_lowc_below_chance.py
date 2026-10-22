"""Why do the U-Nets trained at c=15 / on the c-M relation score AUC slightly BELOW 0.5 (referee round 20)?

Each c=60 test image has a noise-matched subhalo-free twin (images_control_noiseless + the image's own noise
realisation), so the perturbation's effect on a network's score can be measured lens by lens, free of the
lens-to-lens sampling variation that an AUC against the independent clean control carries. For each network:
AUC(c=60 vs clean), AUC(twins vs clean) (the null: same lenses, no subhalo), the paired fraction of lenses whose
score DROPS when the subhalo is added, and that fraction per mass bin. Writes results/unet_lowc_below_chance.json.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detector.dataset import normalize  # noqa: E402
from detector.unet import UNet  # noqa: E402

NETS = {"c60_seed0": "unet_30k_seed0", "c15_seed0": "unet_30k_c15_seed0", "c15_seed1": "unet_30k_c15_seed1",
        "cm_seed0": "unet_30k_cm_seed0", "cm_seed1": "unet_30k_cm_seed1"}
dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    allv = np.concatenate([neg, pos]); r = allv.argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def scores(model, scale, imgs):
    out = []
    with torch.no_grad():
        for k in range(0, len(imgs), 64):
            x = torch.from_numpy(normalize(imgs[k:k + 64], scale)).unsqueeze(1).to(dev)
            out.append(torch.sigmoid(model(x)).amax(dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


clean = np.load(ROOT / "data/no_subhalo/images_full_noisy.npy")
p60 = ROOT / "data/test_fixed60"
full = np.load(p60 / "images_full_noisy.npy"); noiseless = np.load(p60 / "images_full_noiseless.npy")
twin = np.load(p60 / "images_control_noiseless.npy") + (full - noiseless)
truths = [json.loads(l) for l in open(p60 / "truth.jsonl")]
has = np.array([t.get("subhalo") is not None for t in truths]); m = np.array([t["subhalo"]["log10_M200"] if t.get("subhalo") else np.nan for t in truths])
n1, n2 = len(clean), int(has.sum())
res = {"null_auc_sd_independent_sets": float(np.sqrt((n1 + n2 + 1) / (12 * n1 * n2))), "networks": {}}
for name, ck in NETS.items():
    c = torch.load(ROOT / "checkpoints" / ck / "model_best.pt", map_location=dev)
    net = UNet(in_ch=1, base=c.get("base", 16)).to(dev); net.load_state_dict(c["model_state"]); net.eval()
    sc, sf, st = scores(net, c["scale"], clean), scores(net, c["scale"], full), scores(net, c["scale"], twin)
    d = sf[has] - st[has]
    bins = {}
    for lo in (8.0, 8.5, 9.0, 9.5, 10.0, 10.5):
        sel = (m[has] >= lo) & (m[has] < lo + 0.5)
        bins[f"{lo}-{lo+0.5}"] = {"n": int(sel.sum()), "frac_score_drops": float((d[sel] < 0).mean()), "median_delta": float(np.median(d[sel]))}
    res["networks"][name] = {"auc_c60_vs_clean": auc(sc, sf[has]), "auc_twins_vs_clean": auc(sc, st[has]),
                             "paired_auc_c60_vs_own_twin": float((d > 0).mean() + 0.5 * (d == 0).mean()),
                             "frac_score_drops": float((d < 0).mean()), "by_mass": bins}
    r = res["networks"][name]
    print(f"{name}: AUC c60 {r['auc_c60_vs_clean']:.3f}  twins {r['auc_twins_vs_clean']:.3f}  paired {r['paired_auc_c60_vs_own_twin']:.3f}  "
          + " ".join(f"{k}:{v['frac_score_drops']:.2f}" for k, v in bins.items()), flush=True)
print("null AUC sd for independent 1000 vs ~900 sets:", round(res["null_auc_sd_independent_sets"], 4))
(ROOT / "results/unet_lowc_below_chance.json").write_text(json.dumps(res, indent=2))
