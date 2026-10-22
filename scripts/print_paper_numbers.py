"""Print, in one place, every number the revised manuscript quotes that is not already in a generated
table, read from the results files (so the prose can be checked against the runs)."""
import json
from pathlib import Path
import numpy as np
from paper_config import ROOT, BINS, A_JOINT_FULL, A_FROZEN_FULL, load_scan, UNET_AGG

def J(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None

def pc(x, nd=0):
    return "--" if x is None else f"{100*x:.{nd}f}"

js = J("results/joint_scan.json") or {}
print("=== ROC (results/roc.json)")
roc = J("results/roc.json")
if roc:
    for fam, d in roc["families"].items():
        for pop, r in d.items():
            print(f"  {fam:9s} {pop:18s} AUC {r['auc']:.3f}  TPR@10% {100*r['tpr_at_fpr0.1']:.0f}  TPR@1% {100*r['tpr_at_fpr0.01']:.0f}")
print("=== matched denominators (results/matched_denominators.json)")
md = J("results/matched_denominators.json")
if md:
    print("  common retained:", md["n_common_retained"], "thresholds", {k: round(v, 2) for k, v in md["thresholds"].items()}, "clean fpr", {k: round(v, 4) for k, v in md["clean_fpr"].items()})
    for fam in ("A", "A_abs", "B", "C"):
        for pop in ("test_fixed60", "test_fixed15"):
            m = md["families"][fam][pop]
            print(f"  {fam:6s} {pop:13s} matched", " ".join(pc(m['matched'][f'{lo}-{hi}']['p']) for lo, hi in BINS), "| own", " ".join(pc(m['own'][f'{lo}-{hi}']['p']) for lo, hi in BINS),
                  ("| conservative " + " ".join(pc(m['conservative'][f'{lo}-{hi}']['p']) for lo, hi in BINS)) if "conservative" in m else "")
print("=== signal variables (results/completeness_vs_signal.json)")
sg = J("results/completeness_vs_signal.json")
if sg:
    for fam in ("A", "B", "C"):
        for var, d in sg["joint"][fam].items():
            P = sg["permutation"][fam].get(var)
            rows = sg["comparison"][fam][var]
            md_ = np.mean([r["diff"] for r in rows]) if rows else np.nan
            print(f"  {fam} {var:16s} bins {len(rows)} mean diff {100*md_:+.0f} pts  chi2/dof {d['chi2_per_dof']:.1f} p {d['p_value']:.3g}  p_perm {P['p_permutation'] if P else '--'}")
        print("   aperture scan:", " | ".join(f"{r['aperture']:.3f}: p={r['p_value']:.3f}" for r in sg["aperture_scan"][fam] if r["p_value"] is not None))
    for fam in ("A", "B", "C"):
        for var, d in sg["power"][fam].items():
            if d: print(f"   power {fam} {var}: {d['min_detectable_diff_pts']:.1f} pts")
print("=== mass function reweighting")
mf = J("results/mass_function_reweighting/results.json")
if mf:
    for r in mf["results"]:
        print(f"  {r['label']:28s} flat {100*r['naive_unweighted']:.1f} -> weighted {100*r['cdm_mass_function_weighted']:.1f}")
print("=== localization (results/localization/summary_abc.json)")
lo = J("results/localization/summary_abc.json")
if lo:
    print("  A:", {k: v for k, v in lo["family_a"].items() if k not in ("a_root",)})
    print("  C:", lo["family_c_unet"])
print("=== depth (results/depth.json)")
dp = J("results/depth.json")
if dp:
    for arm in ("deep", "shallow"):
        A = dp[arm]["A"]; C = dp[arm]["C"]
        print(f"  {arm} A: thr {A['threshold_10pct']:.1f} clean abs {100*A['clean_fpr_abs']:.2f}% excluded {A['excluded']} | c60 cal", " ".join(pc(A['c60_cal'][k]['p']) for k in A['c60_cal']), "| abs", " ".join(pc(A['c60_abs'][k]['p']) for k in A['c60_abs']),
              "| c15 cal", " ".join(pc(A['c15_cal'][k]['p']) for k in A['c15_cal']), "| c15 abs", " ".join(pc(A['c15_abs'][k]['p']) for k in A['c15_abs']), f"| mp {100*A['mp_cal']:.0f}/{100*A['mp_abs']:.0f} | flips {A['flip_cal']} {A['flip_abs']}")
        if C:
            print(f"  {arm} C deep-trained: clean FPR at stored thr {100*C['fpr_clean_at_stored_threshold']:.0f}%  AUC {C['auc_c60']['mean']:.3f} | c60 naive", " ".join(pc(C['c60_naive'][k]['mean']) for k in C['c60_naive']), "| recal", " ".join(pc(C['c60_recal'][k]['mean']) for k in C['c60_recal']), "| c15 recal", " ".join(pc(C['c15_recal'][k]['mean']) for k in C['c15_recal']))
    CR = dp["shallow"].get("C_retrained")
    if CR:
        print(f"  shallow C retrained: AUC {CR['auc_c60']['mean']:.3f} | c60", " ".join(pc(CR['c60_recal'][k]['mean']) for k in CR['c60_recal']), "| c15", " ".join(pc(CR['c15_recal'][k]['mean']) for k in CR['c15_recal']))
print("=== joint core extras")
if js.get("core"):
    C = js["core"]
    for pop in ("multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
        recs = load_scan(A_JOINT_FULL, pop)
        if recs:
            rel = [r["delta_chi2"] for r in recs if r["reliable_fit"]]
            print(f"  {pop}: median dchi2 {np.median(rel):.0f}  n_rel {len(rel)}")
    recs = load_scan(A_FROZEN_FULL, "multipole_m4_a3"); print("  frozen mp_a3 median dchi2", round(float(np.median([r['delta_chi2'] for r in recs if r['reliable_fit']])), 1))
    for k in ("cal", "abs"):
        o = C[k]
        print(f"  {k}: clean FPR {100*o['fpr_clean']['p']:.2f}", {p: pc(o[f'fpr_{p}']['p'], 1) for p in ('multipole_m4_a1', 'multipole_m4_a3', 'multipole_m3_a3') if f'fpr_{p}' in o},
              "| c60", " ".join(pc(v['p']) for v in o['completeness_c60'].values()), "| c30", " ".join(pc(v['p']) for v in o['completeness_c30'].values()) if 'completeness_c30' in o else "", "| c15", " ".join(pc(v['p']) for v in o['completeness_c15'].values()),
              "| flips", o.get("flip_60_to_15"), o.get("flip_60_to_30"), o.get("flip_30_to_15"), "| det", o["detections_c60"]["n"], "loc", pc(o["detections_c60"].get("frac_localized"), 1), "merr", round(o["detections_c60"]["mass_error_all"]["median"], 2), "loc merr", (o["detections_c60"].get("mass_error_localized") or {}).get("median"))
    print("  runs:", {p: (v["n"], v["n_unreliable"], v["n_truncated"], round(v["cost_s_median_fit_plus_scan"], 1)) for p, v in C["runs"].items()})
    print("  clean quantiles", C["clean_dchi2_quantiles"], "neg frac", round(C["clean_frac_negative"], 4), "gates", [(g["gate"], round(g["frac_clean_pass"], 2), round(g["multipole_m4_a3"]["frac_pass"], 3), pc(g["multipole_m4_a3"]["fpr_cal"]), pc(g["multipole_m4_a3"]["fpr_abs"])) for g in C["gate"] if "multipole_m4_a3" in g])
for key in ("psf015", "psf015_frozen", "shallow", "lenslight_single", "lenslight_double", "tidal", "decoys", "los", "freepos"):
    d = js.get(key)
    if not d:
        print(f"=== {key}: none"); continue
    print(f"=== {key}")
    if key == "decoys":
        for pop, v in d["populations"].items(): print(f"  {pop}: cal {pc(v['cal']['p'])} abs {pc(v['abs']['p'])} n_rel {v['n_reliable']}")
    elif key == "los":
        for arm, v in d["arms"].items():
            for k in ("cal", "abs"):
                det = v[k].get("detections", {})
                print(f"  {arm} {k}: flag {pc(v[k]['flag_rate']['p'])} bins", " ".join(pc(b['p']) for b in v[k]['bins'].values()), "| loc", pc(det.get("frac_localized")), "off", det.get("median_offset_arcsec"), "merr", det.get("mass_error_median"), "n_rel", v["n_reliable"], "trunc", v["n_truncated"])
    elif key == "tidal":
        for pop, v in d["populations"].items():
            for k in ("cal", "abs"): print(f"  {pop} {k}:", " ".join(pc(b['p']) for b in v[k].values()), "n", [b['n'] for b in v[k].values()])
    else:
        print("  runs", {k: (v["n"], v["n_unreliable"]) for k, v in d["runs"].items()})
        for k in ("cal", "abs"):
            o = d[k]; line = f"  {k}: thr {o['threshold']:.1f} clean FPR {100*o['fpr_clean']['p']:.2f}"
            if "fpr_multipole_m4_a3" in o: line += f" mp {pc(o['fpr_multipole_m4_a3']['p'])}"
            if "completeness_c60" in o: line += " | c60 " + " ".join(pc(v['p']) for v in o['completeness_c60'].values()) + " n " + str([v['n'] for v in o['completeness_c60'].values()])
            if "completeness_c15" in o: line += " | c15 " + " ".join(pc(v['p']) for v in o['completeness_c15'].values())
            if "flip_60_to_15" in o: line += f" | flip {o['flip_60_to_15']}"
            if o.get("detections_c60", {}).get("n"): line += f" | det {o['detections_c60']['n']} loc {pc(o['detections_c60']['frac_localized'])} merr {o['detections_c60']['mass_error_all']['median']:+.2f}"
            print(line)
print("=== U-Net populations")
up = J("results/unet_populations.json")
if up:
    for pop, v in up["populations"].items():
        a = v["aggregate"]; line = f"  {pop:28s} flag {100*a['flag_rate']['mean']:.1f}±{100*a['flag_rate']['std']:.1f}"
        if "completeness" in a: line += " compl " + " ".join(pc(b['mean']) for b in a['completeness'].values())
        print(line)
for tag in ("_psf015_deeptrained", "_psf015_psftrained", "_lenslight_raw"):
    u = J(f"results/unet_populations{tag}.json")
    if u:
        print(f"  -- {tag}")
        for pop, v in u["populations"].items():
            a = v["aggregate"]; line = f"  {pop:28s} flag {100*a['flag_rate']['mean']:.1f} recal {pc(a.get('flag_rate_recal', {}).get('mean'))}"
            if "completeness" in a: line += " compl " + " ".join(pc(b['mean']) for b in a['completeness'].values()) + " | recal " + " ".join(pc(b) for b in a.get('completeness_recal', {}).values()) + f" AUC {a.get('auc_vs_clean', {}).get('mean')}"
            print(line)
ll = J("results/lens_light_experiment.json")
if ll:
    print("=== lens light experiment keys:", list(ll))
    for k in ("family_a", "family_a_double_sersic", "family_a_frozen", "family_a_frozen_double_sersic", "family_b_single_sersic", "family_b_double_sersic"):
        d = ll.get(k)
        if d: print(f"  {k}: rej {d['n_unreliable']} thr {d['threshold_10pct_fpr']:.1f} clean abs {d['fpr_clean_abs']} mp {d.get('fpr_multipole_a3')} compl", {b: pc(v['completeness']) for b, v in d.get('completeness_c60', {}).items()})
    u = ll.get("unet")
    if u: print("  unet:", {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('fpr_clean', 'auc_c60_vs_clean', 'fpr_multipole_a3', 'fpr_clean_at_original_threshold', 'fpr_multipole_a3_recalibrated')}) for k, v in u.items() if k != 'seeds'})

# ---- referee round 19 additions: gate bookkeeping, c-M mass bias, free-position LOS, joint multipole freedom, low-c U-Net
print("=== gate bookkeeping (results/joint_scan.json: gate)")
for pop, g in (js.get("gate") or {}).items():
    print(f"  {pop:16s} gated {g['gated_initial']:3d} -> {g['gated_final']:3d}  basin {g['rescued_basin']} joint {g['rescued_joint']} multistart {g['rescued_multistart']}  excluded subhalo bins {g.get('excluded_subhalo_by_bin')}")
C = (js.get("core") or {})
if C:
    print("  clean quantiles", {k: round(v, 2) for k, v in C["clean_dchi2_quantiles"].items()}, "frac<0", round(C["clean_frac_negative"], 4), "clean FPR abs", round(100 * C["abs"]["fpr_clean"]["p"], 3), "%")
    for k in ("cal", "abs"):
        o = C[k]
        print(f"  {k}: c60 retained", " ".join(pc(v["p"]) for v in o["completeness_c60"].values()), "| conservative", " ".join(pc(v["p"]) for v in o["conservative_c60"].values()),
              "| c15", " ".join(pc(v["p"]) for v in o["completeness_c15"].values()), "| c30", " ".join(pc(v["p"]) for v in o["completeness_c30"].values()))
        print(f"       flips 60->15 {o['flip_60_to_15']}  60->30 {o['flip_60_to_30']}  30->15 {o['flip_30_to_15']}")
        d = o["detections_c60"]; print(f"       det {d['n']} loc {100*d['frac_localized']:.1f}% off {d['median_offset_arcsec']:.2f} mass all {d['mass_error_all']['median']:+.2f} (low {d['mass_error_all']['n_low']}, within0.5 {d['mass_error_all']['n_within_0p5']}) localized {d['mass_error_localized']}")
        for pop in ("multipole_m4_a1", "multipole_m4_a3", "multipole_m3_a3"):
            f = o[f"fpr_{pop}"]; print(f"       {pop} FPR {100*f['p']:.1f}% [{100*f['lo']:.1f},{100*f['hi']:.1f}] n={f['n']}")
    print("  runs:", {p_: (v["n_unreliable"], v.get("n_unreliable_subhalo_only")) for p_, v in C["runs"].items()})
    print("  abs100 multipole a3", pc(C["abs100"]["fpr_multipole_m4_a3"]["p"], 1), "a1", pc(C["abs100"]["fpr_multipole_m4_a1"]["p"], 1))
    mpd = C["cal"].get("fpr_multipole_m4_a3")
cm = js.get("cm_template_on_cm_population")
if cm:
    print("=== c-M template on c-M population:", {k: cm[k] for k in ("n", "n_unreliable")}, "true c median", round(cm["true_concentration_median"], 1), "| abs:", {k: cm["abs"].get(k) for k in ("n", "frac_localized", "mass_error_all")})
lf = js.get("los_freepos")
if lf:
    for arm, d in lf["arms"].items():
        print(f"=== LOS free-position {arm}: abs flag {100*d['abs']['flag_rate']['p']:.0f}% det {d['abs'].get('detections')}")
fp = js.get("freepos")
if fp:
    for k in ("cal", "abs"):
        d = fp[k]["detections_c60"]; print(f"=== freepos {k}: thr {fp[k]['threshold']:.1f} det {d['n']} loc {100*d['frac_localized']:.1f}% off {d['median_offset_arcsec']:.3f} | c60 compl", " ".join(pc(v['p']) for v in fp[k]['completeness_c60'].values()), "| mp", pc(fp[k]['fpr_multipole_m4_a3']['p']), "| clean fpr", pc(fp[k]['fpr_clean']['p'], 2))
mj = J("results/multipole_orders_joint_summary.json")
if mj:
    print("=== joint multipole freedom")
    for k, d in mj.items():
        print(f"  {k:5s} thr {d['threshold_10pct_fpr']}  rejected {d['rejected']}  a1 {d.get('multipole_m4_a1')}  a3 {d.get('multipole_m4_a3')}  m3 {d.get('multipole_m3_a3')}")
        if d.get("c60"):
            print("        c60 cal", " ".join(pc(v["p"]) for v in d["c60"]["completeness"]["cal"].values()), "| abs", " ".join(pc(v["p"]) for v in d["c60"]["completeness"]["abs"].values()), "| loc", pc(d["c60"]["frac_localized"]), "| flip", d.get("flip_60_to_15"))
ul = J("results/unet_lowc.json")
if ul:
    print("=== U-Net low-c training")
    for k, d in ul.items():
        if d.get("present"):
            print(f"  {k:4s} seeds {d['n_seeds']} AUC {d['auc']['mean']:.3f} | c60", " ".join(pc(v['mean']) for v in d['completeness_c60'].values()), "| c15", " ".join(pc(v['mean']) for v in d['completeness_c15'].values()), "| flip", d.get("paired_flip"), "| mp a3", pc(d["confounder_fpr"]["multipole m=4, a=0.03*thetaE"]["mean"]))
dp = J("results/depth.json")
if dp and dp.get("mid"):
    A = dp["mid"]["A"]; Cc = dp["mid"].get("C")
    if A:
        print("=== depth 600 s A: thr", round(A["threshold_10pct"], 1), "clean abs", pc(A["clean_fpr_abs"], 2), "excluded", A["excluded"], "| c60 cal", " ".join(pc(A["c60_cal"][k]["p"]) for k in A["c60_cal"]), "| abs", " ".join(pc(A["c60_abs"][k]["p"]) for k in A["c60_abs"]), "| mp", (pc(A.get("mp_cal")), pc(A.get("mp_abs"))))
    if Cc:
        print("    600 s C deep-trained: clean FPR at deep thr", pc(Cc["fpr_clean_at_stored_threshold"]), "AUC", round(Cc["auc_c60"]["mean"], 3), "| c60 recal", " ".join(pc(Cc["c60_recal"][k]["mean"]) for k in Cc["c60_recal"]))
ll = J("results/lens_light_experiment.json")
if ll:
    for key in ("family_a", "family_a_double_sersic"):
        d = ll.get(key)
        if d: print(f"=== lens light {key}: n_unreliable {d.get('n_unreliable')} fpr_clean_abs {d.get('fpr_clean_abs')} fpr_mp {d.get('fpr_multipole_a3')} thr {d.get('threshold_10pct_fpr')} compl", {k: round(v['completeness'], 2) if v.get('completeness') is not None else None for k, v in d.get('completeness_c60', {}).items()})
