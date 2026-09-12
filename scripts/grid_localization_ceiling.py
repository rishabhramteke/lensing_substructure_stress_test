"""Geometric localization ceiling of Family A's 3-radii x 8-angle scan grid (provenance for the
21.5% / 0.23" numbers quoted in Sect. 4.3; added 2026-09-12 at a referee's request).

Draws subhalo positions from the simulator's own placement prior (uniform in radius over
0.6-1.3 theta_E, uniform in angle, theta_E ~ U(0.8, 1.2)) and asks how many lie within the
2-pixel (0.16") localization criterion of the nearest grid node (radii 0.5/0.95/1.4 theta_E,
8 angles), i.e. the largest localized fraction a perfect scan on this grid could report.

    python scripts/grid_localization_ceiling.py  -> results/grid_localization_ceiling.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0); N = 400_000
tE = rng.uniform(0.8, 1.2, N); r = rng.uniform(0.6, 1.3, N) * tE; a = rng.uniform(0, 2 * np.pi, N)
x, y = r * np.cos(a), r * np.sin(a)
radii = np.linspace(0.5, 1.4, 3)[:, None] * tE[None, :]; angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
best = np.full(N, np.inf)
for i in range(3):
    for th in angles:
        best = np.minimum(best, np.hypot(radii[i] * np.cos(th) - x, radii[i] * np.sin(th) - y))
out = {"n_draws": N, "criterion_arcsec": 0.16, "grid": "radii 0.5, 0.95, 1.4 theta_E x 8 angles",
       "placement_prior": "r ~ U(0.6, 1.3) theta_E, angle ~ U(0, 2pi), theta_E ~ U(0.8, 1.2)",
       "fraction_within_criterion_of_a_node": float((best < 0.16).mean()),
       "median_nearest_node_distance_arcsec": float(np.median(best)),
       "radial_node_spacing_arcsec_at_thetaE_1": 0.45, "angular_node_spacing_arcsec_at_r_1": float(2 * np.pi / 8)}
(ROOT / "results/grid_localization_ceiling.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
