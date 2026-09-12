"""Generate the Zel'dovich cosmic-web particle set used by web/dark_matter_lab.html.

ΛCDM Gaussian random field (BBKS transfer, normalised to sigma8) on an N^3 grid in a
periodic box; first-order Lagrangian displacement psi = -i k/k^2 delta_k; CIC density at
D=1 for colouring. Output: JSON with int16 displacements (units of cell/S) and uint8 density.

    source ~/myenv/bin/activate
    python src/make_cosmic_web.py --out cosmic_web.json
    python - <<'PY'   # then inject into the template
    t=open('web/dark_matter_lab.template.html').read(); d=open('cosmic_web.json').read()
    open('web/dark_matter_lab.html','w').write(t.replace('__CW_DATA__', d))
    PY
"""
import argparse, base64, json, time
import numpy as np


def bbks_T(k, gamma):
    q = k / gamma + 1e-12
    return np.log(1 + 2.34 * q) / (2.34 * q) * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25


def main(N=80, L=200.0, Om=0.3, h=0.7, ns=0.96, s8=0.8, seed=42, S=3000.0, out="cosmic_web.json"):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    gamma = Om * h
    P_un = lambda k: k ** ns * bbks_T(k, gamma) ** 2
    kk = np.logspace(-4, 2, 4000)
    W = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    s2 = np.trapezoid(P_un(kk) * W(kk * 8.0) ** 2 * kk ** 2, kk) / (2 * np.pi ** 2)
    A = s8 ** 2 / s2

    dx = L / N; Vc = dx ** 3
    kf = 2 * np.pi * np.fft.fftfreq(N, d=dx); kz = 2 * np.pi * np.fft.rfftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(kf, kf, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2; K = np.sqrt(K2); K[0, 0, 0] = 1e-12
    Wk = np.fft.rfftn(rng.standard_normal((N, N, N)))
    Pk = A * P_un(K); Pk[0, 0, 0] = 0
    dk = Wk * np.sqrt(Pk / Vc)            # <|delta_k^FFT|^2> = N^3 P(k) / Vc
    K2[0, 0, 0] = 1.0
    psi = np.stack([np.fft.irfftn(-1j * Ki / K2 * dk, s=(N, N, N), axes=(0, 1, 2)) for Ki in (KX, KY, KZ)], axis=-1).reshape(-1, 3)

    q = np.indices((N, N, N)).reshape(3, -1).T * dx
    g = ((q + psi) % L) / dx; i0 = np.floor(g).astype(int); f = g - i0
    rho = np.zeros((N, N, N))
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                w = (f[:, 0] if cx else 1 - f[:, 0]) * (f[:, 1] if cy else 1 - f[:, 1]) * (f[:, 2] if cz else 1 - f[:, 2])
                np.add.at(rho, ((i0[:, 0] + cx) % N, (i0[:, 1] + cy) % N, (i0[:, 2] + cz) % N), w)
    dens = rho[i0[:, 0] % N, i0[:, 1] % N, i0[:, 2] % N]
    ld = np.log10(dens + 0.05); ld = (ld - ld.min()) / (ld.max() - ld.min())

    blob = {"N": N, "L": L, "dx": dx, "S": S, "Om": Om, "h": h, "ns": ns, "s8": s8, "seed": seed,
            "psi_b64": base64.b64encode(np.clip(np.round(psi / dx * S), -32767, 32767).astype("<i2").tobytes()).decode(),
            "dens_b64": base64.b64encode((ld * 255).astype(np.uint8).tobytes()).decode()}
    json.dump(blob, open(out, "w"))
    print(f"{N**3:,} particles; psi rms {psi.std():.2f} Mpc/h; wrote {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=80); p.add_argument("--L", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--out", default="cosmic_web.json")
    a = p.parse_args(); main(N=a.N, L=a.L, seed=a.seed, out=a.out)
