# Real lens: JVAS B1938+666

`HST_7255_07_NIC_NIC1_F160W_drz.fits` — real, drizzled HST/NICMOS F160W imaging of
JVAS B1938+666, copied from Şengül, Dvorkin, Ostdiek & Tsang 2022's own public
code+data release: https://github.com/acagansengul/interlopers_with_lenstronomy
(arXiv:2112.00749, MNRAS 515, 4391). Used by `../../scripts/validate_real_lens.py`
as a real-world sanity check for Family A — see `first_results.md`'s
"Real-world sanity check" section for what was found.

Their own fitting script (`jvas_nested2.py`) additionally needs two
intermediate arrays (`bckg.npy`, `pois.npy`) that are NOT in that public repo —
only this FITS file, their notebooks, and their MCMC results are. This copy is
exactly what was in their repo, nothing added or altered.
