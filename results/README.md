# results/ — detector evaluation outputs

Each `detector_v*/` subfolder is one full evaluation run of `../scripts/evaluate_detector.py`
against a trained checkpoint: `results.json` (every number), `results.md` (formatted),
`completeness_vs_mass.png` (RQ4), `roc_fixed60.png`, `confounder_fpr.png` (RQ2).

`detector_v0` — first pass, `checkpoints/unet_v0/model_best.pt` and `model_last.pt`
(U-Net, base width 16, trained on 8,000 `tsang_fixed60` images; see that checkpoint's
`training_log.json` for the loss curve and which epoch was selected as "best").
