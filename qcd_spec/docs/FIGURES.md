# Figure guide: architecture, training and results

All figures are in `docs/figs/more/` (made by `make_more_figures.py`), `docs/figs/full/` (`report_full.py`) and
`docs/figs/mog/` (per-run dashboards of the supervisor-spec runs, `plot_runs.py`). The numbers below are taken
from the result JSONs; parameter counts come from building the models in `models.py`. State on 2026-11-17: 131
finished runs (laptop + server), of which 10 used the supervisor's mixture-of-Gaussians setup.

## A. Architecture

| figure | what it shows | what to say |
|---|---|---|
| `A1_pipeline.png` | The whole system at k = 8: 297 sensor channels → encoder f → 8 numbers per step over the link → q0/q1 → CUSUM alarm; decoder g on the side. | f 86,536 parameters, g 125,609, q0 = q1 88,580 each (MoG head; the old diagonal head has 68,560). Only f runs on the aircraft; g is used for training and reconstruction, not for detection. |
| `A2_mog_head.png` | The supervisor's mixture-of-multivariate-Gaussians head. | 4 components, each with mean z_t + δ, diagonal scale and a rank-8 factor W: Σ = W Wᵀ + diag(d²). 324 outputs per step at k = 8. Low-rank because a full Cholesky at k = 200 needs 20,100 numbers per component. |
| `A3_families.png` | The four backbones (transformer, GRU, TCN, linear AR(4)) with layer lists and parameter counts. | At k = 8: encoder tf 86,552 / gru 44,568 / tcn 77,400; diagonal q tf 68,560 / gru 26,576 / tcn 59,408 / linear 272. |
| `A4_loss_terms.png` | Every loss term (D̄, L0, L1, H, J, T, L), its formula and which parameters it trains. | L = (1 − λ) D̄ + λ T + η H with η = 0.1. q0 and q1 are always fitted by maximum likelihood; the encoder cannot cheat by changing q. |
| `A5_schedule.png` | The 4 training stages with epochs, and how λ ramps in. | 10 / 10 / 15 / 10 epochs, 70 iterations each, batch 8 normal + 8 abnormal, Adam lr 1e-3, clip 1.0. |
| `docs/figs/03_pipelines.png`, `04_model_families.png`, `05_training_schedule.png` | Earlier overview versions (raw vs compressed pipeline). | |

## B. Results across every variation

| figure | what it shows |
|---|---|
| `full/F1_main_lambda.png` | Detection, delay, false alarms, D̄ against λ for each family at k = 8 (3 seeds). |
| `full/F2_code_size.png` | The same against code size k = 2 … 128. |
| `full/F3_bits.png` | Quantising the code to b bits per number. |
| `full/F4_pairing.png` | Every encoder × q-model family pairing. |
| `full/F5_decoder.png` | Conv vs MLP vs BiGRU decoder. |
| `full/F8_all_settings.png` | Every setting ranked. |
| `full/F9_mog_k8.png`, `F10_mog_code_size.png` | Supervisor spec vs the old head at k = 8, and its k sweep. |
| `more/R1_detect_per_type.png`, `R2_delay_per_type.png` | Detection % and delay for each of the 14 anomaly types (10 test flights each) for 15 key settings. Throttle faults and surface oscillation are easy; sensor drift and surface loss of effectiveness are hard for every model. The MoG λ = 0.1 setting reaches 90 % on pitot blockage and 100 % on throttle lever bias and thrust loss. |
| `more/R3_drift_curves.png` | Mean CUSUM increment after the anomaly starts, in units of the normal-flight spread. A curve far above 0 means the detector sees the change. These are means over flights, so a few flights with huge increments can lift a curve (the TF λ = 0.25 curve is high although it detects only 20 %). |
| `more/R4_R5_all_runs.png` | All 131 runs: detection vs false alarms, and detection vs reconstruction error. Hollow markers failed the sign check. The MoG runs (diamonds) sit top left in R5: better detection and better reconstruction at the same time. |
| `more/R6_sign_check.png` | The sign check on validation vs test, and the four cross-entropies of three example runs. |
| `more/R7_mog_curves_by_lambda.png` | Supervisor spec at k = 8: validation D̄, L0, J, combined loss, code scale and the largest mixture weight, epoch by epoch, for each λ and seed. |
| `more/R8_mog_curves_by_k.png` | The same for each code size k (λ = 0). |
| `more/R9_family_curves.png` | Training curves of the three neural families (old head, λ = 0.25). |
| `more/R10_family_lambda_heat.png` | Family × λ heatmaps of detection, delay and the fraction of valid seeds. |
| `more/R11_cost.png` | Parameters per component and wall-clock minutes per run. GRU about 7 min, TCN about 8 min, transformer 13–27 min on the laptop; MoG runs 52–106 min on the shared server GPU. |
| `more/R12_threshold.png` | The CUSUM threshold h grows with k, and the two calibrations give similar false-alarm rates. |
| `mog/*_losses.png` | For each supervisor-spec run: train (solid) and validation (dashed) for D̄, L0, L1, H, J, T, combined loss, code scale and largest mixture weight. Train and validation track each other, so there is no overfitting. |
| `runs/*` | The same dashboards for every earlier run. |

## C. What the checks found (say these in the talk)

1. **The MoG setup is the best compressed model so far.** At k = 8, λ = 0.1 it detects 55 ± 5 % (2 seeds) at 32 s
   median delay and 5.4 % false alarms. The best old-head setting at k = 8 with all seeds valid reaches 23 ± 6 % (TCN, λ = 0.25). The uncompressed
   transformer detects 52 %.
2. **But the mixture mostly collapses to one component.** The largest mixture weight ends at 0.99–1.00 in 8 of
   10 runs (0.63 in only one). So the gain comes mainly from the correlated (low-rank) covariance and/or the
   linear encoder output, not from multi-modality.
3. **The linear encoder output lets the code scale drift** (R7, code-scale panel). At λ = 0 the η·H term shrinks
   the codes (std 2.3 → 0.3), because smaller codes get a higher likelihood for free. At λ > 0 the task term
   blows them up (λ = 0.1: 12–15; λ = 0.25: 96). Two consequences:
   - L0, L1, H and T values are **not comparable between runs** with different code scales.
   - At λ = 0.25 the scale blow-up goes with worse results (19 % detected, validation sign check failed).
   The LayerNorm-output MoG runs (queued) will separate the effect of the head from the effect of the output.
4. **The sign check is fragile when E0 is near 0.** In 14 of 131 runs, validation and test disagree on validity. The MoG
   runs have E0 between −0.01 and −0.28 at k = 8, so they are only just valid.
5. **Most numbers are 1–3 seeds.** The MoG k sweep is 1 seed per point; seeds 1–2 and the remaining server queue
   are still running.
6. **False alarms scatter around 5 %.** The threshold is set on 140 validation normals and checked on 140 test
   normals; one flight is 0.7 %, and the binomial standard deviation at 5 % is 1.8 %, so values from about 3 % to 7 % are consistent with the 5 % target.
