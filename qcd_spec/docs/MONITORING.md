# Monitoring dashboards and the extra experiments

## 1. How to look at training while it runs

```bash
cd cruise_lab/qcd_spec
../../.venv/Scripts/python plot_runs.py --live     # all runs, including ones still training
../../.venv/Scripts/python report_ext.py           # summary plots of the extra experiments
```

Every run writes one JSON line per epoch to `results/live/<tag>.jsonl` (`results/ext/live/` for the extra lane)
while it trains. Runs started after 13:55 on 2026-11-13 also save every iteration's losses to `<tag>_trace.npz`.
Pictures go to `docs/figs/runs/`, listed in `docs/figs/runs/INDEX.md`.

## 2. The training monitor (`<tag>_training.png`)

The x-axis is the epoch, all stages in order. The background colour shows the stage: blue = stage 0 (reconstruct), green = stage 1 (fit q0, q1), orange = stage 2 (alternate), purple = stage 3 (refit q0, q1).

| Panel | What it is | Healthy pattern | Warning sign |
|---|---|---|---|
| D̄ reconstruction error | how badly the decoder rebuilds the 297 channels | falls in stage 0; rises a little in stage 2 when λ > 0 (the price of the task term) | keeps rising in stage 2: the task term is overpowering reconstruction |
| L0 / L1 | training NLL of q0 on normal and q1 on abnormal | falls, then drops again in stage 3 as the lr decays | — |
| L0 on val_normal | q0's NLL on held-out normal flights | follows L0 | **rises while L0 falls = q0 overfits** (seen in stage 3 of transformer λ = 0.25) |
| E0[s] | average evidence on normal flights | below 0 | above 0 means normal flights drift toward an alarm (more false alarms, higher h) |
| E1[s] | average evidence on abnormal flights | above 0, large | near 0: the code no longer separates the classes |
| margin | E0[s] / spread | below 0 | — |
| J, J_ema | the task term the encoder pushes up (drift on abnormal) | rises in stage 2 | grows by orders of magnitude while E0[s] also rises: it is separating the classes by making everything look abnormal |
| T̄ = −log J | the task loss itself | falls | — |
| H̄ | modellability: NLL of the code under q0/q1 | falls (the code becomes easier to predict) | — |
| λ schedule | ramps to the target over 5 epochs | — | — |
| gradient norms | size of each update before clipping at 1.0 | settle down | sustained spikes: unstable training |
| learning rate | constant, then cosine decay to 0 in the final fit | — | — |
| σ at floor | share of predicted spreads stuck at the minimum σ_min | 0 | > 0: the model wants to be more certain than allowed (LLR spikes) |

`<tag>_iters.png` shows the same losses at every iteration. The light line is each step; the dark line is a moving average.

## 3. The evaluation sheet (`<tag>_eval.png`)

1. **Per-step evidence histogram.** s_t on normal flights (blue) and abnormal flights (orange). They should sit on opposite sides of 0. Overlap is what makes detection hard.
2. **Calibration.** The maximum CUSUM of each normal validation flight, sorted, with the threshold h (the 134th of 140). A few points far above the rest show that rare normal spikes are setting h.
3. **Outcome.** False alarms on normal flights (target 5%, dashed line), alarms before the change, detected, missed; the median delay is in the title.
4. **Per fault type.** Detection rate for each of the 14 fault types, 10 flights each.
5. **Evidence around the change.** Average s_t against seconds since the change. It should jump at 0 and stay up.
6. **Diagnostics.** Cross-entropies, K, E0[s] and E1[s], the sign check, and D̄.

## 4. The extra experiments (`run_ext.sh`, seed 0, results in `results/ext/`)

| Experiment | Question | What varies | Figure |
|---|---|---|---|
| A. Bit budget | How many bits per latent number do we really need on the link? | the trained k = 8 code quantised to b = 1, 2, 3, 4, 6, 8 bits (rate = 8 × b × 4 bit/s); q0/q1 either kept (frozen) or refit on the quantised code (refit) | `figs/ext_bits.png` |
| B. Latent size | Does a bigger or smaller code help? | k = 2, 4, 16, 32, 64, 128 (k = 8 is in the main grid), transformer and GRU, λ = 0 and 0.25 | `figs/ext_latent_k.png` |
| C. Model pairing | Does the density model have to match the encoder? | encoder transformer or GRU × q0/q1 transformer, GRU, TCN or linear | `figs/ext_pairing.png` |
| D. Decoder | Does the decoder choice change what the encoder keeps? | conv (default), per-step MLP (sees only z_t), bidirectional GRU (sees the whole code) | `figs/ext_decoder.png` |

All numbers go to `results/ext/EXT_TABLES.md`.

**Link-rate reference:** raw data is 297 × 32 bit × 4 Hz = 38,016 bit/s. A float32 k = 8 code is 1,024 bit/s. An 8 × 4-bit code is 128 bit/s.

**Quantiser details.** It is uniform, with 2^b levels per latent number. Its range is the 0.1–99.9 percentile of the training codes, and values outside it are clipped. In the refit receiver, q0/q1 are retrained for 5 epochs on the quantised code plus one step of uniform dither, so the Gaussian learns how much probability falls in each bin instead of collapsing onto the grid values. Evaluation uses the exact quantised values; the bin width cancels in the LLR.
