# Task-aware compression for quickest change detection: visual guide

This guide implements the spec in `cruise_lab/task_aware_compression.pdf` on a new JSBSim B747 dataset with **pure abnormal** trajectories. It covers:
- what was built;
- every model and where it is used;
- how training, calibration and scoring work;
- how long things take.

Results are in `results/quick/REPORT_TABLES.md` (the time-boxed pass) and `results/REPORT_TABLES.md` (the full grid, still to run).

---

## 1. The dataset: `qcd_pure` (new)

![three kinds of trajectory](figs/01_trajectory_kinds.png)

The spec (slides 6–8) needs **pure** classes:
- **D0** is normal from start to end.
- **D1** is abnormal from the **first sample** (`t = 1`).
- **Change** trajectories have a normal prefix and the anomaly from a change point ν to the end. They are used only to measure detection delay.

The old `qcd_data` had no pure-abnormal samples and used T = 480. This new set follows the spec.

| Item | Value |
|---|---|
| Builder | `cruise_lab/lab/qcd_pure.py` (reuses the simulator; `qcd_dataset.py` untouched) |
| Location | `C:\Users\karti\IISc_data\qcd_pure` (4.9 GB; the D: drive is full) |
| Trajectory | T = 1200 samples = 300 s at 4 Hz, flight time 720–1020 s, one per flight |
| Features | D = 297 (291 changing channels, 7 angles as sin/cos). Clock/position bookkeeping and 1 simulator artefact (`moments/n-prop-lbsft`, exactly 0 with four identical engines) are dropped |
| Normal (D0) | 560 train / 140 val / 140 test |
| Pure abnormal (D1) | 560 / 140 / 140: 40 / 10 / 10 per anomaly type, 14 types |
| Change | 140 val_change + 140 test_change (10 per type); ν uniform in samples 100–800 (25–200 s), ≥ 100 s runway after |
| Normalisation | z-score with train_normal mean/std (std floored at 0.1 × all-train std) |
| Flights | 1,960 total; 6.4 s wall time per flight on average; about 25 min on 10 CPU workers |
| Envelope | 21 flights got a milder anomaly redraw; 16 had their scenario re-seeded because one fault (e.g. a pitot blockage in heavy turbulence) left the flight envelope at every severity. All of this is recorded in `flights.csv` and the cache metadata |

One real channel looks like this in the three kinds of trajectory (throttle thrust loss):

![real examples](figs/02_real_examples.png)

---

## 2. The three pipelines

![pipelines](figs/03_pipelines.png)

| Pipeline | What it is | Role |
|---|---|---|
| (i) **raw** | q0, q1 model the 297-channel stream directly | detectability "ceiling" (K_X) |
| (ii) **unaware AE** (λ = 0) | a causal encoder compresses 297 → **k = 8** numbers per step, trained only to reconstruct | the baseline to beat; this is the plain "autoencoder on top of the data" |
| (iii) **task-aware** (λ > 0) | same encoder, but its loss also rewards keeping the evidence of abnormality | the method |

Detection always works the same way:
1. Two next-step density models give a log-likelihood ratio per step, s_t = log q1(z_t | past) − log q0(z_t | past).
2. CUSUM accumulates it: W_t = max(0, W_{t−1} + s_t).
3. An alarm fires when W_t ≥ h.

The decoder g is never in the detection path. It only measures how much reconstruction quality (D̄) the code gives up.

---

## 3. Every model and how it is used

![model families](figs/04_model_families.png)

| Block | Families trained | Shape | Causal? | Trained on | Used for |
|---|---|---|---|---|---|
| **Encoder f** | causal **transformer** (2 layers, d = 64, 4 heads, attends to the last 128 steps) · **GRU** (d = 64) · **TCN** (7 dilated causal conv layers, receptive field 128) | x (T, 297) → z (T, 8), LayerNorm on the output | yes (required) | D0 + D1, loss L_λ | produces the code the detector sees |
| **Decoder g** | the same for all: 3 non-causal 1-D conv layers (kernel 5, width 128) | z (T, 8) → x̂ (T, 297) | no | D0 + D1, loss D̄ | measures distortion D̄ only |
| **q0** (normal model) | transformer (spec choice, context m = 128 = 32 s) · GRU (whole history) · TCN (receptive field 128) · **linear AR(4)** (raw only, classical VAR baseline) | predicts the next step as a Gaussian (μ, σ) per dimension; μ = z_t + correction | yes | **D0 only**, plain maximum likelihood | log q0 in the LLR |
| **q1** (abnormal model) | same family as q0 | same | yes | **D1 only**, plain maximum likelihood | log q1 in the LLR |

Pairing in the experiments: encoder family = q family (transformer + transformer, GRU + GRU, TCN + TCN). Raw runs all four q families.

**Guards from spec slide 16:**
- σ ≥ σ_min = 0.01 × the latent std, so no density can collapse to a spike.
- LayerNorm on the encoder output.
- ε = 10⁻³ inside the log.

---

## 4. The loss (spec slides 12–15)

For the encoder only:

**L_λ = (1 − λ)·D̄ + λ·(−log Ĵ^sg) + η·H̄,  η = 0.1**

| Term | Formula | Meaning |
|---|---|---|
| D̄ | ½ Σ_class mean ‖X − g(f(X))‖² / (D·T) / σ²_X | reconstruction error, normalised by the data variance |
| Ĵ^sg | mean over D1 trajectories and t > t0 of [log q1(z_t) − log q0(z_t)], with **stop-gradient** on q0's and q1's parameters | post-change drift in nats/step; delay ≈ h / Ĵ, so −log Ĵ ≈ log(delay) |
| H̄ | (L0 + L1) / (k·T) | keeps the code something the density models can represent |

- **q0 and q1 never see the encoder's loss.** They only do maximum likelihood on their own class. This keeps q0 a calibrated density, which the false-alarm guarantee depends on.
- **The −log Ĵ term uses an EMA denominator**: the gradient is −∇Ĵ_t / Ĵ_ema, so a small batch does not explode the log.

![training schedule](figs/05_training_schedule.png)

| Stage | What trains | Why |
|---|---|---|
| 0 | f, g on D̄ | warm-start a sensible code |
| 1 | q0, q1 (f frozen) | fit both densities; check Ĵ > 0 |
| 2 | 3 steps on (q0, q1) : 1 step on (f, g) with L_λ; λ ramps over 5 epochs | the densities must catch up after every code change |
| 3 *(added)* | q0, q1 refit (f frozen), learning rate cosine-decayed to 0 | see below |

**Why stage 3 was added (a measured problem).** In the first attempt, the final E0[s] of the unaware transformer jumped between −4.4 and +2.9 nats/step from one epoch to the next. The LLR is the difference of two noisy fits, so it depends on when you stop. Decaying the learning rate settles both fits. The same rule is applied to every pipeline (raw: stage 1, lr → 0). The first-attempt results are kept in `results/_superseded_v1/`.

---

## 5. Calibration and scoring (slides 20–24)

![calibration](figs/07_calibration.png)

**Calibration** uses normal validation trajectories only (N = 140):
1. Run CUSUM on each trajectory and take its maximum M_i.
2. Set h = M_(r), the r-th smallest, with r = ⌈(N + 1)(1 − α)⌉ = 134 for α = 0.05.
3. This guarantees P(false alarm) ≤ 5 % without any distribution assumption.
4. h is **recalibrated for every run**, then checked on test_normal.

![CUSUM example](figs/06_cusum_example.png)

**Scoring** on test_change uses three outcomes:

| Outcome | Rule |
|---|---|
| False alarm | alarm before ν |
| Detection | alarm at or after ν; the delay counts |
| Miss | no alarm |

We report the detection rate (DR), the pre-change false-alarm rate (FA_pre), and the mean and **median** delay in seconds. The LLR at array index j scores sample j + 1, and the stopping time uses that offset.

**What the example shows.** After the change, s_t jumps to about +300 nats/step and W_t climbs steadily. But h is very large (about 6.6 × 10⁴), because some *normal* trajectories also produce huge LLR spikes. Across 297 channels with tight σ, one benign surprise can cost thousands of nats. That is why the raw detector takes about a minute to fire even on an obvious fault.

---

## 6. Results (full grid: 3 seeds, full schedule, k = 8)

Mean over seeds 0, 1, 2. Delays are in seconds; "detected" is the share of the 140 test change flights detected
after the change. FA is the false-alarm rate on unseen normal flights (target 5 %). "Signs ok" is the spec's
validity check (CE10 > CE11 and CE01 > CE00, i.e. E0[s] < 0 and E1[s] > 0): a run that fails it is not
interpretable. Full numbers with ± std: `results/REPORT_TABLES.md`. Figures 08–10 in `docs/figs/`.

**Raw pipeline** (no compression; the ceiling):

| q family | signs ok | FA normal | detected | median delay |
|---|---|---|---|---|
| Transformer | 2/3 | 2.6 % | **52 %** | 49 s |
| GRU | 3/3 | 3.6 % | 25 % | 44 s |
| TCN | 3/3 | **8.6 %** (above target) | **61 %** | 44 s |
| Linear AR(4) | 3/3 | 4.3 % | 19 % | 60 s |

**Compressed pipelines** (297 → 8 numbers per step):

| Family | λ | D̄ normal | signs ok | E0[s] | FA normal | detected | median delay |
|---|---|---|---|---|---|---|---|
| Transformer | 0 (plain AE) | 0.15 | 3/3 | −0.34 | 4.5 % | 12 % | 58 s |
| Transformer | **0.25** | 0.19 | **1/3** | +0.07 | 4.3 % | **20 %** | **28 s** |
| Transformer | 0.5 | 0.23 | 1/3 | +0.90 | 4.5 % | 10 % | 71 s |
| Transformer | 0.75 / 1 | 0.25 / 0.32 | 0/3 | > 0 | 5 % / 4 % | 10 % / 7 % | 49 s / 95 s |
| GRU | 0 (plain AE) | 0.18 | 3/3 | −1.11 | 4.5 % | 13 % | 54 s |
| GRU | **0.25** | 0.18 | **2/3** | −0.31 | 6.2 % | **17 %** | **48 s** |
| GRU | 0.5 / 0.75 / 1 | 0.24 / 0.29 / 0.33 | 0/3 | > 0 | 3–6 % | 18 / 9 / 4 % | 71 / 92 / 88 s |
| TCN | 0 (plain AE) | 0.16 | 3/3 | −0.44 | 3.3 % | 12 % | 51 s |
| **TCN** | **0.25** | **0.18** | **3/3** | **−2.87** | **3.8 %** | **23 %** | **6.4 s** |
| TCN | 0.5 | 0.24 | 2/3 | −0.11 | 3.1 % | 26 % | 12 s |
| TCN | 0.75 / 1 | 0.30 / 0.38 | 0/3 | > 0 | 5 % / 4.5 % | 17 % / 12 % | 77 s / 102 s |

**What it means:**
1. **A plain autoencoder throws the evidence away.** All three families at λ = 0 detect about 12–13 % of faults
   in about 50–58 s, against 25–61 % on raw data. It is trained to rebuild the large signals, and fault signatures are small.
2. **The task term helps — at a small weight.** λ = 0.25 is the only weight that improves every family:
   about 2× the detection rate and a shorter delay than the plain autoencoder.
3. **The TCN at λ = 0.25 is the clear result.** It is valid on all 3 seeds with margin (E0[s] ≈ −3). It detects
   23 % of faults with a median delay of **6.4 s** (≤ 7.5 s on every seed), against 51 s for the plain TCN
   autoencoder and 44 s for raw TCN, while D̄ rises only 0.16 → 0.18 and false alarms stay at 3.8 %.
   Compression with the task term detects *faster* than the uncompressed model, on fewer faults.
   The frontier plot (fig. 08) shows it: the TCN λ = 0.25 and 0.5 points sit far below the raw-data lines.
4. **Too much weight breaks validity.** λ ≥ 0.75 fails the sign check in every family and seed: the encoder
   separates the classes by making *normal* flights look abnormal (E0[s] > 0). λ = 0.5 fails in 4 of 9 runs.
5. **The transformer and GRU sit on the edge at λ = 0.25** (valid on 1/3 and 2/3 seeds; E0[s] near 0). Their
   gain is real on average but not reliably valid. A smaller λ (e.g. 0.1) is the natural next test.
6. **The false-alarm side is still the bottleneck.** Rare spikes on normal flights set h; nothing in the loss pushes
   E0[s] down (the spec's deferred term). Raw TCN overshoots the 5 % target on every seed (7.9–9.3 %).

**Extra experiments (seed 0)** — code size k = 2…128, bits per number, encoder × predictor pairing, decoder:
see `results/ext/EXT_TABLES.md`, `docs/figs/ext_*.png` and `docs/MONITORING.md`. Headlines:
- Code size: a bigger plain autoencoder reconstructs better but detects no better (≤ 20 %); with λ = 0.25 the best
  valid points are transformer k = 64 (31 %, 27 s) and GRU k = 32 (24 %, 52 s). Past about 64 numbers q1 overfits
  and the sign check starts to fail even at λ = 0.
- Bits: 6–8 bits per number (192–256 bit/s, 150–200× less than raw) match the float code, provided q0/q1 are
  refit on the quantised code; below about 6 bits the delay rises to 90–150 s.
- Pairing (encoder ≠ q0/q1 family, k = 8): **the predictor family matters more than the encoder.** TCN predictors
  give the best valid result for both encoders (transformer encoder + TCN q: 20 % detected, median 6.4 s — the same
  speed as the full TCN pipeline; GRU encoder + TCN q: 21 %, 36 s). A linear AR(4) predictor cannot use the code at
  all (K ≈ 0, 6–11 % detected). GRU encoder + transformer q detects 36 % but fails the sign check (E0[s] = +1.7).
- Decoder (conv vs per-step MLP vs bidirectional GRU, k = 8): no consistent effect. With λ = 0 the strongest
  decoder (bidirectional GRU) gives the best reconstruction but the lowest detection (8–11 %). With λ = 0.25 the best
  decoder differs by encoder (GRU: MLP 21 %; transformer: bidirectional GRU 21 % at 9 s, while MLP fails the sign
  check). Decoder choice matters less than λ and the predictor family.

The quick pass (one seed, shortened schedule) that preceded this is kept in `results/quick/`.

---

## 7. How long things take

| Job | Time |
|---|---|
| Dataset build (1,960 flights, 10 workers) | about 25 min (6.4 s per flight) |
| Raw run (30 epochs) | 1.5–7 min (linear < GRU < TCN < transformer) |
| Compressed run, full schedule (10/10/15/10 epochs) | transformer about 13 min, GRU about 5 min, TCN about 6 min |
| Compressed run, quick schedule (5/5/8/6) | transformer about 7 min, GRU/TCN 2–3 min (slower when 3 run in parallel) |
| **Quick pass** (seed 0, λ ∈ {0, 0.5, 1}, 3 families) | **about 30 min** (done now) |
| **Full grid** (3 seeds × {4 raw + 3 families × 5 λ}) | **about 6.5 h** (`bash run_all.sh`; still to do) |

---

## 8. Files

| File | What |
|---|---|
| `cruise_lab/lab/qcd_pure.py` | dataset builder |
| `qcd_spec/models.py` | encoders, decoder, AR Gaussian density models |
| `qcd_spec/spec_run.py` | one experiment: train → calibrate → verify → detect → metrics |
| `qcd_spec/run_all.sh` | full grid (later) · `quick.sh`: the time-boxed pass |
| `qcd_spec/report.py` | tables + frontier / retention / drift figures (`QCD_RES=results/quick` for the quick pass) |
| `qcd_spec/make_guide_figures.py` | the figures in this guide |
| `qcd_spec/results/` | per-run JSON (every metric + training log), `runs.csv`, logs |
| `C:\Users\karti\IISc_data\qcd_spec_ckpt` | checkpoints and per-step LLRs |

## 9. Deviations from the spec (all deliberate)

| Spec | Here | Why |
|---|---|---|
| D = 100 (run #1 table) | D = 297 | keep every changing channel, as with the earlier dataset |
| Spliced validation streams for the drift curve | simulated val_change trajectories | splicing two different flights creates a jump at the splice; simulated changes don't |
| No stage 3 | stage 3: refit q0, q1 with lr → 0 | the final LLR was otherwise a function of the stopping epoch (§4) |
| Run the supervisor's original code | not run on this data | her loader reads whole splits into RAM (about 1.7 GB); about 1.3 GB is free |
