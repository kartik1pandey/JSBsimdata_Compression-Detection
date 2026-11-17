# How to explain it: figure by figure, model by model, step by step

Use this sheet when presenting. Each figure has:
- **Show**: what is on it;
- **Say**: the sentence to say;
- **If asked**: likely questions and answers.

Numbers come from `results/quick/` (seed 0, shortened schedule) unless stated.

---

## Part 1. The story in 30 seconds

> "A plane streams about 300 sensor channels four times a second. We want to detect a fault as soon as it starts, but we can only keep a compressed version of the stream. The supervisor's spec asks whether we can train the compressor to keep the *evidence of a change*, not just reconstruct well. I built a new JSBSim dataset exactly as the spec requires: normal flights, flights abnormal from the first sample, and change flights. Then I implemented the spec's loss, training schedule, calibration and scoring, and compared transformer, GRU, TCN and linear models. Over three seeds, a plain autoencoder catches about 12% of faults in about 55 seconds. With a small task weight (λ = 0.25), the TCN catches 23% in a median of 6 seconds, on every seed, while still passing the spec's validity check and keeping false alarms under 5%. Too much task weight backfires: the encoder starts making normal flights look abnormal."

---

## Part 2. Figures

### Figure 1: `01_trajectory_kinds.png` (three kinds of trajectory)
- **Show:** three bars, each 300 s long.
  - Blue: normal throughout.
  - Orange: abnormal from the first sample.
  - Change: blue, then orange from a change point ν.
- **Say:** "Training needs pure examples of each class. Normal trains the normal model q0; pure-abnormal trains the abnormal model q1. Change trajectories are only for testing how fast we detect."
- **If asked why pure abnormal:** the spec's q1 must be a clean maximum-likelihood fit of abnormal behaviour. The old dataset only had change samples, so q1 also learned normal behaviour, which shrinks the evidence. That was one diagnosed cause of the earlier poor task-aware results.
- **If asked about ν:** it is uniform between 25 s and 200 s, so there are at least 10 s of burn-in before it and at least 100 s after it to detect, otherwise delays would be cut off.

### Figure 2: `02_real_examples.png` (real data)
- **Show:** the same channel (rudder position magnitude) in three real trajectories with a throttle thrust loss.
  - Normal: just noise around a level.
  - Pure abnormal: it rises over about 30 s and settles at a new level.
  - Change: flat, then at ν it rises and settles.
- **Say:** "This is closed-loop behaviour. When one engine loses thrust, the yaw damper and autopilot react, so the fault shows up in the rudder, not only in the engine. The detector has to pick up these secondary effects."
- **If asked why the rise takes time:** the autopilot and actuator dynamics respond over tens of seconds. That is also why a detector needs time to collect evidence.

### Figure 3: `03_pipelines.png` (the three pipelines)
- **Show:** three rows.
  - Raw: x goes straight to q0, q1 and CUSUM.
  - Unaware AE: x → encoder → 8 numbers → q0, q1 → CUSUM.
  - Task-aware: the same, but the encoder's loss includes the detection term.
- **Say:** "The detector is identical in all three. Only what it sees changes: all 297 channels, a code trained to reconstruct, or a code trained to reconstruct *and* keep the evidence. The decoder is grey because it is never used for detection, only to measure distortion."

### Figure 4: `04_model_families.png` (what each model can look at)
- **Show:** four panels.
  - **Transformer:** a diagonal band. Each step attends to the last m steps; in the runs m = 128 steps = 32 s.
  - **GRU:** a full triangle. It summarizes the whole history in a hidden state.
  - **TCN:** a tree of dilated convolutions, dilations 1, 2, 4, … 64, so it sees exactly 128 steps.
  - **Linear AR(4):** a straight line from the last 4 steps. This is the classical baseline.
- **Say:** "All four do the same job: predict the next step as a Gaussian, a mean and a spread. They differ in how much past they use and how flexible they are. The spec asks for a transformer with a finite context. I added the others to see whether the conclusions depend on the architecture."
- **If asked why finite context:** right after a change, the context is still mostly normal, so q1 (trained on all-abnormal contexts) is confused. A 32 s window bounds how long that confusion lasts (spec slide 17).

### Figure 5: `05_training_schedule.png` (training stages)
- **Show:** four coloured stages, and λ ramping up in stage 2.
- **Say:**
  - "Stage 0 teaches the encoder and decoder to reconstruct.
  - Stage 1 freezes the encoder and fits the two density models.
  - Stage 2 alternates three density steps with one encoder step, and slowly turns on the detection term λ. The densities must keep up with the changing code, otherwise the drift estimate is stale.
  - Stage 3 refits the densities with a decaying learning rate so they settle."
- **If asked why stage 3:** in the first attempt, the final pre-change drift jumped between −4.4 and +2.9 nats/step from one epoch to the next. The answer depended on when training stopped. A decaying learning rate fixes that, and it is applied identically to every run.

### Figure 6: `06_cusum_example.png` (CUSUM on one real trajectory)
- **Show:**
  - Top: the per-step evidence s_t on a log scale. It is noisy around 0 before the change and jumps to about +300 nats/step after it.
  - Bottom: the running CUSUM W_t, flat at 0 before the change, then climbing in a straight line until it crosses h. The alarm comes 61 s after the change.
- **Say:** "Each step votes 'abnormal' or 'normal' through the log-likelihood ratio. CUSUM adds the votes but never goes below zero, so it forgets quiet periods. After the change it climbs at the drift rate. Delay is roughly threshold divided by drift."
- **Key insight:** the drift is enormous here, yet the delay is still a minute, because h is about 66,000. That is the main finding (Figure 7).

### Figure 7: `07_calibration.png` (how the threshold is chosen)
- **Show:** for each of 140 normal validation flights, the highest CUSUM value it ever reached, sorted. The orange dot is the 134th value; that is h.
- **Say:** "No labels are needed. We pick h so that only about 5% of normal flights would ever raise an alarm. Rank 134 of 140 gives that guarantee without assuming any distribution."
- **Key insight:** the curve spans 0 to 100,000 nats. Some perfectly normal flights produce huge CUSUM values, because with 297 channels and tight predicted spreads, one surprising-but-benign sample costs thousands of nats. Those flights set h, and h sets the delay.
- **If asked whether calibration works:** yes. The false-alarm rate on unseen normal test flights was 3.6–6.4% for a 5% target, except raw TCN (9.3%) and GRU at λ = 0.5 (9.3%).

### Figure 8: `08_frontier.png` (distortion–delay frontier, the spec's main deliverable)
- **Show:** x-axis is reconstruction error D̄; y-axis is median detection delay. One line per family, points labelled by λ. Dotted lines are the raw delay for each family.
- **Say:** "The spec's hope is that moving right (giving up a little reconstruction) moves down (faster detection). Over three seeds that is exactly what happens at a small task weight. The TCN drops from 51 s to 6 s when D̄ rises only from 0.16 to 0.18, far below the dotted raw-data line. The transformer drops from 58 s to 28 s. Then, past λ = 0.5, every curve turns back up: the code is being distorted for the wrong reason."
- **Honest caveat:** the median delay is computed only over detected flights (12–26% of them), so it carries seed noise; the TCN λ = 0.25 point is the robust one (≤ 7.5 s on every seed, valid on all three).

### Figure 9: `09_retention.png` (retention vs λ)
- **Show:** the fraction of raw detectability kept in the 8-number code, against λ.
- **Say:** "This is where the method visibly works inside the code. The plain transformer autoencoder keeps about 0.1% of the raw evidence; at λ = 0.25 the code carries about as much as raw data (retention ≈ 1.1, mean of 3 seeds). The loss term does what it was designed to do."
- **Then the catch:** "Retention is a mean, dominated by a few huge values, and it varies a lot between seeds (± 0.7). It goes back down at higher λ, where the code also starts making normal flights look abnormal. Compare detection and delay, not retention, when choosing a setting."

### Figure 10: `10_drift_curves.png` (evidence after the change)
- **Show:** average s_t against seconds since the change, on val_change flights, for raw and for transformer codes at λ = 0, 0.5 and 1.
- **Say:** "Raw evidence jumps immediately at the change and stays high. The compressed codes carry much less. The dotted line at 32 s is the transformer's context length: by then the context is fully post-change, so the curve should have levelled off."

---

## Part 3. Models in one table

| Model | Plain-English job | Input → output | Trained on | Loss |
|---|---|---|---|---|
| Encoder f | squeeze 297 numbers per step into 8, using only the past | (T, 297) → (T, 8) | normal + abnormal | (1−λ)·reconstruction + λ·(−log drift) + 0.1·modellability |
| Decoder g | rebuild the 297 channels from the 8 (offline) | (T, 8) → (T, 297) | normal + abnormal | reconstruction |
| q0 | "what does normal look like next step?" | past → Gaussian over the next step | normal only | negative log-likelihood |
| q1 | "what does abnormal look like next step?" | past → Gaussian over the next step | abnormal only | negative log-likelihood |
| CUSUM | accumulate evidence, alarm at h | s_t → alarm time | (no training) | h from 140 normal validation flights |

**The four architecture families:**

| Family | Where used | Key setting |
|---|---|---|
| **Transformer** (the spec's choice) | encoder + q0/q1 | 2 layers, width 64, 4 heads, banded attention over the last 128 steps |
| **GRU** | encoder + q0/q1 | width 64, sees the whole history |
| **TCN** | encoder + q0/q1 | 7 dilated causal convolutions, sees exactly 128 steps |
| **Linear AR(4)** | q0/q1 in raw only | a linear map from the last 4 steps; the textbook baseline |

**Three safety rules from the spec, all implemented:**
1. **Stop-gradient.** q0 and q1 never learn from the encoder's detection loss, so they stay honest densities and the false-alarm guarantee holds.
2. **Spread floor.** σ ≥ 1% of the latent spread, so a model cannot claim infinite certainty.
3. **Early alarms count as false alarms,** not as zero-delay detections. This was a bug in the earlier code.

---

## Part 4. The process, step by step

1. **Simulate** 1,960 B747 cruise flights in JSBSim with the autopilot in the loop. Inject 14 fault types (sensor faults before the autopilot, actuator faults after it). Cut one 300 s window per flight. This took about 25 min on 10 CPU workers.
2. **Normalise** with training-normal statistics, keeping 297 changing channels.
3. **Train**:
   - raw density models for each family (30 epochs);
   - compressed pipelines for each family × λ (stages 0–3).
4. **Calibrate** h on 140 normal validation flights per run.
5. **Verify** the false-alarm rate on 140 unseen normal flights.
6. **Detect** on 140 test change flights:
   - detection rate;
   - false alarms before the change;
   - mean and median delay;
   - per fault type.
7. **Diagnose**:
   - cross-entropies and sign checks;
   - pre-change drift and its margin;
   - retention;
   - drift curve.

---

## Part 5. Findings to state (and how strongly)

Numbers are means over 3 seeds (full grid, k = 8) unless marked "seed 0".

| Finding | Evidence | Strength |
|---|---|---|
| The calibrated threshold controls false alarms | 2.6–6.2% on unseen normal flights for a 5% target; raw TCN overshoots on every seed (8.6%) | solid |
| A plain autoencoder throws the evidence away | all three families at λ = 0: about 12–13% detected, 51–58 s, vs 25–61% raw | solid (3 seeds) |
| **The task term makes the TCN code detect faster than raw** | TCN λ = 0.25: 23% detected, median 6.4 s (≤ 7.5 s every seed), valid 3/3 with E0[s] ≈ −3, FA 3.8% | **solid (3 seeds)** |
| λ = 0.25 helps the transformer and GRU too, but sits on the validity edge | transformer 20% / 28 s (valid 1/3), GRU 17% / 48 s (valid 2/3) | mixed |
| Too much task weight backfires | λ ≥ 0.75 fails the sign check in all 18 runs: normal flights drift toward an alarm | solid |
| 6–8 bits per number are enough | match the float code at 192–256 bit/s if q0/q1 are refit | seed 0 |
| A bigger code alone does not help | plain autoencoder never above 20% detected for k = 2…128 | seed 0 |

**Natural next steps:**
1. Try λ = 0.1 for the transformer and GRU, to keep the gain while passing the sign check on every seed.
2. Add the spec's deferred false-alarm-side term (push E0[s] down).
3. Try a heavier-tailed head, such as the spec's mixture of Gaussians, so benign surprises cost less.
4. Repeat the best extra-experiment settings (TCN λ = 0.25 at 6–8 bits; transformer k = 64) on 3 seeds.
