# 2. Model structure, block by block, in the order we built it

Each block is something we added because the block before it failed in a specific, measured way. Every block has a plain version and a technical version, and lists the code that implements it. Block numbers match `run.py` stages and the tables in [04_RESULTS.md](04_RESULTS.md).

```
                         once per frame, while the video comes in                       once, when asked
  frame t ──► [0] Qwen image encoder ──► 64 tokens ──► [2] measuring part ──► (visible?, x, z)
   224×224        (frozen, cached)            │                │                   │
                                              │                └──► [6] zoomed look ──► refined (x, z)   (Versions 1-3)
                                              │                         chosen by [7]
                                              ▼                                             ▼
                                         (last frame only)                     written record: every ball's track
                                              │                                             │
                                              │                     [3][4][5] calculator: extend each track to the
                                              │                          asked time; write a sentence
                                              ▼                                             ▼
                                    [1] language model (W + record LoRA) reads: last frame + question + sentence ──► answer
```

---

## Block 0: The base model, Qwen2.5-VL-3B-Instruct (frozen)

**Plain.** This is a small open AI model that looks at pictures and answers in words. We give it the 8 frames of a video and a question, and read off which of two answer words it prefers, such as "left" vs "right".

**Technical.**
- Vision tower: a ViT with 675M parameters, 14×14 patches and 32 blocks (window attention, plus 4 global blocks).
- A 2×2 patch merger and an MLP (5120 → 2048) turn each 224×224 frame into 64 tokens.
- Decoder: 36 layers, width 2048. GQA attention with 16 query heads and 2 KV heads; SwiGLU MLP (2048 → 11008 → 2048); M-RoPE positions.
- The answer is read at the assistant-start position by comparing the logits of the two option words (`sts.py`, `answer`).
- Model loading: `mve_nowrope.build('native', seed)`, called by `sts.build`.

**Measured flaw (eval9).** Model W (block 1) sees all 8 frames and still gets **5.3%** twin accuracy, below random guessing (25%). It answers from the last frame, where the ball is gone.

---

## Block 1: Model W, a LoRA fine-tune on ordinary videos

**Plain.** We teach the base model the task format on ordinary videos by training a thin "correction" on top of its weights. The big pretrained weights never change.

**Technical.**
- LoRA with r = 16 and α = 32 on W_q, W_k, W_v and W_o of all 36 decoder layers. A ~ N(0, 1/2048) and B = 0 at initialisation.
- Trained for 1000 steps on train3, seed 0. Vision tower, MLPs and embeddings stay frozen.
- Command: `sts.py run --arm lora --train_split train3 --n_train 1000 --save_lora w1k_s0.pt`.
- Every later LoRA starts from W.

---

## Block 2: The measuring part (state estimator), the "S1 estimates"

**Plain.** This is a small separate network, about 2.5M weights, roughly 0.4% of the image encoder. For every frame and each asked-about ball, it reads the frame's 64 tokens plus the ball's colour and outputs two things:
- whether the ball is visible;
- where it is on the floor, in metres.

The camera's own motion is assumed known, as a robot knows its wheel speed. That lets every frame's reading be moved into the coordinates of the last frame. The resulting list of positions is the **estimated track** used by everything below.

**Technical.**
- `StateEstimator` in `src/state_estimator.py`:
  - Linear(2048 → 256) + LayerNorm, plus a learned 64-slot position embedding;
  - a 2-layer pre-norm Transformer encoder (4 heads, FFN 1024);
  - a colour-embedding query attends over the 64 tokens;
  - MLP → (visibility logit, x/3, z/6), in the frame's own camera axes.
- Training: 3000 steps on train3 + train8 from the frozen cached tokens.
- Odometry (`frame_to_t7`) maps each frame's reading to the t = 7 camera axes, using the camera's speed and yaw rate.
- Output: `<split>_est_tracks.json`.
- Quality on eval8: visibility 99.9% correct, median error 10.2 cm, 90th percentile 23.5 cm.

---

## Block 3: Fix A, a written record outside the model plus a calculator

**Plain.** The model cannot carry how a ball moved forward in time. So we keep a written record outside the model. After every frame it stores each ball's estimated position. When a question arrives, a simple calculator extends each ball's path to the asked time with one straight line, and the result is added to the question as a sentence:

> "Computed positions at the asked time, in the camera's own axes at that time (x = metres to the right, z = metres forward): the red ball at (-1.20, +3.00), the blue ball at (+1.50, +3.80)."

The model then reads only the **last frame**, the question and this sentence. Its answer time therefore does not grow with the length of the video.

**Technical.**
- Calculator: `state_rules.py`. The straight-line rule is `lsq`, a least-squares line over all sightings, read out at the fitted value at t = 7.
- `state_rules.answer` propagates the read-out to t* through the known camera arc (`transition_np`).
- Sentence: `sts.py` → `rule_text` (`--text_state --state_rule <rule>`).
- The model reading the record is a second LoRA, `r1_lsq_s1.pt`:
  - starts from W;
  - 1000 steps on train3 + train8, seed 1;
  - `--text_state --vel_fit lsq --hist last1`, where `last1` means only the last frame's 64 tokens go to the decoder.
- Result on eval9: **5.3% → 51.9%**.

---

## Block 4: Fix B, distances in the sentence

**Plain.** Given right/forward numbers, the model could not work out which ball is closer to the camera. So the calculator also writes each ball's distance:

> "Distances from the camera at that time: the red ball 3.20 m, the blue ball 4.10 m."

**Technical.**
- `sts.py --dist_text append`.
- With exact positions, the model's "which is closer?" accuracy rises from 56.7% to 67.2%, matching the calculator reading the same sentence.
- With estimated positions: 51.9% → 53.7%.

---

## Block 5: Fix C, the change-aware rule

**Plain.** One straight line through all sightings is wrong when the ball stopped, was moved, or turned before leaving. The change-aware rule watches the record frame by frame. When the latest 2–3 sightings fit their own line much better than the old path, it decides "the motion changed here" and restarts the line from that moment. If the new line barely moves, the ball is treated as standing still.

**Technical.**
- `CAState` / `r_ca` in `src/state_rules.py`: change-aware segmented least squares.
  - Bounded state: 8 running sums of the current segment plus the last 3 sightings.
  - Split test: the residual reduction from giving the last W ∈ {2, 3} sightings their own line must exceed κσ².
  - Read-out: the segment's line at t = 7; "settled" (v = 0, mean position) if |slope| < v_rest.
- Parameters are tuned on train8 + train9 by grid search (`state_rules.py` → `state_rules_results.json`).
- The usual tracking method, IMM (`r_imm`), is kept as the comparison.
- Results on eval9 (estimated positions): model A + B + C = **66.8%** (the **current system**); calculator alone 66.8%; calculator with exact positions 97.8%.

**What is left.** Splitting the error into parts showed that measuring how the ball moved from frame to frame is the main remaining loss (strict twins: 66% → 86% when only the path shape is made exact). Blocks 6 and 7 target this.

---

## Block 6: An extra, zoomed look (Versions 1, 2, 3, 3c)

**Plain.** For one ball per frame, chosen by a fixed schedule (ball t mod 2 if visible), a second small network re-measures the ball's position. The versions differ only in what extra picture it gets:

| Version | Extra picture | Extra tokens |
|---|---|---|
| **1 / 1h** | None: the frame's 64 tokens only | 0 |
| **2 / 2h** | The whole frame enlarged to 448×448 | 256 |
| **3** | A 96×96-pixel square cut around the estimated ball position, enlarged to 448×448 | 256 |
| **3c** | The same square at 224×224 | 64 (4× fewer) |

The "h" versions (1h, 2h) get the same head start as Version 3: the square's location is given to them, and they answer inside it. This makes the comparison fair. The refined position replaces the estimate for that ball in that frame, and everything downstream (blocks 3–5) is unchanged.

**Technical.**
- Head: `crop_ablation.Head`, about 2.2M weights.
  - Linear(384 → 256) + LayerNorm over the 64 base tokens and the extra tokens (each with learned position embeddings).
  - 2-layer pre-norm Transformer encoder.
  - A query = colour embedding + MLP(box/224) attends over the tokens.
  - MLP → r ∈ ℝ², mapped to frame pixels as `box_xy + side·(0.5 + r)` for the crop and the "h" arms, or `224·(0.5 + r)` otherwise.
- Variants: `crop_ablation_r6.HeadBox` (1h, 2h) and `crop_ablation_r7.HeadBox64` (3c).
- Features are projected by a fixed random orthonormal 2048 → 384 matrix (seed 0), the same for every arm.
- Geometry: the floor → pixel → floor mapping is exact pinhole geometry (`own_to_pix`, `pix_to_own`).
- Training: 4000 steps, AdamW at 3e-4, OneCycle schedule, batch 256, on train8 + train9; target = the true ball-centre pixel.
- 3 seeds per version. Scores are averaged over the seeds, with a pair bootstrap.
- Code: rounds 5, 6 and 7 = `crop_ablation_r5.py`, `_r6.py`, `_r7.py`, reusing `crop_ablation.py` (round 4).
- Results:

  | Comparison | Split | Change |
  |---|---|---|
  | Version 2 − Version 1 | eval10 | +9.6 |
  | Version 3 − Version 2 | eval10 | +3.1 |
  | Version 3 − Version 2h | eval10 | +3.8 |
  | Version 3c − Version 3 | eval11 | +1.4 (as good, with 4× fewer tokens) |
  | Version 3c − current system | eval11 | +8.9 |

---

## Block 7: Choosing which frames to zoom on

**Plain.** Version 3 zooms on a fixed schedule, about 6.8 looks per video. Instead, zoom on **the last frame in which each ball was seen**, one look per ball, so 2 looks per video.

- **Why that frame.** The answer is a forecast past the end of the track. Lay a ruler along the sightings: a wrong point at the end swings the forecast most, while a wrong point in the middle barely moves it.
- **Prior work.** Optimal extrapolation design puts measurements at the ends of the observed range (Hoel & Levine 1964).
- **What it gives.** With the same 2 looks, this beats the schedule's last 2 looks on three separate sets of twins.

**Technical.**
- Every sighting the measuring part reports as visible gets Version 3's zoomed reading. This uses the 3 trained heads of round 5, with no new training (`look_choice_r8.predict`).
- A chooser picks which readings replace the estimates. It sees only the estimates, never the answers.
  - `fixed2`: the last 2 looks of the schedule.
  - `last_each2`: each ball's last estimated-visible frame.
- Rule C is re-tuned on train9 for each chooser and seed (`look_choice_r8.score`).
- Round 9 (`look_choice_r9.py`) repeats the test on eval13.
- Results, `last_each2 − fixed2`:

  | Split | Change |
  |---|---|
  | eval12 | +4.0 [+0.5, +7.6] |
  | eval10 | +4.2 |
  | eval13 | +4.5 [+1.3, +7.8] |

  With 4 or more looks, the placement of the looks barely matters.

---

## What is NOT in this repository (tried, did not work, documented only)

See [06_HISTORY.md](06_HISTORY.md). It covers the ideas that did not work:
- rotary-position tricks (NowRoPE);
- shared-trajectory supervision;
- memory tokens and visual-stream state (Stage C arms C2 and C3);
- a reference picture against shared error;
- choosing looks by residual, weighted ends, or an adaptive agent step;
- a simulate-and-check ceiling.

They are left out on purpose, to keep this repository to the structure that holds.
