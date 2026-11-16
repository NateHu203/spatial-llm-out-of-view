# 4. Replication results: what the project shows, in three tables

**One sentence.** A small video-language model cannot say where a ball is after it has rolled out of view. Adding a written record outside the model, a calculator with a change-aware rule, and zoomed looks at the right frames lifts its twin accuracy from **5%** to about **75%**.

## How to read every number

- **Twin accuracy.** Each test uses pairs of videos whose last frames are identical. In one video the ball that left the view kept rolling straight; in the other it stopped, was moved, or turned before leaving.
  - We keep only questions whose right answer differs between the two videos.
  - A question counts as correct only if **both** answers are right.
  - Random guessing scores 25%. A model that only looks at the last frame scores 0%.
- **[low, high]** is the 95% range from resampling whole twin pairs (bootstrap). Two numbers whose difference has a range that excludes 0 are reliably different.
- **"Current system"** means fixes A + B + C on the estimated positions, with no zoomed looks.
- Every test split was made fresh and never used for tuning (see [01_DATA.md](01_DATA.md)).

The reference values are in [`reference/expected_results.json`](../reference/expected_results.json). `python run.py report` prints yours next to them.

---

## Table 1: Fixes A, B, C (eval9, 695 question pairs)

| Step | What the model gets | Model twin accuracy |
|---|---|---|
| Model only | All 8 frames, no record (model W) | **5.3** [2.7, 8.4] |
| + A | Last frame + a sentence from the record (straight-line calculator) | **51.9** [46.0, 57.8] |
| + B | … + each ball's distance in the sentence | **53.7** [47.6, 59.5] |
| A + C | Change-aware rule, no distances | 63.0 [57.2, 68.6] |
| **A + B + C** | Change-aware rule + distances = **current system** | **66.8** [61.0, 72.0] |
| Side test | A + B + C with exact positions | 97.0 [95.2, 98.5] |

Calculator answering directly, with no model:

| Rule | Estimated positions | Exact positions |
|---|---|---|
| Straight line | 53.5 [47.4, 59.4] | 66.6 |
| IMM (usual tracking method) | 66.0 [60.9, 71.5] | 89.9 |
| **Change-aware (C)** | **66.8** [61.1, 72.0] | **97.8** [96.1, 99.2] |

**Plain reading.**
- The model alone ignores how the ball moved (5%).
- With the record it reaches 52–54%. Adding the change-aware rule brings it to 67%, the same as the calculator answering by itself. From then on, the model is no longer the bottleneck.
- With exact positions the system would reach 97–98%, so the remaining loss comes from **measuring** positions.

**Technical.**
- Model runs: `sts.py run --n_train 0` with LoRA `r1_lsq_s1.pt`, `--text_state --hist last1`, on eval9 with estimated tracks. `m_w` uses LoRA W instead.
- Scored by `m_pairacc` (1000 draws).
- Calculator: `calc_ladder.py`, using `state_rules_results.json` parameters.

## Table 2: An extra look at the ball (3 trained heads each)

Fresh twins, rule C, mean of 3 trained heads.

| Version | Extra picture (tokens) | eval10 | eval11 |
|---|---|---|---|
| Current system | none | 64.8 | 62.5 [56.9, 67.9] |
| Version 1 | none: frame tokens only | 61.8 [56.6, 66.6] | – |
| Version 1h | none, with Version 3's head start | 64.6 [59.6, 69.4] | 61.2 [56.6, 65.8] |
| Version 2 | whole frame at 448 (256) | 71.4 [66.8, 75.4] | – |
| Version 2h | whole frame, with the head start (256) | 70.7 [66.2, 74.6] | 66.8 [62.3, 71.0] |
| **Version 3** | 96 px square around the ball at 448 (256) | **74.5** [69.9, 78.2] | **70.0** [65.5, 74.3] |
| **Version 3c** | same square at 224 (**64**, 4× fewer) | – | **71.5** [67.3, 75.6] |

| Comparison | Change | Verdict |
|---|---|---|
| Version 2 − Version 1 (eval10) | +9.6 [+5.6, +13.7] | An extra look helps |
| Version 3 − Version 2 (eval10) | +3.1 [−0.3, +6.5] | Aiming it at the ball: small, not significant |
| Version 3 − Version 2h (eval10) | +3.8 [+0.4, +7.2] | Not explained by the head start |
| Version 3c − Version 3 (eval11) | +1.4 [−2.0, +5.0] | The cheap look is as good |
| Version 3c − current system (eval11) | +8.9 [+3.6, +14.2] | The cheap look beats the current system |

**Plain reading.**
- A second, closer look at the ball measures it better and adds 9–12 points.
- Zooming exactly onto the ball adds a little more than a wide look at the same cost.
- The zoom works just as well with a quarter of the tokens (Version 3c).

## Table 3: Which frames to zoom on

Version 3's zoom, 3 trained heads, rule C re-tuned per rule.

| Rule (looks per video) | eval12 | eval13 |
|---|---|---|
| No zoom (current system) (0) | 61.8 | 69.2 |
| Schedule, last 2 looks (2) | 68.7 [63.8, 73.3] | 70.9 [66.3, 75.0] |
| **Last sighting of each ball (2)** | **72.6** [68.1, 76.9] | **75.4** [71.1, 79.0] |
| Version 3's full schedule (≈6.8) | 74.6 [70.3, 78.7] | 76.4 [72.4, 79.8] |

| Comparison | eval12 | eval10 | eval13 |
|---|---|---|---|
| Last sighting − schedule, 2 looks each | +4.0 [+0.5, +7.6] | +4.2 [+1.0, +7.5] | +4.5 [+1.3, +7.8] |

**Plain reading.**
- With only two zoomed looks per video, look at **the last moment each ball was seen**. That beats spending the same two looks on the schedule's last frames, by about 4 points, on three separate sets of twins.
- Why it works: the answer is a forecast past the end of the track. An error at the end swings the forecast most; an error in the middle barely moves it.
- With 4 or more looks, where they go barely matters (round 9; [06_HISTORY.md](06_HISTORY.md)).

**Technical.**
- `look_choice_r8.py`:
  - zooms on every estimated-visible sighting (train9, eval10, eval12);
  - choosers `fixed2` (schedule) and `last_each2` (each ball's last estimated-visible frame);
  - CA re-tuned on train9 per chooser and seed (64-point grid);
  - pair bootstrap jointly over seeds (2000 draws).
- `look_choice_r9.py` adds eval13.

---

## Checked in this repository

| What | How | Result |
|---|---|---|
| Video regeneration | First 4 videos of all 11 splits regenerated by `src/` and fingerprinted | Bit-identical |
| Calculator ladder | `calc_ladder.py` on our estimated tracks and tuned rules | Identical to our log |
| Round 8 and round 9 scoring | `look_choice_r8.py score`, `look_choice_r9.py score` on our zoomed readings | Accuracies, differences, gates and tuned parameters identical |
| Rounds 5–7 scoring | `crop_ablation_r{5,6,7}.py score` on our refined tracks | Accuracies, differences, per-seed values and gates identical |
| Every GPU stage | `run.py data cache model_w estimator record_lora rules ladder versions look_choice report --quick` (RTX 5090 Laptop) | Runs end to end in 11 min (round 9 has no quick mode) |

A full retraining from scratch (about 4–5 GPU hours) is expected to land within the 95% ranges. It has not been run end to end in this repository's layout.
