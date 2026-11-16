# 1. Data: what the videos are, and how every file is made

Nothing is downloaded except the language model itself. Every video is **regenerated on your machine from fixed random seeds**, and the result is bit-identical to ours. `run.py data` checks this at the end by comparing fingerprints of the first videos of every split with [`reference/data_fingerprints.json`](../reference/data_fingerprints.json).

The steps below are in the order `run.py` runs them. Each one has a plain version and a technical version.

---

## Step 1: Ordinary videos (`train3`, `eval3`)

**Plain.**
- A small 3D renderer draws a floor, three coloured balls rolling on it, and a camera that drives forward and slowly turns. It takes one picture per second for 8 seconds, so every video has 8 frames.
- Each video comes with questions about two of the balls, for example "Is the red ball left or right of the blue ball?" or "Which is closer to the camera?".
- Some questions ask about the moment of the last frame. Others ask about 1, 2 or 3 seconds later, when the balls have moved on.
- These ordinary videos teach the model the basics: reading the pictures and answering the questions.

**Technical.**
- Generator: `scenes.sample_scene_multi(seed)` in `src/scenes.py`, which renders 224×224 frames at t = 0…7.
- Seeds: `40,000,000 + i` for train3 and `50,000,000 + i` for eval3 (in `src/sts.py`, `_gen_one`).
- Stored as `workspace/sts_data/<split>/<i:05d>.npz`:
  - `frames`: uint8, shape (8, 224, 224, 3);
  - `meta`: JSON with the questions, labels, ball names and hidden-at-the-end flags.
- Size: 3000 train3 videos and 300 eval3 videos.
- The command `python src/gen_tracks.py 3000` replays the same seeds. It saves the exact positions of the two queried balls in every frame to `sts_data/{train3,eval3}_tracks.json`.

## Step 2: Exit twins (`train8`, `eval8`)

**Plain.**
- Each twin is a pair of videos built so that the last frames are exactly the same, while one ball moved differently before it left the view.
- One of the asked-about balls rolls out of the camera's view before the last frame.
- In twin 0 the ball has some speed. In twin 1 only that ball's speed is changed, and the place where it was last seen is kept the same.
- A model that only looks at the last frame must give the same answer for both twins, so it can never get both right. Getting both right requires remembering how the ball was moving.

**Technical.**
- Generator: `scenes.sample_exit_twin`.
- Pair `i` is written as videos `2i` (twin 0) and `2i+1` (twin 1).
- Seeds are in the 170M and 180M ranges.
- Size: 600 pairs (1200 videos) in train8 and 300 pairs (600 videos) in eval8.
- Generation is slow because it uses rejection sampling, about 8 s per pair on one core; `run.py` uses all cores.

## Step 3: Motion-mode twins (`train9`, `eval9`, `eval9s`, `eval10` … `eval13`)

**Plain.** These are the twins every main result is scored on.
- In **twin 0** the ball that leaves the view does one of three things before leaving:
  - **stops** and stays put;
  - is **picked up and moved** to a new spot and stays there;
  - **turns** by 60–150° and keeps rolling.
- In **twin 1** the same ball rolls straight at a constant speed through the same last-seen spot, at the same moment.
- After that moment the two videos are pixel-identical. Only the frames before the ball left can tell the twins apart.

All of these splits are made by the same generator. Each new split simply uses new random seeds:

| Split | Used for |
|---|---|
| `train9` | Tuning the calculator's rules and the look-choosing rules |
| `eval9` | The A → A+B → A+B+C ladder |
| `eval9s` | A strict copy of eval9 (below) |
| `eval10` | Versions 1–3 (extra looks) |
| `eval11` | The cheaper look (Version 3c) |
| `eval12` | Choosing which frames to zoom on, round 8 |
| `eval13` | Choosing which frames to zoom on, round 9 |

A new evaluation split was made every time an idea was tested for the first time. We never tuned anything on the split that decided a test.

**Technical.**
- Generator: `scenes_modes.sample_mode_twin(seed, mode, strict)` in `src/scenes_modes.py`. Run it as `python src/scenes_modes.py <split> 300`, which makes 300 pairs (600 videos).
- Mode: `i % 3` → stop, reloc, turn. Twin 1 is always constant velocity (`cv`).
- Seed bases:

  | Split | Seed base |
  |---|---|
  | train9 | 210M |
  | eval9 | 200M |
  | eval9s | 200M (same seeds as eval9, but strict) |
  | eval10 | 230M |
  | eval11 | 250M |
  | eval12 | 270M |
  | eval13 | 290M |

- Constraints on twin 0, checked on the rendered visibility (`counts_ok`):
  - the change happens at frame `tc`, with at least 2 visible frames after it, all at or before the last visible frame `k` (4 ≤ k ≤ 6);
  - at least 2 visible frames before the change, or at least 1 for `reloc`;
  - the ball is invisible from frame k+1 to frame 7.
- **Strict splits** (eval9s, eval10–eval13):
  - zero pixels of the exited ball after frame k in both twins;
  - every frame after k pixel-identical between the twins.

  The non-strict eval9 allowed a few pixels of leakage (36 of its 300 pairs).
- Metadata per video: `mode`, `change_t`, `k_last`, `exit_slot`, `twin`, `pair`, and `tracks`: the exact positions of the two queried balls in the t = 7 camera axes, used only for scoring and side tests.

**How a result is scored (twin accuracy).**
- Only questions whose right answer differs between the two twins are kept.
- A question counts as correct only if both twins are answered correctly.
- Random guessing scores 25%. A model that ignores how the ball moved before leaving scores 0%.
- 95% ranges come from resampling whole twin pairs (bootstrap, 1000 or 2000 draws, seed 0).
- Code:
  - `m_pairacc.by_key` / `m_pairacc.stats` for model runs;
  - `state_rules.summarize` for the calculator;
  - `look_choice_r8._boot_seeds` for averages over the 3 trained heads.

## Step 4: Feature cache (`sts_data/vcache/`)

**Plain.**
- Qwen's image encoder turns each frame into 64 "tokens", which are lists of numbers describing small patches of the picture. This encoder is never trained here.
- Running it is the slowest part of reading a video, so its output for every frame is saved once and reused by every later step.

**Technical.**
- Command: `python src/precache.py split:n ...`.
- It calls `sts.vis_feats`: Qwen2.5-VL-3B vision tower plus merger, 224×224 → 8×8 = 64 merged tokens of width 2048, stored in bf16.
- Output: one file `vcache/<split>/<i:05d>.pt` per video, shape (8·64, 2048), about 2.1 MB each.
- About 9,300 videos, so about **20 GB of disk** and about 25 minutes on one RTX 5090 laptop GPU.
- It is a deterministic function of the frames, so it is never shared, only rebuilt.

## Step 5: Estimated tracks (`<split>_est_tracks.json`)

**Plain.**
- The measuring part (model structure, block 2) reads each frame's 64 tokens and says, for each of the two asked-about balls, whether it is visible and where it is on the floor.
- The camera's own motion is assumed known, as a robot knows its own wheel speed. That lets every frame's reading be moved into the coordinates of the last frame.
- The result is the **estimated track**: what our system believes each ball did. Everything that answers questions uses estimated tracks. Exact tracks are used only to train the measuring part and for clearly labelled side tests.

**Technical.**
- `state_estimator.py` trains the measuring part on train3 + train8 for 3000 steps. It writes `state_estimator.pt` and the eval8 and eval3 tracks.
- `est_infer.py` writes the tracks for every other split.
- Format: a list over videos of `{vis: [8][2] bool, pos: [8][2][2] (x right, z forward, metres, t = 7 camera axes), cam_speed, yaw_rate}`.
- Quality on eval8: visibility 99.9% correct, median position error about 10 cm.

## Step 6: Files produced later, by stage

| Stage | Files in `workspace/` |
|---|---|
| `model_w` | `w1k_s0.pt` (LoRA of model W), `res_w1k_local.jsonl` |
| `record_lora` | `r1_lsq_s1.pt` (LoRA that reads the record), `res_r_est.jsonl` |
| `rules` | `state_rules_results.json` (tuned parameters and scores of every calculator rule) |
| `ladder` | `res_m_*.jsonl`, `res_flow_*.jsonl` (model answers), `calc_ladder.json` (calculator alone) |
| `versions` | `crop_runs_r{5,6,7}/` (zoomed features, heads, refined tracks), `crop_ablation_r{5,6,7}.json` |
| `look_choice` | `look_runs_r8/`, `look_runs_r9/` (zoomed readings of every sighting), `look_choice_r{8,9}.json` |
| `report` | `my_results.md`, `my_results.json` |
| every stage | `logs/<stage>/<step>.log` |

## Disk and time, all splits

| Item | Size | Time |
|---|---|---|
| Videos (11 splits, 9,300 videos) | about 0.5 GB | about 45 min CPU (exit twins are the slow part) |
| Feature cache | about 20 GB | about 25 min GPU |
| Zoomed features, round 5 (round 6 hard-links them) | about 14 GB | included in the round times |
| Zoomed features, round 7 | about 15 GB | included in the round times |
| Language model download | about 7.5 GB | once |
