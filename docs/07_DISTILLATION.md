# 7. Distilling the calculator into the model: design, rules and log

*Started 2026-11-21. All runs so far on one RTX 4080 SUPER (16 GB), Windows.*

## The question

Everything that works in this project sits outside the language model (measuring part, record, calculator). Can the
model learn to do it itself, so that at test time it gets **only the 8 frames and the question** and nothing else?

- **Teacher:** the change-aware calculator (rule C) reading exact tracks. On eval9 it scores 97.8% twin accuracy.
- **Student:** Qwen2.5-VL-3B + LoRA, starting from model W. No measuring part, no calculator, no camera-motion numbers.
- **Code:** `src/distill.py` (the arms), `tools/distill_queue.py` (run the queue, show the status).

## Rules (fixed before the runs below)

1. **Same information at test time.** Every model compared gets the 8 frames and the question, nothing more. Arms that
   were given the camera's speed and yaw rate (`--odo`) are not results; they are kept out of the tables.
2. **Same data and steps.** Each method is compared with the `answer` arm: same training splits, same 3000 steps, same
   teacher answers, but trained on the answer word only. Only the difference to that arm counts as the method's effect.
3. **Three seeds.** A difference counts only with seeds 0, 1 and 2, reported as the mean and a pair bootstrap.
4. **Chance is 25%.** Going from model W's 5% to 25% only means the model stopped answering from the last frame. A
   method "tracks" only if its twin accuracy is above 25% with the 95% range excluding 25.
5. **eval9 is the development set.** It has been looked at many times. A final claim must be repeated once on a fresh
   set (eval10 to eval13).

### Pass rules for the queue of 2026-11-22

| Run | Passes if | If it fails |
|---|---|---|
| `locate` (3 seeds) | Median position error on balls visible in the last frame, eval3, mean of 3 seeds, is **0.20 m or less** (the measuring part has about 0.12 m on the same balls) | Above 0.40 m: writing positions as text does not localise; change the supervision (regression on hidden states), not the format. Between: partly; try longer training before changing the method |
| `state` minus `answer` (3 seeds) | Mean difference in twin accuracy above 0, pair bootstrap range excluding 0, **and** `state` above 25% with its range excluding 25 | Writing the state has no established effect |
| `record` (1 seed, no `--odo`) | Exploratory only: more accurate written positions than `state` (0.95 m) | Dropped, as `motion` was |

## Arms

| `--target` | What the student writes | Status |
|---|---|---|
| `answer` | the answer word (control) | seed 0 done |
| `state` | each ball's position at the asked time, then the answer | seed 0 done |
| `motion` | camera motion, each ball's position and velocity at the last frame, the state, the answer | seed 0 done; dropped |
| `record` | every sighting frame by frame, the state, the answer | only run with `--odo`; not a result |
| `locate` | diagnostic: where the two balls are in the last frame, nothing else | queued |

## Results so far (eval9, twin accuracy, one seed each)

| | Student's own answer [95% range] | Answer from the numbers it wrote | Median error of written positions |
|---|---|---|---|
| Model W (baseline) | 4.9 | – | – |
| `answer` | 22.2 [17.3, 27.2] | – | – |
| `state` | 27.5 [22.9, 32.4] | 29.9 | 0.95 m |
| `motion` | 12.5 [8.9, 16.4] | 23.5 | 1.07 m |
| Current system A + B + C (uses the measuring part and the calculator) | 64.7 | – | 0.10 m |
| Teacher | 97.8 [96.1, 99.2] | – | – |

**Reading.**
- `state` minus `answer` = +5.3 [+0.1, +10.3] (pair bootstrap, one seed): not established.
- Both are at chance. They no longer answer from the last frame, but they do not track either.
- The student reads its own numbers correctly (its answer follows them 94.7% of the time in `state`), and a calculator
  given those numbers does no better. The numbers themselves are wrong: about 1 m off, against 0.10 m for the
  measuring part. Localisation is the bottleneck, not extrapolation or reading.
- `motion`: the camera's motion is learned (0.06 m/s, 0.025 rad/s median error), the balls are not (0.67 m, 0.27 m/s),
  and the longer text makes the final answer drift from what was written (83.5% agreement).

### Localisation diagnostic (`locate`, 3000 steps, 3 seeds, 2026-11-22)

Median position error on balls visible in the last frame, written as text by the student:

| Seed | eval3 (n ≈ 221) | eval9 (n = 156) | Visibility correct |
|---|---|---|---|
| 0 | 0.291 m | 0.238 m | 99.8% |
| 1 | 0.298 m | 0.264 m | 99.7–100% |
| 2 | 0.246 m | 0.224 m | 99.5–99.9% |
| **Mean** | **0.278 m** | **0.242 m** | |
| Measuring part, same balls | 0.140 m | 0.119 m | |

**Verdict by the pass rule: between 0.20 and 0.40 m, so "partly".** The rule's next step is longer training
(`locate_long_s0`, 9000 steps) before any change of method.

**What it changes in the reading above.** The model can read a visible ball's position as text to about 0.28 m, twice
the measuring part's error but nowhere near the 1 m seen in `state` and `motion`. On eval9 most asked-about balls are
hidden in the last frame, and in `state` the error was 0.46 m for balls still visible and 0.94 m for hidden ones. So
the bottleneck is not reading one frame: it is carrying a ball across frames to where it is after it left. That is
what the per-frame `record` arm supervises directly, and what a `record` run started from the `locate` weights would
test (`record_from_locate_s0`).

**Known unfairness.** The students train on train9 (same kind of twins as eval9) for 3000 steps; model W saw train3
only, 1000 steps. The `answer` arm exists to absorb exactly this.

## Running on 16 GB without losing work

- `--mem_cap_gb` (default 14): the process may not take more GPU memory than this. Going over is an error, and a test
  batch is then split in two. Without the cap, Windows spills into system memory and a run slows down about 20 times
  (this stalled one run for 1 h 40 min on 2026-11-21).
- Training writes a rolling checkpoint every 250 steps and a loss point every 100 steps; testing saves every finished
  scene. `--resume` continues both.
- `workspace/runs.jsonl` records every run that was started, how it ended, its numbers and its peak GPU memory.
- `python tools/distill_queue.py status` shows all of it in one table.

## Log

| Date | Step | Outcome |
|---|---|---|
| Nov 21 | Baseline rebuilt on 16 GB (model W with one question per forward) | All 12 ladder numbers inside the reference ranges; current system 64.7% |
| Nov 21 | `state` and `answer`, seed 0 | 27.5% vs 22.2% |
| Nov 21 | `record` and `state` with `--odo` | Stopped: the camera numbers are extra information (rule 1). The `record --odo` weights are kept, untested |
| Nov 21–22 | `motion`, seed 0 | 12.5%; dropped |
| Nov 22 | `locate` arm, checkpoints, memory cap, run registry, queue; pass rules written | Queue started (see `distill_queue.py`) |
| Nov 22 | `locate`, seeds 0–2 | 0.278 m on eval3 (measuring part 0.140 m): "partly" by the pass rule. Peak GPU memory 10.5 GB. Added `locate_long_s0` and `record_from_locate_s0` to the end of the queue |
