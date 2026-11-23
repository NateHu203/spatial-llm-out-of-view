# Results of the runs on the 16 GB desktop (RTX 4080 SUPER), Nov 21–22 2026

Raw outputs of every run that finished on the desktop, copied from `workspace/` (which git ignores). The reading of
these numbers, the rules they were judged by and the dated log are in [docs/07_DISTILLATION.md](../../docs/07_DISTILLATION.md).

Not here: the trained weights (`d_*.pt`, 28 MB each, 13 files), the videos and the image-feature cache. They are
rebuilt by `run.py` and `tools/distill_queue.py`. Local paths in the logs were replaced by `<repo>`.

## Baseline rebuilt on the desktop

| File | What |
|---|---|
| `my_results.md`, `my_results.json` | the 12 ladder numbers next to the reference ranges (all inside) |
| `res_m_w.jsonl` | model only (W): 4.9% |
| `res_m_r_lsq.jsonl`, `res_flow_lsq_est_dist.jsonl`, `res_m_r_ca.jsonl`, `res_m_r_ca_dist.jsonl` | A 51.4, A+B 54.2, A+C 61.2, A+B+C 64.7 |
| `res_m_r_ca_orc_dist.jsonl` | A+B+C with exact positions: 97.3 |
| `res_r_est.jsonl`, `res_w1k_local.jsonl` | training-time tests of the record LoRA and of W |
| `state_rules_results.json`, `calc_ladder.json`, `state_estimator_report.json` | tuned calculator rules, calculator-only ladder, measuring part report |

## Distillation (eval9, twin accuracy; guessing gives 25%)

Each `res_d_<name>.jsonl` holds one line per tested split: the run's settings, a `summary`, and `rows` with every
question, the text the student wrote and whether it was right. `res_d_<name>_train.jsonl` is the loss curve.

| File | Arm, seed | Student's answer [95% range] | Median error of written positions |
|---|---|---|---|
| `res_d_answer.jsonl` | `answer`, 0 | 22.2 [17.3, 27.2] | – |
| `res_d_answer_s1.jsonl` | `answer`, 1 | 23.6 [18.7, 28.3] | – |
| `res_d_answer_s2.jsonl` | `answer`, 2 | 22.2 [17.2, 27.9] | – |
| `res_d_state.jsonl` | `state`, 0 | 27.5 [22.9, 32.4] | 0.96 m |
| `res_d_state_s1.jsonl` | `state`, 1 | 23.2 [18.6, 28.3] | 1.00 m |
| `res_d_state_s2.jsonl` | `state`, 2 | 22.0 [17.8, 26.9] | 0.99 m |
| `res_d_motion.jsonl` | `motion`, 0 | 12.5 [8.9, 16.4] | 1.07 m |
| `res_d_record_s0.jsonl` | `record`, 0 | 17.4 [13.4, 21.7] | 1.54 m (sightings 0.73 m) |
| `res_d_locate_s0/1/2.jsonl` | `locate` diagnostic, 0–2 | not a question-answering run | eval3 0.291 / 0.298 / 0.246 m; eval9 0.238 / 0.264 / 0.224 m |

Three seeds: `state` 24.2 [20.1, 28.5], `answer` 22.6, difference +1.6 [−2.6, +5.6] (pair bootstrap over 695 questions).

`runs.jsonl` is the registry: every run that was started, its arguments, the commit, how it ended and its peak GPU
memory. `logs/` has the full output of the queue runs.

`record_from_locate_s0` and `locate_long_s0` were moved to Colab (L4) on Nov 22; their results are added when they finish.
