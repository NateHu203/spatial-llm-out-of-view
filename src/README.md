# src/: the experiment code

These are the exact scripts that produced the reference numbers. Only two kinds of change were made for this repository:
- **Outputs go to the workspace.** In 8 files, `HERE` reads `SPATIAL_LLM_WORK` (set by `run.py`) and falls back to the script's own folder.
- **Round 9 finds round 8's readings in the workspace.** `look_choice_r9.py` takes them from the round-8 run folder instead of the authors' Colab folder.
- `calc_ladder.py` is new. It is the JSON-writing version of the authors' `flow_calc_qtype.py`, and its numbers are identical.

They are research scripts, not a library:
- they read and write files in their working folder;
- they keep module-level settings;
- some of them modify each other's globals; for example, the round-5 to round-7 scripts set `crop_ablation.SPLITS` and `crop_ablation.RUN`.

That is why `run.py` runs each one as its own process. Do not import them into one long-lived process and call several stages in a row.

## Which file does what (in pipeline order)

| File | Block / stage | Role |
|---|---|---|
| `scenes.py` | data | 3D renderer of the floor, balls and moving camera; natural scenes and exit twins; exact camera geometry |
| `scenes_modes.py` | data | Motion-mode twins (stop / moved / turn vs straight); `python scenes_modes.py <split> <pairs>` |
| `sts.py` | data, cache, blocks 0, 1, 3, 4 | Split registry and loading (`load`, `gen`, `scene_tracks`), feature cache (`vis_feats`), LoRA training and evaluation (`run`), the record sentence (`--text_state`, `--state_rule`, `--dist_text`). It also holds flags from earlier, abandoned experiments, which `run.py` never uses |
| `mve_nowrope.py` | block 0 | `build()`: loads Qwen2.5-VL-3B and attaches the LoRA. The rest of the file is the abandoned NowRoPE pilot |
| `gen_tracks.py` | data | Exact per-frame positions for the natural splits |
| `precache.py` | cache | Fills `sts_data/vcache/` |
| `ladder_stage_a.py`, `ladder_stage_b.py` | blocks 2, 3 | The observed-only velocity estimate and the plain record sentence (used by `state_estimator.py` and `sts.py`) |
| `state_estimator.py` | block 2 | The measuring part: model, training, eval8 and eval3 tracks |
| `est_infer.py` | block 2 | Runs the measuring part on any split → `<split>_est_tracks.json` |
| `state_rules.py` | blocks 3, 5 | The calculator: motion rules (straight line, Kalman, IMM, change-aware `ca`), grid tuning, scoring |
| `m_pairacc.py` | scoring | Twin accuracy of a model result file, with a pair bootstrap |
| `calc_ladder.py` | table 1 | The calculator answering directly on eval9 |
| `crop_ablation.py` | block 6 | Round 4 base: geometry floor ↔ pixel, the extra-look encoder, the `Head` network, training and scoring |
| `crop_ablation_r5.py` | block 6, table 2 | Versions 1, 2, 3 on eval10 (3 seeds) |
| `crop_ablation_r6.py` | block 6, table 2 | Versions 1h, 2h (head-start control) |
| `crop_ablation_r7.py` | block 6, table 2 | Version 3c (64-token look) on eval11 |
| `look_choice_r8.py` | block 7, table 3 | Zooms on every sighting; look-choosing rules on eval12 |
| `look_choice_r9.py` | block 7, table 3 | The same on eval13 |

The docstring at the top of each file says what it does, its stages and its outputs. Each `GATES_*` file named in a docstring holds that round's pass rules. Those files are in the authors' project folder, and the rules are summarised in [docs/04_RESULTS.md](../docs/04_RESULTS.md).
