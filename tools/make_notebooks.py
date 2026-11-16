"""Build notebooks/*.ipynb from the cell lists below (edit here, then run: python tools/make_notebooks.py).
Keeping the notebooks generated from one file means the explanations and the code cannot drift apart."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'notebooks'

SETUP = '''import sys; sys.path.insert(0, '.')     # nb_helpers.py sits next to this notebook
import nb_helpers as H
H.setup(quick=False, work=None)            # quick=True: minutes instead of hours, meaningless numbers
                                           # work='...': another workspace folder (default <repo>/workspace)'''


def md(s):
    return dict(cell_type='markdown', metadata={}, source=s.strip('\n'))


def code(s):
    return dict(cell_type='code', metadata={}, source=s.strip('\n'), outputs=[], execution_count=None)


NB = {
'00_overview_and_setup': [
md('''# 00: What this project is, and checking your machine

**Plain.**
- A small video-language model (Qwen2.5-VL-3B) is asked where a ball is **after it has rolled out of the camera's view**, and it gets it wrong.
- We test this on **twin videos**: two 8-frame videos with identical last frames. In one, the ball rolled straight out of view; in the other, it stopped, was moved, or turned before leaving.
- A question counts as correct only if both twins are answered correctly. Random guessing gets 25%; answering from the last frame gets 0%.
- The model alone gets **5%**. Our system (a written record outside the model, a calculator, and zoomed looks) gets **about 75%**.

**Technical.**
- `run.py` runs ten stages in order, writing everything into one workspace folder: data → cache → model_w → estimator → record_lora → rules → ladder → versions → look_choice → report.
- Each step is skipped if its outputs exist, so you can stop and resume.
- The notebooks 01–06 each run a few stages and explain them. See `docs/` for the full guides.'''),
code('''import torch, transformers, numpy
print('torch', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE: GPU stages will fail')
if torch.cuda.is_available():
    print('GPU memory (GB):', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), '(24 or more needed)')'''),
code(SETUP),
md('''## The plan
Each stage lists its steps, a rough time, and `done` for steps whose outputs already exist.'''),
code("H.run('list')"),
md('''## Running everything at once
Uncomment to run all stages, about 4–5 GPU hours plus 1.5 CPU hours. Otherwise go through notebooks 01–06.'''),
code("# H.run('all')"),
],

'01_data': [
md('''# 01: The videos (stage `data`) and the feature cache (stage `cache`)

**Plain.**
- Nothing is downloaded: every video is drawn on your machine by a small 3D renderer from fixed random seeds, so it comes out exactly like ours.
- Three kinds of videos are made:
  - **ordinary videos** to teach the model the task;
  - **exit twins**, where one ball leaves the view with a different speed;
  - **motion-mode twins**, where one ball stops, is moved or turns before leaving, and its twin rolls straight.

  All main results use the motion-mode twins.
- Then the model's frozen image encoder looks at every frame once, and its output is saved (the **cache**). This makes later steps much faster.

**Technical.**
- 11 splits, about 9,300 videos of 8 frames at 224×224, written to `workspace/sts_data/`.
- The last step compares SHA-256 fingerprints of the first 4 videos of each split with ours.
- Cache: `sts_data/vcache/<split>/<i>.pt`, 64 merged tokens × 2048 per frame, bf16, about 20 GB.
- Details: `docs/01_DATA.md`.'''),
code(SETUP),
code("H.run('data')      # about 45 min on CPU the first time"),
md('''## Look at one twin pair
Top row: the ball that leaves stops, is moved, or turns. Bottom row: it rolls straight. From the moment the ball is last seen, the frames are identical; only the earlier frames tell them apart.'''),
code("H.show_twin('eval9', pair=0)"),
code("H.show_twin('eval9', pair=2)"),
code("H.run('cache')     # about 25 min on GPU, about 20 GB of disk"),
],

'02_models_W_estimator_record': [
md('''# 02: Model W, the measuring part, and the record reader

Stages `model_w`, `estimator`, `record_lora` (blocks 1–3 in `docs/02_MODEL_STRUCTURE.md`).

**Plain.**
1. **Model W.** Teach the video-language model the question format on ordinary videos by training a thin "correction" on top of it (LoRA). The big model itself never changes.
2. **The measuring part.** A small separate network reads the model's view of each frame and says, for each ball, whether it is visible and where it is on the floor. Together with the known motion of the camera, this gives each ball's **estimated track**.
3. **The record reader.** A second LoRA teaches the model to answer from the **last frame plus one sentence** written by a calculator, for example "the red ball at (−1.20, +3.00) …", instead of from all 8 frames.

**Technical.**
- W: LoRA r = 16 on q, k, v, o of all 36 decoder layers; 1000 steps on train3 (`w1k_s0.pt`).
- Measuring part: a 2.45M-parameter transformer over the cached 64 tokens; 3000 steps on train3 + train8 (`state_estimator.pt`); tracks for every split (`<split>_est_tracks.json`).
- Record LoRA: from W, 1000 steps on train3 + train8 with `--text_state --hist last1` (`r1_lsq_s1.pt`).'''),
code(SETUP),
code("H.run('model_w')       # ~9 min"),
code("H.run('estimator')     # ~11 min"),
code("H.compare('estimator')"),
code("H.run('record_lora')   # ~8 min"),
],

'03_fixes_ABC_ladder': [
md('''# 03: Fixes A, B, C: the ladder (table 1)

Stages `rules` and `ladder`.

**Plain.** The same twin questions (eval9) are answered by:
- **the model alone**, which sees all 8 frames: about 5%;
- **+ A**: the model sees the last frame and a sentence from the written record, in which a calculator drew one straight line through each ball's sightings;
- **+ B**: the sentence also gives each ball's distance from the camera;
- **+ C**: the calculator's **change-aware rule** restarts the line when a ball stopped, was moved or turned. This is the **current system**, about 67%;
- **a side test** with exact positions instead of estimated ones: about 97%. The remaining loss is in *measuring* the positions.

We also score the calculator answering directly, with no model.

**Technical.**
- `state_rules.py` tunes every rule on train8 + train9 by grid search; `ca` has (σ, κ, v_rest).
- Six evaluation-only `sts.py run` calls on eval9 (600 videos, 695 question pairs).
- `calc_ladder.py` scores the calculator directly.
- Scoring: twin accuracy with a pair bootstrap (`m_pairacc`).'''),
code(SETUP),
code("H.run('rules')     # ~20 min CPU"),
code("H.run('ladder')    # ~15 min GPU"),
code("H.compare('ladder'); H.compare('calc')"),
],

'04_extra_looks_versions': [
md('''# 04: An extra look at the ball: Versions 1, 2, 3, 3c (table 2)

Stage `versions` (block 6).

**Plain.**
- In each frame, one ball (taken in turn) gets a second, closer measurement from a small extra network. The versions differ only in what extra picture that network sees:
  - **Version 1**: nothing extra;
  - **Version 2**: the whole frame enlarged;
  - **Version 3**: a 96-pixel square cut around the ball and enlarged;
  - **Version 3c**: the same square with 4× fewer tokens.
- The "h" versions (1h, 2h) get the same head start as Version 3: they are told where the square is. This makes the comparison fair.
- Each version is trained 3 times and scored on fresh twins.

**Technical.**
- Rounds 5, 6 and 7: `crop_ablation_r5.py` (eval10), `_r6.py` (eval10), `_r7.py` (eval11). Each runs encode → train (4000 steps × 3 seeds) → score.
- Rule C is re-tuned per version on train8 + train9.
- Trained heads are handed between rounds by the pipeline (hard links).
- About 3 GPU hours in total; on an A100, encoding is the slow part.'''),
code(SETUP),
code("H.run('versions')"),
code("H.compare('versions')"),
],

'05_choosing_looks': [
md('''# 05: Which frames to zoom on (table 3)

Stage `look_choice` (block 7).

**Plain.**
- With only two zoomed looks per video, where should they go?
- We compare Version 3's schedule (its last two looks) with **the last frame in which each ball was seen**.
- The answer is a forecast past the end of each ball's track, so an error at the end of the track swings the forecast the most. The last sighting is the most valuable place to measure precisely.
- This rule won on three separate sets of fresh twins (about +4 points).

**Technical.**
- Round 8 (`look_choice_r8.py`):
  - zooms on every estimated-visible sighting of train9, eval10 and eval12 with the 3 round-5 Version-3 heads (no new training);
  - each chooser picks which readings replace the estimates;
  - rule C is re-tuned on train9 per chooser and seed.
- Round 9 (`look_choice_r9.py`) adds eval13.'''),
code(SETUP),
code("H.run('look_choice')   # ~35 min"),
code("H.compare('look')"),
],

'06_report': [
md('''# 06: All your numbers next to ours

- **match** = inside our 95% range;
- **close** = within twice that range;
- **CHECK** = further away.

GPU training is not bit-reproducible, so trained results land near ours, not on them. Scoring steps given the same inputs reproduce our digits exactly. The report is also written to `workspace/my_results.md`.'''),
code(SETUP),
code("H.run('report')"),
code("H.compare()"),
],
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, cells in NB.items():
        nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                             language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
        for i, c in enumerate(nb['cells']):
            c['id'] = f'{name[:2]}-{i:02d}'
        json.dump(nb, open(OUT / f'{name}.ipynb', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print('wrote', OUT / f'{name}.ipynb')


if __name__ == '__main__':
    main()
