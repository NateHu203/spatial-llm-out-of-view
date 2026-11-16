"""Helpers shared by the notebooks: run a stage through run.py, show results next to ours, look at a twin pair."""
import json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline import config, report  # noqa: E402

QUICK = False      # True: short training/eval, meaningless numbers (checks that everything runs)
WORK = None        # None: <repo>/workspace (or $SPATIAL_LLM_WORK); or a path string


def setup(quick=False, work=None):
    """call once at the top of a notebook"""
    global QUICK, WORK
    QUICK, WORK = quick, work
    if work:
        config.set_work(work)
    print('repo     :', REPO)
    print('workspace:', config.WORK, '(QUICK mode: numbers are meaningless)' if quick else '')


def run(*stages, extra=()):
    """run stages with run.py and stream its output; raises if a step fails (its log path is printed)"""
    cmd = [sys.executable, str(REPO / 'run.py'), *stages] + (['--quick'] if QUICK else []) + \
          (['--work', str(WORK)] if WORK else []) + list(extra)
    print('$', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                         errors='replace')
    for line in p.stdout:
        print(line, end='')
    if p.wait():
        raise RuntimeError('a step failed: read the log named above, fix, and re-run this cell (finished steps are skipped)')


def compare(prefix=''):
    """table of your numbers next to ours, for result keys starting with prefix"""
    ref = json.load(open(config.REFERENCE / 'expected_results.json'))['values']
    mine = report.collect(lambda name: str(config.WORK / name))
    rows = []
    for k, desc, _, _ in report.ITEMS:
        if not k.startswith(prefix):
            continue
        r, m = ref.get(k), mine.get(k)
        rows.append((k, desc, f'{m[0]:.2f}' if m else '(not run)', report._fmt(r),
                     (report.verdict(m, r) if (m and r and not QUICK) else '')))
    try:
        import pandas as pd
        from IPython.display import display
        display(pd.DataFrame(rows, columns=['key', 'what', 'yours', 'ours [95% range]', 'verdict']).set_index('key'))
    except ImportError:
        for r in rows:
            print(' | '.join(r))


def show_twin(split='eval9', pair=0):
    """draw the 8 frames of both videos of a twin pair, and print its questions"""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
    for w in (0, 1):
        z = np.load(config.WORK / 'sts_data' / split / f'{2 * pair + w:05d}.npz')
        meta = json.loads(str(z['meta']))
        for t in range(8):
            ax[w, t].imshow(z['frames'][t]); ax[w, t].set_xticks([]); ax[w, t].set_yticks([])
            if w == 0:
                ax[w, t].set_title(f't = {t}')
        ax[w, 0].set_ylabel(f"twin {w}: {meta.get('mode', 'cv')}", fontsize=11)
        if w == 0:
            print('balls asked about:', meta['names'], '| motion of the ball that leaves:', meta.get('mode'),
                  '| change at frame', meta.get('change_t'), '| last seen at frame', meta.get('k_last'))
            for q in meta['questions']:
                if q.get('differs'):
                    print(f"  t* = {q['t']:.0f}  {q['qtype']:9s}  answer twin 0 = {q['options'][q['label']]}")
    plt.suptitle(f'{split}, pair {pair}: identical last frames, different past'); plt.tight_layout(); plt.show()
