"""Writes notebooks/07_distillation_colab.ipynb (authors-only helper): python tools/make_colab_notebook.py"""
import json
from pathlib import Path

REPO_URL = 'https://github.com/NateHu203/spatial-llm-out-of-view.git'
BRANCH = 'distillation'


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.strip('\n').splitlines(keepends=True))


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip('\n').splitlines(keepends=True))


CELLS = [
    md('''
# 07 · Distillation on Colab

Trains a student that sees only the 8 frames and the question (no measuring part, no calculator) and compares it with
the current system. Use a GPU runtime with 24 GB or more (**L4** or A100); a T4 is too small.

**How it survives a disconnect.** The workspace lives on the fast local disk (`/content/workspace`). After every step
the small files (weights, checkpoints, results, logs, videos) are packed into one archive on your Google Drive. After a
disconnect, run the cells again from the top: the archive is unpacked, finished steps are skipped, and only the image
feature cache (about 15 GPU minutes) is rebuilt. Set `USE_DRIVE = False` to skip Drive.

**Continuing from the desktop.** Put the archive made on the desktop (`spatial_llm_distill_backup.tar`, about 0.8 GB)
into the top folder of your Google Drive before running. Section 1 then only rebuilds the image-feature cache.
'''),
    code('''
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
'''),
    code(f'''
USE_DRIVE = True          # keep weights, checkpoints and results on Google Drive
import os, subprocess
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
BACKUP = '/content/drive/MyDrive/spatial_llm_distill_backup.tar'

if not os.path.exists('/content/repo'):
    !git clone -q -b {BRANCH} {REPO_URL} /content/repo
%cd /content/repo
!git pull -q
!pip -q install -r requirements.txt
os.environ['SPATIAL_LLM_WORK'] = '/content/workspace'
os.environ['PYTHONPATH'] = '/content/repo/src'
os.makedirs('/content/workspace', exist_ok=True)


def restore():
    if USE_DRIVE and os.path.exists(BACKUP):
        subprocess.run(['tar', '-xf', BACKUP, '-C', '/content/workspace'], check=True)
        print('restored from', BACKUP)


def backup():
    """everything except the image-feature cache (18 GB, rebuilt in minutes) -> one archive on Drive"""
    if USE_DRIVE:
        subprocess.run(['tar', '--exclude=./sts_data/vcache', '-cf', BACKUP + '.tmp', '-C', '/content/workspace', '.'], check=True)
        os.replace(BACKUP + '.tmp', BACKUP)
        print('backed up to', BACKUP)


restore()
'''),
    md('''
## 1 · What the student needs first

Videos, the image-feature cache, model W (the starting LoRA), the measuring part's tracks and the tuned calculator
rules. On the authors' 16 GB desktop: data 19 min, cache 14 min, model W 8 min, estimator 3 min, rules 9 min. The data
and rules steps use the CPU only and are slower on Colab's fewer cores. Finished steps are skipped.
'''),
    code('''
%cd /content/repo
!python run.py data
backup()
!python run.py cache model_w estimator
backup()
!python run.py rules
backup()
'''),
    md('''
## 2 · The experiments that are still open

`tools/distill_queue.py` skips every experiment whose result file is already in the workspace (the archive from the
desktop holds all finished ones) and runs the rest. Each one is backed up to Drive when it ends; after a disconnect,
run the notebook again from the top and training continues from the last backed-up state.

| Experiment | What it is | On the 16 GB desktop |
|---|---|---|
| `record_from_locate_s0` | the student writes every sighting frame by frame, then the state, then the answer; it starts from the weights that already localise (`d_locate_s0.pt`) | about 2 h |
| `locate_long_s0` | the localisation diagnostic with 9000 steps instead of 3000 | about 35 min |
'''),
    code('''
NAMES = ['record_from_locate_s0', 'locate_long_s0']
%cd /content/repo
for name in NAMES:
    !python tools/distill_queue.py run {name}
    backup()
!python tools/distill_queue.py status
'''),
    md('''
## 3 · Any other arm (optional)

`--target` is `answer`, `state`, `motion`, `record` or `locate`; `--seed` picks the seed; `--resume` continues an
interrupted run. Tested so far (eval9, twin accuracy, chance is 25%): `answer` 22.6 (3 seeds), `state` 24.2 (3 seeds),
`motion` 12.5, `record` 17.4.
'''),
    code('''
ARMS = []                  # e.g. ['record'] ; nothing is run while this is empty
SEED = 1
%cd /content/workspace
for arm in ARMS:
    !python ../src/distill.py --target {arm} --seed {SEED} --save_lora d_{arm}_s{SEED}.pt --out res_d_{arm}_s{SEED}.jsonl --resume
    backup()
'''),
    md('''
## 4 · Results
'''),
    code('''
import glob, json
for path in sorted(glob.glob('/content/workspace/res_d_*.jsonl')):
    if path.endswith('_train.jsonl'):
        continue
    for line in open(path):
        r = json.loads(line)
        print(os.path.basename(path), '|', r['arm'], '| tested on', r['split'], '| steps', r['n_train'])
        for k, v in r['summary'].items():
            print('   ', k, v)
'''),
    code('''
# optional: download the small result files (the written text of every question is inside the res_d_*.jsonl files)
!cd /content/workspace && zip -q -r /content/distill_results.zip . -i 'res_d_*' 'logs/*' 'state_rules_results.json'
from google.colab import files
files.download('/content/distill_results.zip')
'''),
]

for n, c in enumerate(CELLS):
    c['id'] = f'cell-{n:02d}'
nb = dict(cells=CELLS, metadata=dict(accelerator='GPU', colab=dict(provenance=[], gpuType='L4'),
                                     kernelspec=dict(display_name='Python 3', name='python3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
out = Path(__file__).resolve().parent.parent / 'notebooks' / '07_distillation_colab.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('wrote', out)
