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
## 2 · Localisation diagnostic (run this first)

The three arms tried so far all fail at the same place: the positions the student writes are off by about 1 m, against
0.10 m for the measuring part. This arm isolates that step: no question, no motion, no extrapolation. The student
writes where the two balls are in the **last frame**, or that they are not visible. About 10 minutes.
'''),
    code('''
%cd /content/workspace
!python ../src/distill.py --target locate --eval_split eval3,eval9 --save_lora d_locate_s0.pt --out res_d_locate.jsonl
backup()
'''),
    md('''
## 3 · The distillation arms

Each arm trains for 3000 steps (about 30 minutes on a 16 GB desktop) and is then tested on eval9. Add `--resume` to
continue an interrupted run: training restarts from the last checkpoint and evaluation skips finished scenes.

| `--target` | The student writes | Tested on the authors' desktop |
|---|---|---|
| `answer` | the answer word only (control) | 22.2% |
| `state` | each ball's position at the asked time, then the answer | 27.5% |
| `motion` | camera motion, each ball's position and velocity, the state, the answer | 12.5% |
| `record` | every sighting frame by frame, the state, the answer | not yet run without camera motion |
'''),
    code('''
ARMS = ['record']          # e.g. ['record', 'state', 'answer']; add more seeds with --seed
%cd /content/workspace
for arm in ARMS:
    !python ../src/distill.py --target {arm} --save_lora d_{arm}_s0.pt --out res_d_{arm}.jsonl --resume
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
