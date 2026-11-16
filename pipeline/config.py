"""Where things live, and the sizes of every data split. Everything else in pipeline/ reads from here.

Folders
  REPO       this repository
  SRC        the experiment code (src/): plain scripts that are run as separate processes
  WORK       the workspace: generated videos, feature cache, trained weights, result files, logs.
             Default <repo>/workspace; override with the environment variable SPATIAL_LLM_WORK or `run.py --work`.
             It is never committed (see .gitignore).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / 'src'
REFERENCE = REPO / 'reference'
WORK = Path(os.environ.get('SPATIAL_LLM_WORK', REPO / 'workspace')).resolve()

MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'   # downloaded from Hugging Face on first use (about 7.5 GB)

# Data splits: name -> (generator, number passed to the generator, number of videos it produces)
#   natural : ordinary scenes, numbered videos             (sts.gen; n = videos)
#   exit    : exit twins, two videos per pair              (sts.gen; n = pairs)
#   modes   : motion-mode twins (stop / moved / turn)      (scenes_modes.py; n = pairs)
SPLITS = {
    'train3': ('natural', 3000, 3000),   # ordinary scenes: trains model W and the measuring part
    'eval3': ('natural', 300, 300),      # ordinary scenes: side checks
    'train8': ('exit', 600, 1200),       # exit twins: trains the measuring part, the record LoRA, rule tuning
    'eval8': ('exit', 300, 600),         # exit twins: development test
    'train9': ('modes', 300, 600),       # motion-mode twins: rule tuning, look-choice tuning
    'eval9': ('modes', 300, 600),        # motion-mode twins: the A / B / C ladder
    'eval9s': ('modes', 300, 600),       # strict version of eval9 (no pixel of the hidden ball after it leaves)
    'eval10': ('modes', 300, 600),       # fresh twins: Versions 1-3 (round 5-6)
    'eval11': ('modes', 300, 600),       # fresh twins: the cheaper look (round 7)
    'eval12': ('modes', 300, 600),       # fresh twins: choosing looks (round 8)
    'eval13': ('modes', 300, 600),       # fresh twins: choosing looks (round 9)
}


def last_video(split):
    """relative path of the last video file of a split (its existence means the split is complete)"""
    return f'sts_data/{split}/{SPLITS[split][2] - 1:05d}.npz'


def set_work(path):
    """point every pipeline module at another workspace (run.py --work); call before running steps"""
    global WORK
    WORK = Path(path).resolve()
    os.environ['SPATIAL_LLM_WORK'] = str(WORK)
