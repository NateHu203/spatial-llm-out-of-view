# 3. Running it: the main file, the notebooks, and what each step costs

## The short version

```bash
pip install -r requirements.txt     # after installing a CUDA build of PyTorch
python run.py list                  # the stages, their steps, rough times, and what is already done
python run.py all                   # everything, in order; resumable
python run.py report                # your numbers next to ours
```

`run.py` is the **main running file**. Everything else is called from it:
- the notebooks in `notebooks/` call it one stage at a time;
- the scripts in `src/` are what it runs.

## How a run is organised

**Plain.**
- The project is a chain of stages, and each stage needs the files made by the ones before it.
- `run.py` runs them in order. Each step writes its output files into one folder, the **workspace**.
- If a step's output files already exist, the step is skipped. So when a run stops (power cut, closed laptop, Colab timeout), you type the same command again and it continues where it stopped.
- Every step writes a log file. A failed step prints the end of its log and the path of the full log.

**Technical.**
- `pipeline/stages.py` defines the stages as lists of `Step`s. Each step has a command, its output files, and a GPU flag.
- `pipeline/runner.py` runs each command as `python src/<script> ...` with:
  - `cwd = workspace`;
  - `PYTHONPATH = src/`;
  - `SPATIAL_LLM_WORK = workspace`.

  Every script in `src/` writes its outputs to `SPATIAL_LLM_WORK`, falling back to its own folder when the variable is unset.
- Scripts that append to their result file (`sts.py run`) have that file deleted before they start.
- Logs go to `workspace/logs/<stage>/<step>.log`.
- Trained heads are handed from one round to the next with hard links (`copy_into`), so the large feature files are not duplicated.

## Options

| Option | Effect |
|---|---|
| `--work PATH` | Use another workspace (default `./workspace` or `$SPATIAL_LLM_WORK`) |
| `--quick` | Short training and evaluation for every stage. It checks that the code runs end to end in about 1–2 GPU hours, but the numbers are meaningless. It still needs the full data and cache. Use a separate workspace: `--quick --work workspace_quick` |
| `--step TEXT` | Run only steps whose name contains TEXT, e.g. `python run.py ladder --step m_r_ca_dist` |
| `--force` | Re-run steps even if their outputs exist |
| `--dry` | Print what would run |

## Stages, in order

The times below were measured on one RTX 5090 Laptop GPU (24 GB) or one Colab A100 (rounds 5–8).

| # | Stage | What it does (plain) | GPU | Time |
|---|---|---|---|---|
| 1 | `data` | Generate all 11 video splits from fixed seeds and check they match ours | – | ~45 min CPU |
| 2 | `cache` | Run the frozen image encoder once on every frame | ✓ | ~25 min, 20 GB |
| 3 | `model_w` | Train model W (LoRA) on ordinary videos | ✓ | ~9 min |
| 4 | `estimator` | Train the measuring part; make estimated tracks for every split | ✓ | ~11 min |
| 5 | `record_lora` | Train the LoRA that answers from the written record | ✓ | ~8 min |
| 6 | `rules` | Tune the calculator's motion rules | – | ~20 min CPU |
| 7 | `ladder` | Model + calculator on eval9: A → A+B → A+B+C | ✓ | ~15 min |
| 8 | `versions` | Versions 1, 2, 3 (eval10); 1h, 2h (eval10); 3c (eval11) | ✓ | ~3 h |
| 9 | `look_choice` | Zoom on every sighting; score "last sighting of each ball" (eval12, eval13) | ✓ | ~35 min |
| 10 | `report` | Your numbers next to ours | – | seconds |

The total is about 4–5 GPU hours plus about 1.5 CPU hours. A free Colab GPU (T4) is too small; use a GPU with 24 GB or more.

## On Colab

```python
!git clone <your-repo-url> repo
%cd repo
!pip -q install -r requirements.txt
import os; os.environ['SPATIAL_LLM_WORK'] = '/content/drive/MyDrive/spatial_llm_workspace'   # survives disconnects
!python run.py all
```

A session that ends early loses nothing that was finished. Mount Drive, run the same cell again, and finished steps are skipped. The workspace needs about 50 GB, so keep it on Drive only if your Drive has the space; otherwise use `/content` and expect to restart from the last finished stage.

## The notebooks

`notebooks/` has one notebook per group of stages. Each one explains its stages in plain words, runs them through `run.py`, and shows your numbers next to ours.

| Notebook | Stages |
|---|---|
| `00_overview_and_setup.ipynb` | Environment check; what the project is; `run.py list` |
| `01_data.ipynb` | `data`, `cache`; looks at one twin pair |
| `02_models_W_estimator_record.ipynb` | `model_w`, `estimator`, `record_lora` |
| `03_fixes_ABC_ladder.ipynb` | `rules`, `ladder`: table 1 |
| `04_extra_looks_versions.ipynb` | `versions`: table 2 |
| `05_choosing_looks.ipynb` | `look_choice`: table 3 |
| `06_report.ipynb` | `report`: all tables against ours |

## Debugging

- **A step failed:**
  1. Open `workspace/logs/<stage>/<step>.log`; its first line is the exact command.
  2. To reproduce it by hand, run that command from the workspace folder with `PYTHONPATH=<repo>/src` and `SPATIAL_LLM_WORK=<workspace>`.
- **Re-running one step:** delete its output files (listed by `run.py list`), or use `--force --step <name>`.
- **The data fingerprints say DIFFERENT:** your numpy or Python produced different random numbers or rounding. Everything still runs, but numbers will differ slightly from ours. Tested: numpy 2.5.2 on Python 3.13.
- **Out of GPU memory:** the language-model steps need about 20 GB. On smaller GPUs, nothing in this repo reduces that.

## What is exact and what is statistical

**Exact, the same digits as ours:**
- the videos (checked by fingerprint);
- every scoring step, when given our trained weights and readings. We checked this for the calculator ladder and for rounds 5–9.

**Statistical, within the 95% ranges:**
- everything that trains on the GPU: model W, the measuring part, the record LoRA and the heads. GPU training is not bit-reproducible.
- everything downstream of those, as a result.

`run.py report` marks each number:
- **match**: inside our 95% range;
- **close**: within twice that range;
- **CHECK**: further away.
