"""Runs one step of a stage: a command in the workspace, with its own log file, skipped if its outputs exist.

Why separate processes: the experiment scripts in src/ are the exact code that produced our numbers. They read and
write files in their working folder and keep module-level state, so each step runs as `python src/<script> ...` with
  cwd                = the workspace
  PYTHONPATH         = src/          (so `import sts`, `import state_rules`, ... work)
  SPATIAL_LLM_WORK   = the workspace (the scripts write their outputs there instead of next to themselves)
"""
import os, shutil, subprocess, sys, time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from . import config


@dataclass
class Step:
    name: str                                   # short id, also the log file name
    what: str                                   # one line in plain words
    cmd: Union[List[str], Callable, None]       # ['script.py', args...] run with python, or a Python function
    outputs: List[str] = field(default_factory=list)   # files (relative to the workspace) that mean "done"
    gpu: bool = False
    minutes: str = ''                           # rough full-run time on one RTX 5090 laptop / A100
    fresh: List[str] = field(default_factory=list)     # files to delete before running (scripts that APPEND)


def env():
    e = dict(os.environ)
    e['PYTHONPATH'] = str(config.SRC) + os.pathsep + e.get('PYTHONPATH', '')
    e['SPATIAL_LLM_WORK'] = str(config.WORK)
    e.setdefault('PYTHONIOENCODING', 'utf-8')
    return e


def done(step):
    return bool(step.outputs) and all((config.WORK / o).exists() for o in step.outputs)


def run_step(step: Step, stage: str, force=False, dry=False):
    tag = f'[{stage}/{step.name}]'
    if done(step) and not force:
        print(f'{tag} skip (outputs exist)')
        return True
    if dry:
        print(f'{tag} would run: {step.what}  {step.cmd if isinstance(step.cmd, list) else step.cmd.__name__}')
        return True
    config.WORK.mkdir(parents=True, exist_ok=True)
    logs = config.WORK / 'logs' / stage
    logs.mkdir(parents=True, exist_ok=True)
    for f in step.fresh:
        p = config.WORK / f
        if p.exists():
            p.unlink()
    print(f'{tag} {step.what} ...', flush=True)
    t0 = time.time()
    if callable(step.cmd):
        step.cmd()
        ok = True
    else:
        script, *args = step.cmd
        exe = [sys.executable, str(config.SRC / script)] + args if script.endswith('.py') else [sys.executable] + step.cmd
        log = logs / f'{step.name}.log'
        with open(log, 'w', encoding='utf-8') as fh:
            fh.write('$ ' + ' '.join(exe) + '\n'); fh.flush()
            r = subprocess.run(exe, cwd=config.WORK, env=env(), stdout=fh, stderr=subprocess.STDOUT)
        ok = r.returncode == 0
        if not ok:
            tail = open(log, encoding='utf-8', errors='replace').read().splitlines()[-25:]
            print('\n'.join(tail))
            print(f'{tag} FAILED (exit {r.returncode}); full log: {log}')
            return False
    missing = [o for o in step.outputs if not (config.WORK / o).exists()]
    if missing:
        print(f'{tag} FAILED: expected outputs missing: {missing}')
        return False
    print(f'{tag} done in {time.time() - t0:.0f} s')
    return ok


def copy_into(src_dir: str, dst_dir: str, pattern: str):
    """copy files matching pattern from one workspace folder to another (used to hand trained heads between rounds)"""
    s, d = config.WORK / src_dir, config.WORK / dst_dir
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(s.glob(pattern)):
        if not (d / f.name).exists():
            try:
                os.link(f, d / f.name)          # hard link: no extra disk for the large feature files
            except OSError:
                shutil.copy2(f, d / f.name)     # other drive / file system: plain copy
            n += 1
    print(f'    linked/copied {n} file(s) {src_dir}/{pattern} -> {dst_dir}/')
