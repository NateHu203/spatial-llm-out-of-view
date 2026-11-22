"""Run the distillation experiments one after another, and show where they stand.

    python tools/distill_queue.py run      run every experiment of QUEUE that has no result yet (safe to re-run:
                                           finished ones are skipped, interrupted ones continue from their checkpoint)
    python tools/distill_queue.py status   one line per experiment: state, training step, scenes tested, key numbers

Everything is written under the workspace (default <repo>/workspace, or $SPATIAL_LLM_WORK):
    runs.jsonl                     every run that was started, how it ended, its numbers, its peak GPU memory
    res_d_<name>.jsonl             the result (summary + every question with the text the student wrote)
    res_d_<name>_train.jsonl       the loss curve, one point per 100 steps
    res_d_<name>.jsonl.partial     scenes tested so far (exists only while a test is unfinished)
    d_<name>.pt, d_<name>.pt.ckpt  the trained LoRA and the rolling training checkpoint
    logs/distill/<name>.log        the full output of the run
"""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get('SPATIAL_LLM_WORK', REPO / 'workspace')).resolve()

# name -> extra arguments of src/distill.py. Order = priority. Seed 0 of answer / state / motion was run before this
# queue existed (res_d_answer.jsonl, res_d_state.jsonl, res_d_motion.jsonl) and is listed for the status table only.
QUEUE = {
    'locate_s0': ['--target', 'locate', '--seed', '0', '--eval_split', 'eval3,eval9'],
    'locate_s1': ['--target', 'locate', '--seed', '1', '--eval_split', 'eval3,eval9'],
    'locate_s2': ['--target', 'locate', '--seed', '2', '--eval_split', 'eval3,eval9'],
    'state_s1': ['--target', 'state', '--seed', '1'],
    'answer_s1': ['--target', 'answer', '--seed', '1'],
    'state_s2': ['--target', 'state', '--seed', '2'],
    'answer_s2': ['--target', 'answer', '--seed', '2'],
    'record_s0': ['--target', 'record', '--seed', '0', '--differs_only'],
    # added 2026-11-22 after locate came out at 0.28 m ("partly"): longer training, as the pass rule says, and the
    # per-frame record started from the weights that already localise
    'locate_long_s0': ['--target', 'locate', '--seed', '0', '--n_train', '9000', '--eval_split', 'eval3,eval9'],
    'record_from_locate_s0': ['--target', 'record', '--seed', '0', '--differs_only', '--init_lora', 'd_locate_s0.pt'],
}
EARLIER = {'answer_s0': 'res_d_answer.jsonl', 'state_s0': 'res_d_state.jsonl', 'motion_s0': 'res_d_motion.jsonl'}


def paths(name):
    return dict(out=WORK / f'res_d_{name}.jsonl', lora=WORK / f'd_{name}.pt', log=WORK / 'logs' / 'distill' / f'{name}.log')


def done(name):
    p = paths(name)
    if not p['out'].exists() or Path(str(p['out']) + '.partial').exists():
        return False
    extra = QUEUE[name]   # one result line per tested split (the locate arm tests two)
    n_splits = len(extra[extra.index('--eval_split') + 1].split(',')) if '--eval_split' in extra else 1
    return len([l for l in open(p['out'], encoding='utf-8') if l.strip()]) >= n_splits


def run():
    env = dict(os.environ, SPATIAL_LLM_WORK=str(WORK), PYTHONPATH=str(REPO / 'src'), PYTHONUNBUFFERED='1')
    for name, extra in QUEUE.items():
        if done(name):
            print(f'[{name}] already finished', flush=True)
            continue
        p = paths(name); p['log'].parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO / 'src' / 'distill.py'), *extra, '--save_lora', p['lora'].name, '--out', str(p['out']), '--resume']
        for attempt in (1, 2):   # one automatic retry: it continues from the checkpoint / the partial file
            print(f'[{name}] start (attempt {attempt}) {time.strftime("%H:%M:%S")}', flush=True)
            with open(p['log'], 'a', encoding='utf-8') as log:
                log.write(f'\n$ {" ".join(cmd)}\n'); log.flush()
                code = subprocess.run(cmd, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code == 0:
                break
            print(f'[{name}] exit code {code}', flush=True)
        if code != 0:
            print(f'[{name}] FAILED twice; stopping the queue. Log: {p["log"]}', flush=True)
            sys.exit(1)
        print(f'[{name}] finished {time.strftime("%H:%M:%S")}', flush=True)
    print('queue finished', flush=True)


def last_json(path):
    lines = [l for l in open(path, encoding='utf-8') if l.strip()] if Path(path).exists() else []
    return [json.loads(l) for l in lines]


def key_numbers(res):
    s = res['summary']
    if res['arm'] == 'locate':
        return (f"{res['split']}: position error {s['err_med_m']} m (measuring part {s['measuring_part_err_med_m']} m), "
                f"visibility {s['vis_acc']}%, n={s['n_err']}")
    out = f"{res['split']}: twin accuracy {s['model']['acc']} {s['model']['ci']}"
    if s.get('written'):
        out += f", from its own numbers {s['written']['acc']}, position error {s['err_med_m']} m"
    return out


def status():
    names = list(EARLIER) + list(QUEUE)
    print(f'workspace: {WORK}')
    for name in names:
        out = WORK / EARLIER[name] if name in EARLIER else paths(name)['out']
        curve = Path(str(out).replace('.jsonl', '_train.jsonl')); part = Path(str(out) + '.partial')
        pts = last_json(curve)
        step = f"step {pts[-1]['step']}, loss {pts[-1]['loss']}" if pts else ''
        if out.exists() and not part.exists():
            for res in last_json(out):
                print(f'{name:11s} done      {key_numbers(res)}')
        elif part.exists():
            n = max(0, len([l for l in open(part, encoding='utf-8') if l.strip()]) - 1)
            print(f'{name:11s} testing   {n} scenes saved   ({step})')
        elif pts:
            print(f'{name:11s} training  {step}')
        else:
            print(f'{name:11s} waiting')
    runs = last_json(WORK / 'runs.jsonl')
    ended = {r['run'] for r in runs if r['event'] in ('end', 'error')}
    for r in runs:
        if r['event'] == 'error':
            print(f"error   {r['time']} {r['out']}: {r['error'][:160]}")
        if r['event'] == 'start' and r['run'] not in ended:
            print(f"no end recorded (still running, or was stopped): {r['time']} {r['out']}")
    peaks = [r['peak_gpu_gb'] for r in runs if r.get('peak_gpu_gb')]
    if peaks:
        print(f'highest GPU memory of any recorded run: {max(peaks)} GB')


if __name__ == '__main__':
    {'run': run, 'status': status}[sys.argv[1] if len(sys.argv) > 1 else 'status']()
