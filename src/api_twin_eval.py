"""Twin test on a larger VLM reached through an API: can a big model do what the 3B model cannot?

The model gets exactly what model W gets: the 8 frames and the question text stored with the video. Nothing is trained
and nothing else is added. Only the questions whose right answer differs between the twins are asked (eval9: 695
questions, 1390 calls), and a question counts only if BOTH twins are answered correctly, as everywhere in this repo.
Note that the 5.3% of model W is a LoRA fine-tune on ordinary videos; the models here are zero-shot.

Backends
  anthropic : Claude through the official SDK (`pip install anthropic`; credentials from ANTHROPIC_API_KEY or `ant auth login`).
  openai    : any OpenAI-compatible chat endpoint (--base_url, key read from the variable named by --key_env), for
              example a hosted Qwen2.5-VL-72B, which is the same family as our 3B model.

Each answered call is appended to the result file at once, so a stopped run continues where it was.

Usage (data: python scenes_modes.py eval9 300):
  python api_twin_eval.py --dry_run                                   # build one request, send nothing
  python api_twin_eval.py --backend anthropic --model claude-opus-5-5 --limit_pairs 30     # pilot: 30 pairs
  python api_twin_eval.py --backend anthropic --model claude-opus-5-5                      # all of eval9
  python api_twin_eval.py --backend openai --base_url https://openrouter.ai/api/v1 --key_env OPENROUTER_API_KEY \
                          --model qwen/qwen2.5-vl-72b-instruct
  python api_twin_eval.py --score_only --out res_api_<tag>.jsonl      # print the summary of a finished file"""
import argparse, base64, io, json, os, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
DATA = os.path.join(HERE, 'sts_data')
MODE_NAMES = ('stop', 'reloc', 'turn')   # scenes_modes.MODES: pair i has mode i % 3 in twin 0, constant velocity in twin 1


# ------------------------------------------------------------------ data
def load_easy(split):
    """control: ordinary scenes, both asked balls visible in the last frame, question about the last frame itself"""
    calls = []
    for scene in range(len([f for f in os.listdir(os.path.join(DATA, split)) if f.endswith('.npz')])):
        meta = json.loads(str(np.load(os.path.join(DATA, split, f'{scene:05d}.npz'))['meta']))
        for qi, q in enumerate(meta['questions']):
            if q['t'] == 7.0 and not any(meta['hidden_end']):
                calls.append(dict(scene=scene, pair=scene, twin=0, pair_mode='easy', qi=qi, t=q['t'], qtype=q['qtype'],
                                  question=q['question'], options=q['options'], label=q['label'], differs=False, static_ok=q['static_ok']))
    return calls


def summarize_easy(rows):
    """share of single answers that are right; 95% range from resampling scenes"""
    def acc(rs):
        sc = sorted({r['scene'] for r in rs}); per = {k: [r['correct'] for r in rs if r['scene'] == k] for k in sc}
        rng = np.random.default_rng(0)
        bs = [100 * np.mean([v for k in rng.choice(sc, len(sc)) for v in per[k]]) for _ in range(1000)]
        return dict(acc=round(100 * float(np.mean([r['correct'] for r in rs])), 1), ci=[round(float(np.percentile(bs, q)), 1) for q in (2.5, 97.5)], n=len(rs))
    return dict(all=acc(rows), closer=acc([r for r in rows if r['qtype'] == 'closer']), leftright=acc([r for r in rows if r['qtype'] == 'leftright']),
                no_answer=int(sum(r['pred'] is None for r in rows)), picks_first_option=round(100 * float(np.mean([r['pred'] == 0 for r in rows])), 1),
                tok_in=int(sum(r.get('tok_in') or 0 for r in rows)), tok_out=int(sum(r.get('tok_out') or 0 for r in rows)))


def load_calls(split, limit_pairs=0):
    """one call per (video, question) with differs=True; twins of a pair are videos 2i and 2i+1"""
    files = sorted(f for f in os.listdir(os.path.join(DATA, split)) if f.endswith('.npz'))
    n_pairs = len(files) // 2 if not limit_pairs else min(limit_pairs, len(files) // 2)
    calls = []
    for scene in range(2 * n_pairs):
        meta = json.loads(str(np.load(os.path.join(DATA, split, f'{scene:05d}.npz'))['meta']))
        for qi, q in enumerate(meta['questions']):
            if q['differs']:
                calls.append(dict(scene=scene, pair=meta['pair'], twin=meta['twin'], pair_mode=MODE_NAMES[meta['pair'] % 3],
                                  qi=qi, t=q['t'], qtype=q['qtype'], question=q['question'], options=q['options'],
                                  label=q['label'], differs=True, static_ok=q['static_ok']))
    return calls


def frames_png(split, scene, scale):
    """the 8 frames as base64 PNG; scale > 1 enlarges by pixel repetition (no new information)"""
    from PIL import Image
    out = []
    for fr in np.load(os.path.join(DATA, split, f'{scene:05d}.npz'))['frames']:
        if scale > 1:
            fr = fr.repeat(scale, 0).repeat(scale, 1)
        buf = io.BytesIO(); Image.fromarray(fr).save(buf, format='PNG')
        out.append(base64.standard_b64encode(buf.getvalue()).decode())
    return out


def parse_answer(text, options):
    """index of the option the model named; when it names both, the one named last; None when it names neither"""
    hits = [(m.start(), k) for k, o in enumerate(options) for m in re.finditer(rf'\b{re.escape(o)}\b', (text or '').lower())]
    return max(hits)[1] if hits else None


# ------------------------------------------------------------------ backends
def ask_anthropic(client, a, pngs, question):
    content = []
    for k, b in enumerate(pngs):
        if a.label_frames:
            content.append({'type': 'text', 'text': f'Frame {k + 1} of 8 (t = {k} s):'})
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b}})
    content.append({'type': 'text', 'text': question})
    kw = dict(model=a.model, max_tokens=a.max_tokens, messages=[{'role': 'user', 'content': content}])
    if a.effort:
        kw['output_config'] = {'effort': a.effort}
    r = client.messages.create(**kw)
    text = ''.join(b.text for b in r.content if b.type == 'text')
    return text, dict(stop=r.stop_reason, tok_in=r.usage.input_tokens, tok_out=r.usage.output_tokens)


def ask_openai(a, key, pngs, question):
    content = []
    for k, b in enumerate(pngs):
        if a.label_frames:
            content.append({'type': 'text', 'text': f'Frame {k + 1} of 8 (t = {k} s):'})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b}})
    content.append({'type': 'text', 'text': question})
    body = dict(model=a.model, max_tokens=a.max_tokens, messages=[{'role': 'user', 'content': content}])
    if a.temperature is not None:
        body['temperature'] = a.temperature
    if a.no_reasoning:
        body['reasoning'] = {'enabled': False}
    req = urllib.request.Request(a.base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                r = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (408, 409, 429) and e.code < 500 or attempt == 5:
                raise RuntimeError(f'HTTP {e.code}: {e.read()[:300]!r}')
        except OSError:   # dropped connections, TLS errors, timeouts (URLError is an OSError)
            if attempt == 5:
                raise
        time.sleep(min(60, 2 ** attempt))
    ch, u = r['choices'][0], r.get('usage') or {}
    return ch['message'].get('content') or '', dict(stop=ch.get('finish_reason'), tok_in=u.get('prompt_tokens'), tok_out=u.get('completion_tokens'))


# ------------------------------------------------------------------ scoring
def twin_acc(rows):
    """both-twins-correct per question, 95% range from resampling whole pairs (same rule as distill.twin_acc)"""
    by = {}
    for r in rows:
        by.setdefault((r['pair'], r['t'], r['qtype']), []).append(int(r['correct']))
    both = {k: int(all(v)) for k, v in by.items() if len(v) == 2}
    if not both:
        return None
    pairs = sorted({k[0] for k in both}); per = {p: [v for k, v in both.items() if k[0] == p] for p in pairs}
    rng = np.random.default_rng(0)
    bs = [100 * np.mean([v for p in rng.choice(pairs, len(pairs)) for v in per[p]]) for _ in range(1000)]
    return dict(acc=round(100 * float(np.mean(list(both.values()))), 1), ci=[round(float(np.percentile(bs, q)), 1) for q in (2.5, 97.5)], n=len(both))


def summarize(rows):
    pct = lambda v: round(100 * float(np.mean(v)), 1) if len(v) else None
    by = {}
    for r in rows:
        by.setdefault((r['pair'], r['t'], r['qtype']), []).append(r)
    done = [v for v in by.values() if len(v) == 2]
    s = dict(twin=twin_acc(rows), n_calls=len(rows), plain_acc=pct([r['correct'] for r in rows]),
             no_answer=int(sum(r['pred'] is None for r in rows)),
             # the right answers of the twins differ, so the same answer to both is always one error
             same_answer_both_twins=pct([v[0]['pred'] == v[1]['pred'] for v in done]),
             # the answer a model gives if the hidden ball stayed where it was last seen
             follows_last_seen=pct([r['pred'] == (r['label'] if r['static_ok'] else 1 - r['label']) for r in rows if r['pred'] is not None]),
             by_qtype={k: twin_acc([r for r in rows if r['qtype'] == k]) for k in ('closer', 'leftright')},
             by_time={f'{t:g}': twin_acc([r for r in rows if r['t'] == t]) for t in sorted({r['t'] for r in rows})},
             by_mode={k: twin_acc([r for r in rows if r['pair_mode'] == k]) for k in MODE_NAMES},
             tok_in=int(sum(r.get('tok_in') or 0 for r in rows)), tok_out=int(sum(r.get('tok_out') or 0 for r in rows)))
    return s


def show(s, tag):
    f = lambda d: '   -' if not d else f"{d['acc']:5.1f} [{d['ci'][0]}, {d['ci'][1]}]  n={d['n']}"
    print(f'\n{tag}\n  twin accuracy (both twins right; guessing 25, last-frame answer 0): {f(s["twin"])}')
    print(f'  single answers right {s["plain_acc"]}%   no answer {s["no_answer"]} of {s["n_calls"]} calls')
    print(f'  same answer to both twins {s["same_answer_both_twins"]}%   answer = "ball stayed where last seen" {s["follows_last_seen"]}%')
    for name in ('by_qtype', 'by_time', 'by_mode'):
        print(f'  {name}: ' + '   '.join(f'{k} {f(v)}' for k, v in s[name].items()))
    print(f'  tokens: in {s["tok_in"]:,}  out {s["tok_out"]:,}')


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('anthropic', 'openai'), default='anthropic')
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--base_url', default='https://openrouter.ai/api/v1'); ap.add_argument('--key_env', default='OPENROUTER_API_KEY')
    ap.add_argument('--split', default='eval9'); ap.add_argument('--limit_pairs', type=int, default=0)
    ap.add_argument('--scale', type=int, default=1, help='enlarge the 224 px frames by this factor (pixel repetition)')
    ap.add_argument('--label_frames', action='store_true', help='write "Frame k of 8 (t = k s):" before each frame')
    ap.add_argument('--max_tokens', type=int, default=16000, help='room for the hidden thinking of reasoning models; the answer is one word')
    ap.add_argument('--effort', default='', help='anthropic: low | medium | high | xhigh | max (empty = the model default)')
    ap.add_argument('--temperature', type=float, default=None, help='openai backend only; not sent when omitted')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--out', default='')
    ap.add_argument('--dry_run', action='store_true'); ap.add_argument('--score_only', action='store_true')
    ap.add_argument('--easy', action='store_true', help='control: last-frame questions on scenes where both balls are visible (use with --split easy3)')
    ap.add_argument('--no_reasoning', action='store_true', help='openai backend: ask the provider to switch hidden reasoning off')
    a = ap.parse_args()
    tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', a.model) + (f'_x{a.scale}' if a.scale > 1 else '') + ('_lab' if a.label_frames else '') \
        + (f'_{a.effort}' if a.effort else '') + ('' if a.split == 'eval9' else f'_{a.split}')
    out = a.out or os.path.join(HERE, f'res_api_{tag}.jsonl')

    if a.easy:
        summarize = summarize_easy
        show = lambda s, tag: print(f'\n{tag}\n  easy control, single answers right (guessing 50): ' + '   '.join(
            f"{k} {s[k]['acc']} [{s[k]['ci'][0]}, {s[k]['ci'][1]}] n={s[k]['n']}" for k in ('all', 'closer', 'leftright'))
            + f"\n  no answer {s['no_answer']}   picks the first option {s['picks_first_option']}%   tokens: in {s['tok_in']:,}  out {s['tok_out']:,}")
    if a.score_only:
        rows = [json.loads(l) for l in open(out)]
        show(summarize(rows), out); sys.exit()

    calls = load_easy(a.split)[:2 * a.limit_pairs or None] if a.easy else load_calls(a.split, a.limit_pairs)   # easy: 2 questions per scene
    done = {(r['scene'], r['qi']) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    todo = [c for c in calls if (c['scene'], c['qi']) not in done]
    print(f'{a.split}: {len({c["pair"] for c in calls})} {"scenes" if a.easy else "pairs"}, {len(calls)} calls; {len(done)} done, {len(todo)} to do -> {out}')

    if a.dry_run:
        c = calls[0]; pngs = frames_png(a.split, c['scene'], a.scale)
        print(f'model {a.model} ({a.backend}); 8 PNG frames of {224 * a.scale} px, {sum(len(p) for p in pngs) * 3 // 4:,} bytes in all')
        print(f'question: {c["question"]}\noptions {c["options"]}, right answer {c["options"][c["label"]]}')
        for t in ('red', 'The blue ball.', 'Blue is closer than red, so red', 'unsure'):
            print(f'  parse {t!r:40s} with options [blue, red] -> {parse_answer(t, ["blue", "red"])}')
        sys.exit()

    if a.backend == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(max_retries=5)
        ask = lambda pngs, q: ask_anthropic(client, a, pngs, q)
        soft = (anthropic.APIStatusError, anthropic.APIConnectionError)
    else:
        key = os.environ.get(a.key_env) or sys.exit(f'set {a.key_env} to the API key first')
        ask = lambda pngs, q: ask_openai(a, key, pngs, q)
        soft = (RuntimeError, OSError, ValueError, KeyError)

    lock = threading.Lock(); cache = {}; n_ok = n_err = 0; t0 = time.time()

    def one(c):
        global n_ok, n_err
        with lock:
            if c['scene'] not in cache:
                cache.clear(); cache[c['scene']] = frames_png(a.split, c['scene'], a.scale)   # calls come grouped by video
            pngs = cache[c['scene']]
        try:
            text, info = ask(pngs, c['question'])
        except soft as e:   # left out of the file, so the next run asks again
            with lock:
                n_err += 1; print(f'  failed scene {c["scene"]} q{c["qi"]}: {str(e)[:200]}', flush=True)
            return
        pred = parse_answer(text, c['options'])
        row = dict(c, model=a.model, split=a.split, scale=a.scale, label_frames=a.label_frames, text=text[:400], pred=pred,
                   correct=int(pred == c['label']), **info)
        with lock:
            open(out, 'a').write(json.dumps(row) + '\n'); n_ok += 1
            if n_ok % 50 == 0:
                print(f'  {n_ok}/{len(todo)} answered, {n_err} failed, {time.time() - t0:.0f}s', flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    print(f'{n_ok} answered, {n_err} failed (run again to retry the failed ones)')
    if os.path.exists(out):
        rows = [json.loads(l) for l in open(out)]
        s = summarize(rows); show(s, out)
        json.dump(dict(model=a.model, backend=a.backend, split=a.split, scale=a.scale, label_frames=a.label_frames, effort=a.effort,
                       summary=s), open(out.replace('.jsonl', '_summary.json'), 'w'), indent=1)
