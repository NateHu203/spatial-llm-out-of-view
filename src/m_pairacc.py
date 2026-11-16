"""Pair accuracy (T8) of Stage M result files, overall and by question type, with a pair bootstrap CI.
Usage: python m_pairacc.py res_a.jsonl [res_b.jsonl ...]   (optionally --exclude eval9_leak_pairs.json)"""
import json, sys, numpy as np

def by_key(rows):
    by = {}
    for r in rows:
        if r['differs']:
            by.setdefault((r['pair'], r['t'], r['qtype']), []).append(r)
    return {k: int(all(x['correct' if 'correct' in x else 'ok'] for x in v)) for k, v in by.items() if len(v) == 2}

def stats(both, qtype=None, excl=()):
    b = {k: v for k, v in both.items() if (qtype is None or k[2] == qtype) and k[0] not in excl}
    pairs = sorted({k[0] for k in b}); per = {p: [v for k, v in b.items() if k[0] == p] for p in pairs}
    rng = np.random.default_rng(0)
    bs = [100 * np.mean([v for p in rng.choice(pairs, len(pairs)) for v in per[p]]) for _ in range(1000)]
    return round(100 * np.mean(list(b.values())), 1), [round(float(np.percentile(bs, q)), 1) for q in (2.5, 97.5)], len(b)

if __name__ == '__main__':
    args = sys.argv[1:]; excl = ()
    if '--exclude' in args:
        i = args.index('--exclude'); excl = set(json.load(open(args[i + 1]))); del args[i:i + 2]
    for f in args:
        rows = json.loads(open(f).readline())['rows']; both = by_key(rows)
        print(f, {q or 'all': stats(both, q, excl) for q in (None, 'leftright', 'closer')})
