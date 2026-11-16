"""Round 9 (gates: idea-stage/GATES_round9_2026-11-09.md): spend zoomed looks where they move the forecast.
Weighted ends (Hoel & Levine 1964: 2 looks at the last end : 1 at the first, per ball, within the current motion
segment) vs Version 3's schedule; plus one closed-loop agent arm (adapt6) that reads its first looks before choosing
the rest. Reuses round 8: zoom readings (96 px square on the S1 estimate, 448 px, 3 round-5 heads) for every
S1-visible sighting; chosen sightings take the reading; rule CA re-tuned on train9 per chooser and seed.
eval13 decides; eval10 / eval12 are a screen (already scored in round 8).
Usage: python look_choice_r9.py predict   (GPU, eval13 readings -> look_runs_r9/preds_eval13.json)
       python look_choice_r9.py score [--screen]   (CPU; --screen = eval10/eval12 only) -> look_choice_r9(_screen).json"""
import argparse, json, os, shutil
import numpy as np
import look_choice_r8 as L

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
RUN = os.path.join(HERE, 'look_runs_r9')
R8 = L.RUN   # repo: round-8 readings (preds_<split>.json) live in the round-8 run folder of the workspace
SEEDS = (0, 1, 2)
CHOOSERS = ('none', 'fixed2', 'fixed4', 'fixed6', 'fixed8', 'last_each2', 'ends_each4', 'wends6', 'adapt6')


def segment(e, j, P):
    import state_rules as SR
    vis = np.array(e['vis'], bool)[:, j]; pos = np.array(e['pos'], float)[:, j]; s = [int(t) for t in np.nonzero(vis)[0]]
    st = SR.CAState(P)
    for t in range(8):
        st.update(t, vis[t], pos[t])
    if st.changes:
        seg = [t for t in s if t >= st.changes[-1] - 1]
        if len(seg) >= 3:
            return seg
    return s


def _fill(picked, e, k):
    out = list(dict.fromkeys(picked))
    for x in reversed(L.schedule(e['vis'])):
        if len(out) >= k:
            break
        if x not in out:
            out.append(x)
    for t in range(7, -1, -1):                       # then any latest unused sighting
        for b in range(2):
            if len(out) < k and e['vis'][t][b] and (t, b) not in out:
                out.append((t, b))
    return out[:k]


def choose9(e, P, pr):
    """pr: this seed's zoom readings [t][b] (used by adapt6 only, and only at sightings it has already looked at)"""
    ch = L.choose(e, P); fx = L.schedule(e['vis'])
    ch['fixed6'] = _fill(fx[-6:], e, 6)                # filled to 6 when the schedule is shorter (equal budget)
    segs = [segment(e, j, P) for j in range(2)]
    w = []
    for j, s in enumerate(segs):
        if s:
            w += [(s[-1], j)] + ([(s[-2], j)] if len(s) >= 2 else []) + [(s[0], j)]
    ch['wends6'] = _fill(w, e, 6)
    first = [x for j, s in enumerate(segs) if s for x in ((s[-1], j), (s[0], j))]
    dis = {}
    for j, s in enumerate(segs):
        if s and pr[s[-1]][j] is not None:
            dis[j] = float(np.linalg.norm(np.array(pr[s[-1]][j]) - e['pos'][s[-1]][j]))
    extra = []
    if dis:
        j = max(dis, key=dis.get); s = segs[j]
        extra = [(t, j) for t in s[:-1][::-1] if (t, j) not in first][:2]
    ch['adapt6'] = _fill(first + extra, e, 6)
    return ch


def tracks_for(est, preds, chooser, P):
    out = []
    for e, pr in zip(est, preds):
        pos = [[list(p) for p in f] for f in e['pos']]
        for t, b in choose9(e, P, pr)[chooser]:
            if pr[t][b] is not None:
                pos[t][b] = pr[t][b]
        out.append(dict(e, pos=pos))
    return out


def _job(args):
    chooser, seed, splits = args
    import itertools
    import state_rules as SR
    P0 = L.ca_params(); cache = {}
    for sp in ('train9',) + splits:
        D = json.load(open(os.path.join(RUN, f'preds_{sp}.json')))
        est = json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json')))[:D['n']]
        cache[sp] = tracks_for(est, D['preds'][str(seed)], chooser, P0)
    SR.tracks_of = lambda split, src: cache[split]
    if chooser == 'none':
        P = P0
    else:
        grid = SR.RULES['ca'][1]; best = None
        for vals in itertools.product(*grid.values()):
            Pc = dict(zip(grid.keys(), vals)); s = SR.evaluate('ca', Pc, 'train9', 'x')['both']
            if best is None or s > best[1]:
                best = (Pc, s)
        P = best[0]
    nlook = {sp: float(np.mean([len(choose9(e, P0, pr)[chooser]) for e, pr in
                                zip(json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json'))),
                                    json.load(open(os.path.join(RUN, f'preds_{sp}.json')))['preds'][str(seed)])])) for sp in splits}
    return (chooser, seed), dict(params=P, nlook=nlook, rows={k: SR.evaluate('ca', P, k, 'x', rows_out=True) for k in splits})


def score(screen):
    from multiprocessing import get_context
    splits = ('eval12', 'eval10') if screen else ('eval13', 'eval12', 'eval10')
    jobs = [(c, s, splits) for c in CHOOSERS for s in SEEDS]
    with get_context('spawn').Pool(min(8, os.cpu_count() or 2), maxtasksperchild=1) as p:
        res = dict(p.imap_unordered(_job, jobs))
    B = {k: {sp: L._both(v['rows'][sp]) for sp in v['rows']} for k, v in res.items()}
    S = dict(screen=screen, acc={}, diff={}, looks={}, params={f'{c}/s{s}': res[(c, s)]['params'] for c, s in res})
    for c in CHOOSERS:
        for sp in splits:
            S['acc'][f'{sp}/{c}'] = L._boot_seeds([B[(c, s)][sp] for s in SEEDS])
            S['looks'][f'{sp}/{c}'] = round(float(np.mean([res[(c, s)]['nlook'][sp] for s in SEEDS])), 2)
    comps = [('wends6', 'fixed6'), ('wends6', 'fixed8'), ('wends6', 'last_each2'), ('adapt6', 'wends6'),
             ('wends6', 'ends_each4'), ('adapt6', 'fixed6'), ('wends6', 'none'), ('last_each2', 'fixed2')]
    for a, b in comps:
        for sp in splits:
            S['diff'][f'{sp}/{a}-{b}'] = L._boot_seeds([B[(a, s)][sp] for s in SEEDS], [B[(b, s)][sp] for s in SEEDS])
    if not screen:
        d = S['diff']
        S['gates'] = {'W1_wends6_beats_fixed6': d['eval13/wends6-fixed6'][1][0] > 0,
                      'W2_wends6_noninferior_fixed8': d['eval13/wends6-fixed8'][1][0] > -2,
                      'W3_wends6_beats_last_each2': d['eval13/wends6-last_each2'][1][0] > 0,
                      'A1_adapt6_beats_wends6': d['eval13/adapt6-wends6'][1][0] > 0}
        S['weighted_ends_works'] = S['gates']['W1_wends6_beats_fixed6'] and S['gates']['W2_wends6_noninferior_fixed8']
        print(json.dumps(S['gates'], indent=1)); print('weighted ends works:', S['weighted_ends_works'])
    json.dump(S, open(os.path.join(HERE, 'look_choice_r9_screen.json' if screen else 'look_choice_r9.json'), 'w'), indent=1, default=float)
    for k, v in S['diff'].items():
        print(f'{k:30s}', v)
    for k, v in S['acc'].items():
        print(f'{k:22s}', v, 'looks', S['looks'][k])


def prepare():
    os.makedirs(RUN, exist_ok=True)
    for sd in SEEDS:
        f = os.path.join(L.RUN, f'head_crop_s{sd}.pt')
        if not os.path.exists(os.path.join(RUN, os.path.basename(f))):
            shutil.copy(f, RUN)
    for sp in ('train9', 'eval10', 'eval12'):
        if not os.path.exists(os.path.join(RUN, f'preds_{sp}.json')):
            shutil.copy(os.path.join(R8, f'preds_{sp}.json'), RUN)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['predict', 'score'])
    ap.add_argument('--screen', action='store_true'); a = ap.parse_args()
    prepare()
    if a.stage == 'predict':
        L.SPLITS = ('eval13',); L.RUN = RUN
        L.predict(False)
    else:
        score(a.screen)
