"""Round 8 (gates: idea-stage/GATES_round8_2026-11-09.md): look back at CHOSEN past frames with real zoomed looks.
Stage predict (GPU): for train9, eval10, eval12, every S1-visible sighting (t, b) gets Version 3's look (96 px square
centred on the S1 estimate, resized to 448 -> 256 tokens) read by the round-5 V3 heads (seeds 0-2, no training);
predictions are saved as floor positions in the t=7 axes -> look_runs_r8/preds_<split>.json
Stage score (CPU): each chooser picks its looks from the S1 estimates only (no labels); the chosen sightings take the
head's position; rule CA re-tuned per chooser and seed on train9; twin accuracy on eval12 (primary) and eval10.
--quick: 8 scenes of eval9s stand in for every split (never touches eval12).
Usage: python look_choice_r8.py predict | score | all  [--quick]   -> look_choice_r8(_quick).json"""
import argparse, json, os, time
import numpy as np

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
SPLITS = ('train9', 'eval10', 'eval12')
SEEDS = (0, 1, 2)
CHOOSERS = ('none', 'fixed2', 'fixed4', 'fixed8', 'last_each2', 'ends_each4', 'resid_mix4')
RUN = os.path.join(HERE, 'look_runs_r8')
QUICK_N = 8


def split_map(quick):
    return {sp: ('eval9s' if quick else sp) for sp in SPLITS}


# ------------------------------------------------------------------ choosers (label-free: S1 estimates only)
def schedule(vis):
    out = []
    for t in range(8):
        b = t % 2
        b = b if vis[t][b] else (1 - b if vis[t][1 - b] else -1)
        if b >= 0:
            out.append((t, b))
    return out


def ca_params():
    return json.load(open(os.path.join(HERE, 'state_rules_results.json')))['ca/est']['params']


def resid_ca(e, j, P):
    """distance of each sighting from its ball's fitted path: one line over all sightings; sightings at or after the
    last change found by rule CA are measured against the line of that last segment instead"""
    import state_rules as SR
    pos = np.array(e['pos'], float)[:, j]; vis = np.array(e['vis'], bool)[:, j]; s = np.nonzero(vis)[0]
    if len(s) < 3:
        return {int(t): 0.0 for t in s}
    A = np.stack([s.astype(float), np.ones(len(s))], 1); m, a = np.linalg.lstsq(A, pos[s], rcond=None)[0]
    r = {int(t): float(np.linalg.norm(pos[t] - a - m * t)) for t in s}
    st = SR.CAState(P)
    for t in range(8):
        st.update(t, vis[t], pos[t])
    if st.changes:
        seg = [t for t in s if t >= st.changes[-1] - 1]
        if len(seg) >= 3:
            A2 = np.stack([np.array(seg, float), np.ones(len(seg))], 1); m2, a2 = np.linalg.lstsq(A2, pos[seg], rcond=None)[0]
            for t in seg:
                r[int(t)] = float(np.linalg.norm(pos[t] - a2 - m2 * t))
    return r


def choose(e, P):
    cand = [(t, b) for t in range(8) for b in range(2) if e['vis'][t][b]]
    fx = schedule(e['vis'])
    s = [[t for t in range(8) if (t, j) in cand] for j in range(2)]
    last = [(s[j][-1], j) for j in range(2) if s[j]]
    R = {(t, j): v for j in range(2) for t, v in resid_ca(e, j, P).items()}
    rc = sorted(cand, key=lambda x: (-R.get(x, 0.0), -x[0], x[1]))
    return dict(none=[], fixed2=fx[-2:], fixed4=fx[-4:], fixed8=fx, last_each2=last,
                ends_each4=sorted({(s[j][i], j) for j in range(2) if s[j] for i in (0, -1)}),
                resid_mix4=list(dict.fromkeys(last + rc))[:4])


# ------------------------------------------------------------------ stage predict (GPU)
def predict(quick):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import sts, crop_ablation as CA
    p224 = AutoProcessor.from_pretrained(CA.MODEL, min_pixels=224 * 224, max_pixels=224 * 224).image_processor
    p448 = AutoProcessor.from_pretrained(CA.MODEL, min_pixels=448 * 448, max_pixels=448 * 448).image_processor
    model = AutoModelForImageTextToText.from_pretrained(CA.MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    g = torch.Generator().manual_seed(0)
    W = torch.linalg.qr(torch.randn(2048, CA.PROJ, generator=g))[0].to('cuda', torch.bfloat16)   # same projection as rounds 4-7
    heads = []
    for sd in SEEDS:
        m = CA.Head('crop').cuda(); m.load_state_dict(torch.load(os.path.join(RUN, f'head_crop_s{sd}.pt'), map_location='cuda')); m.eval()
        heads.append(m)

    def feats(proc, imgs):
        enc = proc(images=imgs, return_tensors='pt')
        with torch.no_grad():
            out = model.model.get_image_features(enc['pixel_values'].to('cuda'), enc['image_grid_thw'].to('cuda'), return_dict=True)
        return [(o.to(torch.bfloat16) @ W) for o in out.pooler_output]

    for key, sp in split_map(quick).items():
        path = os.path.join(RUN, f'preds_{key}.json')
        if os.path.exists(path):
            continue
        est = json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json')))
        n = QUICK_N if quick else len(est)
        out = {sd: [] for sd in SEEDS}; t0 = time.time(); npx = []
        for i in range(n):
            sc = sts.load(sp, i); e = est[i]
            fr = [Image.fromarray(f) for f in sc['frames']]
            cand = [(t, b) for t in range(8) for b in range(2) if e['vis'][t][b]]
            per = {sd: [[None, None] for _ in range(8)] for sd in SEEDS}
            if cand:
                base = feats(p224, fr)                                                  # 8 x (64, PROJ)
                crops, boxes = [], []
                for t, b in cand:
                    ce = CA.own_to_pix(CA.t7_to_own(np.array(e['pos'][t][b]), e, t))
                    ce = np.clip(ce, CA.CROP / 2, 224 - CA.CROP / 2)
                    x0, y0 = int(round(ce[0] - CA.CROP / 2)), int(round(ce[1] - CA.CROP / 2))
                    boxes.append([x0, y0, CA.CROP])
                    crops.append(fr[t].crop((x0, y0, x0 + CA.CROP, y0 + CA.CROP)).resize((448, 448), Image.BICUBIC))
                cf = []
                for k in range(0, len(crops), 16):
                    cf += feats(p448, crops[k:k + 16])
                Bt = torch.stack([base[t] for t, b in cand]); Ct = torch.stack(cf)
                col = torch.tensor([CA.COLORS.index(sc['names'][b]) for t, b in cand], device='cuda')
                box = torch.tensor(boxes, dtype=torch.float32, device='cuda')
                with torch.no_grad():
                    for sd, m in zip(SEEDS, heads):
                        uv = m(Bt, Ct, col, box).float().cpu().numpy()
                        for (t, b), u in zip(cand, uv):
                            per[sd][t][b] = np.round(CA.own_to_t7(CA.pix_to_own(u), e, t), 4).tolist()
            for sd in SEEDS:
                out[sd].append(per[sd])
            npx.append(len(cand))
            if i % 100 == 0:
                print(key, i, '/', n, f'{time.time() - t0:.0f}s', flush=True)
        json.dump(dict(split=sp, n=n, looks=int(sum(npx)), preds={str(k): v for k, v in out.items()}), open(path, 'w'))
        print(key, 'done', n, 'scenes', sum(npx), 'looks', f'{time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------ stage score (CPU)
def tracks_for(est, preds, chooser, P):
    out = []
    for e, pr in zip(est, preds):
        pos = [[list(p) for p in f] for f in e['pos']]
        for t, b in choose(e, P)[chooser]:
            if pr[t][b] is not None:
                pos[t][b] = pr[t][b]
        out.append(dict(e, pos=pos))
    return out


def _job(args):
    chooser, seed, quick, run = args
    global RUN
    RUN = run
    import state_rules as SR
    sm = split_map(quick); P0 = ca_params()
    cache = {}
    for key, sp in sm.items():
        D = json.load(open(os.path.join(RUN, f'preds_{key}.json')))
        est = json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json')))[:D['n']]
        cache[key] = tracks_for(est, D['preds'][str(seed)], chooser, P0)
    full = SR.scenes_of
    SR.scenes_of = lambda split: full(sm.get(split, split))[:len(cache[split])]
    SR.tracks_of = lambda split, src: cache[split]
    if chooser == 'none':
        P = P0                                                         # current system: parameters fixed earlier
    else:
        grid = SR.RULES['ca'][1]; best = None
        import itertools
        for vals in itertools.product(*grid.values()):
            Pc = dict(zip(grid.keys(), vals)); s = SR.evaluate('ca', Pc, 'train9', 'x')['both']
            if best is None or s > best[1]:
                best = (Pc, s)
        P = best[0]
    return (chooser, seed), dict(params=P, rows={k: SR.evaluate('ca', P, k, 'x', rows_out=True) for k in ('eval12', 'eval10')})


def _both(rows):
    by = {}
    for r in rows:
        if r['differs']:
            by.setdefault((r['pair'], r['t'], r['qtype']), []).append(r['ok'])
    return {k: int(all(v)) for k, v in by.items() if len(v) == 2}


def _boot_seeds(A, B=None, n=2000):
    keys = sorted(set.intersection(*[set(a) for a in A] + ([set(b) for b in B] if B else [])))
    val = {k: np.mean([a[k] - (b[k] if B else 0) for a, b in zip(A, B or [None] * len(A))]) for k in keys}
    pairs = sorted({k[0] for k in keys}); per = {p: [val[k] for k in keys if k[0] == p] for p in pairs}
    rng = np.random.default_rng(0)
    bs = [100 * np.mean([v for i in rng.integers(0, len(pairs), len(pairs)) for v in per[pairs[i]]]) for _ in range(n)]
    return round(100 * float(np.mean(list(val.values()))), 2), [round(float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)]


def look_err(quick):
    """RMS error (m, per axis) of the real looks vs the oracle, per split (seed-averaged), all visible sightings"""
    import state_rules as SR
    out = {}
    for key, sp in split_map(quick).items():
        D = json.load(open(os.path.join(RUN, f'preds_{key}.json'))); orc = SR.tracks_of(sp, 'orc')[:D['n']]
        est = json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json')))[:D['n']]
        E, E0 = [], []
        for sd in SEEDS:
            for o, e, pr in zip(orc, est, D['preds'][str(sd)]):
                for t in range(8):
                    for b in range(2):
                        if pr[t][b] is not None and o['vis'][t][b]:
                            E.append(np.array(pr[t][b]) - o['pos'][t][b])
                            if sd == 0:
                                E0.append(np.array(e['pos'][t][b]) - o['pos'][t][b])
        out[key] = dict(look_rms=round(float(np.sqrt(np.mean(np.square(E)))), 4), s1_rms=round(float(np.sqrt(np.mean(np.square(E0)))), 4))
    return out


def score(quick):
    from multiprocessing import get_context
    jobs = [(c, s, quick, RUN) for c in CHOOSERS for s in SEEDS]
    with get_context('spawn').Pool(min(8, os.cpu_count() or 2), maxtasksperchild=1) as p:
        res = dict(p.imap_unordered(_job, jobs))
    B = {k: {sp: _both(v['rows'][sp]) for sp in v['rows']} for k, v in res.items()}
    S = dict(quick=quick, acc={}, diff={}, params={f'{c}/s{s}': res[(c, s)]['params'] for c, s in res}, look_err=look_err(quick))
    for c in CHOOSERS:
        for sp in ('eval12', 'eval10'):
            S['acc'][f'{sp}/{c}'] = _boot_seeds([B[(c, s)][sp] for s in SEEDS])
    comps = [('resid_mix4', 'fixed8'), ('last_each2', 'fixed2'), ('ends_each4', 'fixed8'), ('resid_mix4', 'none'),
             ('fixed8', 'none'), ('resid_mix4', 'ends_each4'), ('ends_each4', 'fixed4'), ('resid_mix4', 'fixed4')]
    for a, b in comps:
        for sp in ('eval12', 'eval10'):
            S['diff'][f'{sp}/{a}-{b}'] = _boot_seeds([B[(a, s)][sp] for s in SEEDS], [B[(b, s)][sp] for s in SEEDS])
    d = S['diff']
    S['gates'] = {'L1_resid_mix4_beats_fixed8': d['eval12/resid_mix4-fixed8'][1][0] > 0,
                  'L2_last_each2_beats_fixed2': d['eval12/last_each2-fixed2'][1][0] > 0,
                  'L3_ends_each4_noninferior_fixed8': d['eval12/ends_each4-fixed8'][1][0] > -3,
                  'L4_resid_mix4_beats_current': d['eval12/resid_mix4-none'][1][0] > 0}
    S['lookback_works'] = S['gates']['L1_resid_mix4_beats_fixed8'] and S['gates']['L4_resid_mix4_beats_current']
    json.dump(S, open(os.path.join(HERE, 'look_choice_r8_quick.json' if quick else 'look_choice_r8.json'), 'w'), indent=1, default=float)
    print(json.dumps(S['gates'], indent=1)); print('look-back works:', S['lookback_works'])
    for k, v in S['diff'].items():
        print(f'{k:32s}', v)
    for k, v in S['acc'].items():
        print(f'{k:22s}', v)
    print('look error:', S['look_err'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['predict', 'score', 'all'])
    ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
    if a.quick:
        import shutil
        q = os.path.join(HERE, 'look_runs_r8_quick'); os.makedirs(q, exist_ok=True)
        for sd in SEEDS:
            src = os.path.join(RUN, f'head_crop_s{sd}.pt')
            if os.path.exists(src) and not os.path.exists(os.path.join(q, f'head_crop_s{sd}.pt')):
                shutil.copy(src, q)
        RUN = q
    os.makedirs(RUN, exist_ok=True)
    if a.stage in ('predict', 'all'):
        predict(a.quick)
    if a.stage in ('score', 'all'):
        score(a.quick)
