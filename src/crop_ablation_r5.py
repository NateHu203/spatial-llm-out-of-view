"""Round 5 (gates: idea-stage/GATES_round5_2026-11-06.md): round-4 crop vs global ablation, re-tested once on the sealed
fresh twins eval10 with 3 head seeds. Reuses round-4 code (crop_ablation.py) for geometry, encoding and the head.
Seed 0 = the round-4 heads if they are placed in crop_runs_r5/ as head_{arm}_s0.pt (same recipe); seeds 1, 2 are trained.
--quick: 16 scenes, 50 steps, and eval9s INSTEAD of eval10 (so the smoke test never touches eval10).
Stages:  python crop_ablation_r5.py encode | train | score | all   [--quick] [--seeds 0,1,2]
Output:  crop_ablation_r5.json (main), crop_runs_r5/ (features, heads, tracks)"""
import argparse, json, math, os, time
import numpy as np, torch
import crop_ablation as CA

HERE = CA.HERE
ARMS = CA.ARMS


def setup(quick):
    ev = 'eval9s' if quick else 'eval10'
    CA.SPLITS = {'train8': 1200, 'train9': 600, 'eval8': 600, ev: 600}
    CA.RUN = os.path.join(HERE, 'crop_runs_r5_quick' if quick else 'crop_runs_r5')
    os.makedirs(CA.RUN, exist_ok=True)
    return ev


def train_heads(quick, seeds, steps=4000, bs=256):
    data = {sp: torch.load(os.path.join(CA.RUN, f'{sp}_feats.pt')) for sp in CA.TRAIN_SPLITS}
    IT = {sp: CA.items(D) for sp, D in data.items()}
    for seed in seeds:
        for arm in ARMS:
            path = os.path.join(CA.RUN, f'head_{arm}_s{seed}.pt')
            if os.path.exists(path):
                print('reuse', path, flush=True); continue
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            m = CA.Head(None if arm == 'base' else arm).cuda()
            n_steps = 50 if quick else steps
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=n_steps, pct_start=0.05)
            t0 = time.time()
            for step in range(n_steps):
                sp = CA.TRAIN_SPLITS[step % 2]; D = data[sp]; s, t = IT[sp]
                k = torch.tensor(rng.choice(len(s), min(bs, len(s)), replace=False)); s_, t_ = s[k], t[k]
                ball = D['sched'][s_, t_]
                extra = (D['glob'] if arm == 'global' else D['crop'])[s_, t_].cuda()
                box = D['box'][s_, t_].cuda() if arm == 'crop' else torch.tensor([[0., 0., 224.]]).repeat(len(s_), 1).cuda()
                pred = m(D['base'][s_, t_].cuda(), extra, D['col'][s_, ball].cuda(), box)
                loss = (((pred - D['uv'][s_, t_].cuda()) / 224) ** 2).sum(-1).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
                if (step + 1) % 1000 == 0 or step + 1 == n_steps:
                    print(f'seed {seed}', arm, 'step', step + 1, 'rmse_px', round(224 * math.sqrt(loss.item()), 2),
                          f'{time.time() - t0:.0f}s', flush=True)
            torch.save(m.state_dict(), path)


def make_tracks(seeds):
    for seed in seeds:
        for arm in ARMS:
            m = CA.Head(None if arm == 'base' else arm).cuda()
            m.load_state_dict(torch.load(os.path.join(CA.RUN, f'head_{arm}_s{seed}.pt'))); m.eval()
            for sp in CA.SPLITS:
                out_path = os.path.join(CA.RUN, f'{sp}_{arm}_s{seed}_tracks.json')
                if os.path.exists(out_path):
                    continue
                D = torch.load(os.path.join(CA.RUN, f'{sp}_feats.pt')); n = D['sched'].shape[0]
                est = json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json')))[:n]
                s, t = torch.nonzero(D['sched'] >= 0, as_tuple=True); ball = D['sched'][s, t]
                preds = []
                with torch.no_grad():
                    for k in range(0, len(s), 512):
                        sl = slice(k, k + 512)
                        extra = (D['glob'] if arm == 'global' else D['crop'])[s[sl], t[sl]].cuda()
                        box = D['box'][s[sl], t[sl]].cuda() if arm == 'crop' else torch.tensor([[0., 0., 224.]]).repeat(len(s[sl]), 1).cuda()
                        preds.append(m(D['base'][s[sl], t[sl]].cuda(), extra, D['col'][s[sl], ball[sl]].cuda(), box).cpu())
                preds = torch.cat(preds).numpy() if preds else np.zeros((0, 2))
                out = [dict(e, pos=[[list(p) for p in f] for f in e['pos']]) for e in est]
                for (si, ti, bi), uv in zip(zip(s.tolist(), t.tolist(), ball.tolist()), preds):
                    out[si]['pos'][ti][bi] = np.round(CA.own_to_t7(CA.pix_to_own(uv), est[si], ti), 3).tolist()
                json.dump(out, open(out_path, 'w'))
            print('tracks', arm, f'seed {seed}', flush=True)


def _job(args):
    arm, rule, seed, run, splits, ev = args
    import state_rules as SR
    SR.RULES['imm'] = (SR.RULES['imm'][0], dict(SR.RULES['imm'][1], q_lo=[1e-4]))
    cache = {sp: json.load(open(os.path.join(run, f'{sp}_{arm}_s{seed}_tracks.json'))) for sp in splits}
    full = SR.scenes_of
    SR.scenes_of = lambda split: full(split)[:len(cache[split])]
    SR.tracks_of = lambda split, src: cache[split]
    P, _ = SR.tune(rule, 'x')
    return (arm, rule, seed), dict(params=P, rows={sp: SR.evaluate(rule, P, sp, 'x', rows_out=True) for sp in (ev, 'eval8')})


def _boot_seeds(A, B=None, n=2000):
    """A, B: lists over seeds of {case: 0/1}. Per case, mean over seeds of (A - B); bootstrap over scene pairs (each pair
    resampled jointly across seeds). Returns (mean in points, [CI low, CI high])."""
    keys = sorted(set.intersection(*[set(a) for a in A] + ([set(b) for b in B] if B else [])))
    val = {k: np.mean([a[k] - (b[k] if B else 0) for a, b in zip(A, B or [None] * len(A))]) for k in keys}
    pairs = sorted({k[0] for k in keys}); per = {p: [val[k] for k in keys if k[0] == p] for p in pairs}
    rng = np.random.default_rng(0)
    bs = [100 * np.mean([v for i in rng.integers(0, len(pairs), len(pairs)) for v in per[pairs[i]]]) for _ in range(n)]
    return round(100 * float(np.mean(list(val.values()))), 2), [round(float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)]


def track_err(sp, arm, seed):
    import state_rules as SR
    T = json.load(open(os.path.join(CA.RUN, f'{sp}_{arm}_s{seed}_tracks.json'))); orc = SR.tracks_of(sp, 'orc')[:len(T)]
    e1, ep, ev = [], [], []
    for o, t in zip(orc, T):
        po, pt = np.array(o['pos']), np.array(t['pos']); vis = np.array(t['vis'], bool) & np.array(o['vis'], bool)
        for j in range(2):
            fr = np.nonzero(vis[:, j])[0]
            e1 += [(pt[s, j] - pt[r, j]) - (po[s, j] - po[r, j]) for r, s in zip(fr[:-1], fr[1:])]
            ep += list(pt[fr, j] - po[fr, j])
            if len(fr) >= 3:
                k = fr[-3:]; A = np.stack([k, np.ones(3)], 1)
                ev.append(np.linalg.lstsq(A, pt[k, j], rcond=None)[0][0] - np.linalg.lstsq(A, po[k, j], rcond=None)[0][0])
    f = lambda a: round(float(np.sqrt(np.mean(np.sum(np.array(a) ** 2, -1)))), 4) if len(a) else None
    return dict(disp=f(e1), pos=f(ep), vel_last3=f(ev))


def score(quick, seeds, ev):
    from multiprocessing import get_context
    jobs = [(a, r, s, CA.RUN, list(CA.SPLITS), ev) for s in seeds for a in ARMS for r in ('ca', 'imm')]
    with get_context('spawn').Pool(min(6, os.cpu_count() or 2), maxtasksperchild=1) as p:
        res = dict(p.imap_unordered(_job, jobs))
    B = {k: {sp: CA._both(v['rows'][sp]) for sp in v['rows']} for k, v in res.items()}
    S = dict(quick=quick, eval_split=ev, seeds=seeds, acc={}, diff={}, per_seed={}, err={}, gates={})
    for arm in ARMS:
        for rule in ('ca', 'imm'):
            S['acc'][f'{arm}/{rule}'] = {sp: _boot_seeds([B[(arm, rule, s)][sp] for s in seeds]) for sp in (ev, 'eval8')}
            for s in seeds:
                S['per_seed'][f'{arm}/{rule}/s{s}'] = {sp: CA._boot(B[(arm, rule, s)][sp]) for sp in (ev, 'eval8')}
    for rule in ('ca', 'imm'):
        for sp in (ev, 'eval8'):
            for a, b in (('crop', 'global'), ('crop', 'base'), ('global', 'base')):
                S['diff'][f'{sp}/{rule}/{a}-{b}'] = _boot_seeds([B[(a, rule, s)][sp] for s in seeds], [B[(b, rule, s)][sp] for s in seeds])
                for s in seeds:
                    S['per_seed'][f'{sp}/{rule}/{a}-{b}/s{s}'] = CA._boot(B[(a, rule, s)][sp], B[(b, rule, s)][sp])
    for arm in ARMS:   # does the zoom make the change-aware rule pull ahead of IMM?
        S['diff'][f'{ev}/{arm}/ca-imm'] = _boot_seeds([B[(arm, 'ca', s)][ev] for s in seeds], [B[(arm, 'imm', s)][ev] for s in seeds])
    for sp in (ev, 'eval8'):
        S['err'][sp] = {arm: {f's{s}': track_err(sp, arm, s) for s in seeds} for arm in ARMS}
        for arm in ARMS:
            S['err'][sp][arm]['mean_vel_last3'] = round(float(np.mean([S['err'][sp][arm][f's{s}']['vel_last3'] for s in seeds])), 4)
    a = S['diff'][f'{ev}/ca/crop-global']
    c8, g8 = S['acc']['crop/ca']['eval8'][0], S['acc']['global/ca']['eval8'][0]
    vc, vg = S['err'][ev]['crop']['mean_vel_last3'], S['err'][ev]['global']['mean_vel_last3']
    S['gates'] = {'E5-a': dict(value=a, need='>= +5.0 and CI low > 0', passed=bool(a[0] >= 5.0 and a[1][0] > 0)),
                  'E5-b': dict(value=[c8, g8], need='crop >= global - 2', passed=bool(c8 >= g8 - 2)),
                  'E5-c': dict(value=[vc, vg], need='crop < global', passed=bool(vc < vg))}
    out = os.path.join(HERE, 'crop_ablation_r5_quick.json' if quick else 'crop_ablation_r5.json')
    json.dump(S, open(out, 'w'), indent=1, default=float)
    for k, v in S['gates'].items():
        print(k, v, flush=True)
    print('per-seed', ev, 'CA crop-global:', [S['per_seed'][f'{ev}/ca/crop-global/s{s}'] for s in seeds])
    for k, v in S['acc'].items():
        print(k, ev, v[ev], 'eval8', v['eval8'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['encode', 'train', 'score', 'all'])
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(',')]
    ev = setup(a.quick)
    if a.stage in ('encode', 'all'):
        CA.encode(a.quick)
    if a.stage in ('train', 'all'):
        train_heads(a.quick, seeds); make_tracks(seeds)
    if a.stage in ('score', 'all'):
        score(a.quick, seeds, ev)
