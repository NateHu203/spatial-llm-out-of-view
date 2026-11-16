"""Round 6 (gates: idea-stage/GATES_round6_2026-11-06.md): head-start control. Adds two arms to round 5:
  base_box   (V1h): frame tokens only, but the S1-estimated box goes into the query and the output is box + 96 * (0.5 + r)
  global_box (V2h): + whole-frame tokens, with the same box query and output mapping
so that V3 (crop) can be compared with an arm that has the same head start and answer format but no zoom.
Round-5 heads (base, global, crop; seeds 0-2) are reused if placed in crop_runs_r6/ as head_{arm}_s{seed}.pt.
--quick: 16 scenes, 50 steps, eval9s instead of eval10.
Stages:  python crop_ablation_r6.py encode | train | score | all   [--quick] [--seeds 0,1,2]
Output:  crop_ablation_r6.json"""
import argparse, json, math, os, time
import numpy as np, torch
import crop_ablation as CA
import crop_ablation_r5 as R5

HERE = CA.HERE
ARMS = ('base', 'global', 'crop', 'base_box', 'global_box')


class HeadBox(CA.Head):
    """same layers as CA.Head; always uses the box in the query and maps the output inside the box (like the crop arm)"""
    def forward(self, base, extra, col, box):
        h = self.proj(base.float()) + self.pos_b
        if self.extra:
            h = torch.cat([h, self.proj(extra.float()) + self.pos_e], 1)
        h = self.enc(h)
        q = (self.col(col) + self.geo(box / 224))[:, None]
        o, _ = self.attn(q, h, h)
        r = self.out(o[:, 0])
        return box[:, :2] + box[:, 2:3] * (0.5 + r)


def make_head(arm):
    if arm == 'base_box':
        return HeadBox(None)
    if arm == 'global_box':
        return HeadBox('global')
    return CA.Head(None if arm == 'base' else arm)


def extra_of(arm, D, s, t):
    return (D['glob'] if arm in ('global', 'global_box') else D['crop'])[s, t]


def box_of(arm, D, s, t):
    if arm in ('crop', 'base_box', 'global_box'):
        return D['box'][s, t]
    return torch.tensor([[0., 0., 224.]]).repeat(len(s), 1)


def setup(quick):
    ev = R5.setup(quick)
    CA.RUN = os.path.join(HERE, 'crop_runs_r6_quick' if quick else 'crop_runs_r6')
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
            m = make_head(arm).cuda()
            n_steps = 50 if quick else steps
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=n_steps, pct_start=0.05)
            t0 = time.time()
            for step in range(n_steps):
                sp = CA.TRAIN_SPLITS[step % 2]; D = data[sp]; s, t = IT[sp]
                k = torch.tensor(rng.choice(len(s), min(bs, len(s)), replace=False)); s_, t_ = s[k], t[k]
                ball = D['sched'][s_, t_]
                pred = m(D['base'][s_, t_].cuda(), extra_of(arm, D, s_, t_).cuda(), D['col'][s_, ball].cuda(), box_of(arm, D, s_, t_).cuda())
                loss = (((pred - D['uv'][s_, t_].cuda()) / 224) ** 2).sum(-1).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
                if (step + 1) % 1000 == 0 or step + 1 == n_steps:
                    print(f'seed {seed}', arm, 'step', step + 1, 'rmse_px', round(224 * math.sqrt(loss.item()), 2), f'{time.time() - t0:.0f}s', flush=True)
            torch.save(m.state_dict(), path)


def make_tracks(seeds):
    for seed in seeds:
        for arm in ARMS:
            m = make_head(arm).cuda(); m.load_state_dict(torch.load(os.path.join(CA.RUN, f'head_{arm}_s{seed}.pt'))); m.eval()
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
                        preds.append(m(D['base'][s[sl], t[sl]].cuda(), extra_of(arm, D, s[sl], t[sl]).cuda(),
                                       D['col'][s[sl], ball[sl]].cuda(), box_of(arm, D, s[sl], t[sl]).cuda()).cpu())
                preds = torch.cat(preds).numpy() if preds else np.zeros((0, 2))
                out = [dict(e, pos=[[list(p) for p in f] for f in e['pos']]) for e in est]
                for (si, ti, bi), uv in zip(zip(s.tolist(), t.tolist(), ball.tolist()), preds):
                    out[si]['pos'][ti][bi] = np.round(CA.own_to_t7(CA.pix_to_own(uv), est[si], ti), 3).tolist()
                json.dump(out, open(out_path, 'w'))
            print('tracks', arm, f'seed {seed}', flush=True)


def score(quick, seeds, ev):
    from multiprocessing import get_context
    jobs = [(a, r, s, CA.RUN, list(CA.SPLITS), ev) for s in seeds for a in ARMS for r in ('ca', 'imm')]
    with get_context('spawn').Pool(min(6, os.cpu_count() or 2), maxtasksperchild=1) as p:
        res = dict(p.imap_unordered(R5._job, jobs))
    B = {k: {sp: CA._both(v['rows'][sp]) for sp in v['rows']} for k, v in res.items()}
    S = dict(quick=quick, eval_split=ev, seeds=seeds, acc={}, diff={}, per_seed={}, err={})
    for arm in ARMS:
        for rule in ('ca', 'imm'):
            S['acc'][f'{arm}/{rule}'] = {sp: R5._boot_seeds([B[(arm, rule, s)][sp] for s in seeds]) for sp in (ev, 'eval8')}
    pairs = (('crop', 'global_box'), ('crop', 'global'), ('base_box', 'base'), ('global_box', 'global'), ('global_box', 'base_box'), ('crop', 'base_box'))
    for rule in ('ca', 'imm'):
        for sp in (ev, 'eval8'):
            for a, b in pairs:
                S['diff'][f'{sp}/{rule}/{a}-{b}'] = R5._boot_seeds([B[(a, rule, s)][sp] for s in seeds], [B[(b, rule, s)][sp] for s in seeds])
                for s in seeds:
                    S['per_seed'][f'{sp}/{rule}/{a}-{b}/s{s}'] = CA._boot(B[(a, rule, s)][sp], B[(b, rule, s)][sp])
    for sp in (ev, 'eval8'):
        S['err'][sp] = {arm: {f's{s}': R5.track_err(sp, arm, s) for s in seeds} for arm in ARMS}
        for arm in ARMS:
            S['err'][sp][arm]['mean_vel_last3'] = round(float(np.mean([S['err'][sp][arm][f's{s}']['vel_last3'] for s in seeds])), 4)
    d = S['diff'][f'{ev}/ca/crop-global_box']
    S['reading'] = ('zoom adds value beyond the head start (exploratory)' if d[1][0] > 0 and d[0] >= 2.0
                    else 'no evidence that the zoom itself helps beyond the head start' if d[1][0] <= 0 else 'positive but below +2 (exploratory)')
    out = os.path.join(HERE, 'crop_ablation_r6_quick.json' if quick else 'crop_ablation_r6.json')
    json.dump(S, open(out, 'w'), indent=1, default=float)
    for a, b in pairs:
        print(f'{ev} CA {a} - {b}:', S['diff'][f'{ev}/ca/{a}-{b}'], flush=True)
    print('reading:', S['reading'])
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
