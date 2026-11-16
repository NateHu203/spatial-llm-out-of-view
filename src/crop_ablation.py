"""Round 4 (reviewer's round-3 design; gates in idea-stage/GATES_round4_2026-11-05.md): does a close-up look measure a
ball better than an equally expensive wide look?

Schedule (identical in every arm, causal): at frame t look at ball t % 2 if the estimator says it is visible, else the
other ball if visible, else no extra call.
Every arm sees the shared 224x224 frame encoding (64 merged tokens). Extra call per scheduled frame:
  base   : none                                       (same head, base tokens only)
  global : the whole frame resized to 448x448         (256 merged tokens)
  crop   : a CROP x CROP px box centred on the ESTIMATED ball position (deployable), resized to 448x448 (256 tokens)
A head of the same size in every arm predicts the ball centre in full-frame pixels; exact pinhole back-projection gives
its floor position in the frame's own camera axes, then exact odometry maps it to the t=7 axes. Only the scheduled
(ball, frame) positions are replaced in the existing estimated tracks; everything else is shared by all arms.
Then CA and IMM are re-tuned per arm on train8+train9 and scored on eval9s / eval8 (pair accuracy), with paired
scene-pair bootstrap CIs, and track errors are measured.
Stages:  python crop_ablation.py encode | train | score | all     [--quick]
Outputs: crop_runs/ (features, heads, tracks), crop_ablation.json"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
RUN = os.path.join(HERE, 'crop_runs')
MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
SPLITS = {'train8': 1200, 'train9': 600, 'eval8': 600, 'eval9s': 600}
TRAIN_SPLITS = ('train8', 'train9')
ARMS = ('base', 'global', 'crop')
CROP = 96              # px box in the 224 frame (ball radius is 8-40 px)
PROJ = 384             # fixed random projection of the 2048-d vision features (same matrix for every arm)
COLORS = ['red', 'blue', 'green', 'yellow']


# ------------------------------------------------------------------ geometry (exact; matches scenes.render)
def _cam():
    import scenes
    s, c = math.sin(scenes.PITCH), math.cos(scenes.PITCH)
    fwd = np.array([0.0, -s, c]); right = np.array([1.0, 0.0, 0.0]); up = np.cross(fwd, right)
    return scenes, fwd, right, up


def own_to_pix(p):
    """floor point (x right, z forward) in the frame's own camera axes -> continuous pixel (u, v) of the ball centre"""
    scenes, fwd, right, up = _cam()
    d = np.array([p[0], scenes.R_OBJ - scenes.H_CAM, p[1]]); zc = d @ fwd
    return np.array([scenes.FOCAL * (d @ right) / zc + scenes.IMG / 2, -scenes.FOCAL * (d @ up) / zc + scenes.IMG / 2])


def pix_to_own(uv):
    scenes, fwd, right, up = _cam()
    a = (uv[0] - scenes.IMG / 2) / scenes.FOCAL; b = (uv[1] - scenes.IMG / 2) / scenes.FOCAL
    D = fwd + a * right - b * up
    if D[1] >= -1e-6:
        D[1] = -1e-6
    lam = (scenes.R_OBJ - scenes.H_CAM) / D[1]
    return np.array([lam * D[0], lam * D[2]])


def pose(tr, t):
    import scenes
    scn = dict(yaw0=0.0, yaw_rate=tr['yaw_rate'], cam_speed=tr['cam_speed'])
    return scenes.cam_pose(scn, float(t))


def t7_to_own(p7, tr, t):
    import scenes
    c7, y7 = pose(tr, 7); c, y = pose(tr, t)
    r7, f7 = np.array([math.cos(y7), -math.sin(y7)]), np.array([math.sin(y7), math.cos(y7)])
    return scenes.to_cam(c7 + p7[0] * r7 + p7[1] * f7, c, y)


def own_to_t7(po, tr, t):
    import scenes
    c7, y7 = pose(tr, 7); c, y = pose(tr, t)
    r, f = np.array([math.cos(y), -math.sin(y)]), np.array([math.sin(y), math.cos(y)])
    return scenes.to_cam(c + po[0] * r + po[1] * f, c7, y7)


# ------------------------------------------------------------------ stage 1: encode
def schedule(vis):
    out = np.full(8, -1)
    for t in range(8):
        b = t % 2
        out[t] = b if vis[t][b] else (1 - b if vis[t][1 - b] else -1)
    return out


def encode(quick):
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import sts
    p224 = AutoProcessor.from_pretrained(MODEL, min_pixels=224 * 224, max_pixels=224 * 224).image_processor
    p448 = AutoProcessor.from_pretrained(MODEL, min_pixels=448 * 448, max_pixels=448 * 448).image_processor
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    g = torch.Generator().manual_seed(0)
    W = torch.linalg.qr(torch.randn(2048, PROJ, generator=g))[0].to('cuda', torch.bfloat16)   # fixed orthonormal projection

    def feats(proc, imgs):
        enc = proc(images=imgs, return_tensors='pt')
        with torch.no_grad():
            out = model.model.get_image_features(enc['pixel_values'].to('cuda'), enc['image_grid_thw'].to('cuda'), return_dict=True)
        return [(o.to(torch.bfloat16) @ W).cpu() for o in out.pooler_output]

    rng = np.random.default_rng(0)
    os.makedirs(RUN, exist_ok=True)
    for sp, n in SPLITS.items():
        n = 16 if quick else n
        path = os.path.join(RUN, f'{sp}_feats.pt')
        if os.path.exists(path):
            continue
        est = json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json')))
        t0 = time.time(); S, BOX, UV, COL = [], [], [], []
        B = torch.zeros(n, 8, 64, PROJ, dtype=torch.bfloat16)
        G = torch.zeros(n, 8, 256, PROJ, dtype=torch.bfloat16); C = torch.zeros_like(G)
        for i in range(n):
            sc = sts.load(sp, i); tr = sts.scene_tracks(sp, i, sc); e = est[i]
            sch = schedule(e['vis']); fr = [Image.fromarray(f) for f in sc['frames']]
            B[i] = torch.stack(feats(p224, fr))                                      # (8, 64, PROJ)
            glob, crops, boxes, uvs = [], [], [], []
            for t in range(8):
                b = sch[t]
                if b < 0:
                    boxes.append([0, 0, 224]); uvs.append([np.nan, np.nan]); continue
                ce = own_to_pix(t7_to_own(np.array(e['pos'][t][b]), tr, t))
                if sp == 'train8':                                                   # estimator saw train8 in training:
                    ce = ce + rng.normal(0, 6, 2)                                    # widen its centre error to eval size
                ce = np.clip(ce, CROP / 2, 224 - CROP / 2)
                x0, y0 = int(round(ce[0] - CROP / 2)), int(round(ce[1] - CROP / 2))
                boxes.append([x0, y0, CROP])
                crops.append(fr[t].crop((x0, y0, x0 + CROP, y0 + CROP)).resize((448, 448), Image.BICUBIC))
                glob.append(fr[t].resize((448, 448), Image.BICUBIC))
                uvs.append(own_to_pix(t7_to_own(np.array(tr['pos'][t][b]), tr, t)) if tr['vis'][t][b] else [np.nan, np.nan])
            idx = [t for t in range(8) if sch[t] >= 0]
            if idx:
                for t, f in zip(idx, feats(p448, glob)):
                    G[i, t] = f
                for t, f in zip(idx, feats(p448, crops)):
                    C[i, t] = f
            S.append(sch); BOX.append(boxes); UV.append(uvs)
            COL.append([COLORS.index(c) for c in sc['names']])
            if i % 100 == 0:
                print(sp, i, '/', n, f'{time.time() - t0:.0f}s', flush=True)
        torch.save(dict(base=B, glob=G, crop=C, sched=torch.tensor(np.array(S)),
                        box=torch.tensor(np.array(BOX), dtype=torch.float32), uv=torch.tensor(np.array(UV, float), dtype=torch.float32),
                        col=torch.tensor(COL)), path)
        print(sp, 'encoded', n, f'{time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------ stage 2: heads
class Head(nn.Module):
    def __init__(self, extra, d=256, layers=2):
        super().__init__()
        self.extra = extra
        self.proj = nn.Sequential(nn.Linear(PROJ, d), nn.LayerNorm(d))
        self.pos_b = nn.Parameter(torch.randn(64, d) * 0.02); self.pos_e = nn.Parameter(torch.randn(256, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), layers)
        self.col = nn.Embedding(len(COLORS), d); self.geo = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, base, extra, col, box):
        h = self.proj(base.float()) + self.pos_b
        if self.extra:
            h = torch.cat([h, self.proj(extra.float()) + self.pos_e], 1)
        h = self.enc(h)
        q = (self.col(col) + self.geo(box / 224))[:, None]
        o, _ = self.attn(q, h, h)
        r = self.out(o[:, 0])
        if self.extra == 'crop':                      # crop arm predicts inside its box, mapped back to frame pixels
            return box[:, :2] + box[:, 2:3] * (0.5 + r)
        return 224 * (0.5 + r)


def items(D):
    """scheduled (scene, frame) with a visible target -> index tensors"""
    s, t = torch.nonzero(D['sched'] >= 0, as_tuple=True)
    ok = ~torch.isnan(D['uv'][s, t, 0])
    return s[ok], t[ok]


def train_heads(quick, steps=4000, bs=256):
    data = {sp: torch.load(os.path.join(RUN, f'{sp}_feats.pt')) for sp in TRAIN_SPLITS}
    for arm in ARMS:
        path = os.path.join(RUN, f'head_{arm}.pt')
        if os.path.exists(path):
            continue
        torch.manual_seed(0); rng = np.random.default_rng(0)
        m = Head(None if arm == 'base' else arm).cuda()
        n_steps = 50 if quick else steps
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=n_steps, pct_start=0.05)
        IT = {sp: items(D) for sp, D in data.items()}
        t0 = time.time()
        for step in range(n_steps):
            sp = TRAIN_SPLITS[step % 2]; D = data[sp]; s, t = IT[sp]
            k = torch.tensor(rng.choice(len(s), min(bs, len(s)), replace=False)); s_, t_ = s[k], t[k]
            ball = D['sched'][s_, t_]
            extra = (D['glob'] if arm == 'global' else D['crop'])[s_, t_].cuda()
            box = D['box'][s_, t_].cuda() if arm == 'crop' else torch.tensor([[0., 0., 224.]]).repeat(len(s_), 1).cuda()
            pred = m(D['base'][s_, t_].cuda(), extra, D['col'][s_, ball].cuda(), box)
            loss = (((pred - D['uv'][s_, t_].cuda()) / 224) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
            if (step + 1) % 500 == 0 or step + 1 == n_steps:
                print(arm, 'step', step + 1, 'rmse_px', round(224 * math.sqrt(loss.item()), 2), f'{time.time() - t0:.0f}s', flush=True)
        torch.save(m.state_dict(), path)


def make_tracks(quick):
    import sts
    for arm in ARMS:
        m = Head(None if arm == 'base' else arm).cuda(); m.load_state_dict(torch.load(os.path.join(RUN, f'head_{arm}.pt'))); m.eval()
        for sp, n in SPLITS.items():
            D = torch.load(os.path.join(RUN, f'{sp}_feats.pt')); n = D['sched'].shape[0]
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
                out[si]['pos'][ti][bi] = np.round(own_to_t7(pix_to_own(uv), est[si], ti), 3).tolist()
            json.dump(out, open(os.path.join(RUN, f'{sp}_{arm}_tracks.json'), 'w'))
        print('tracks', arm, flush=True)


# ------------------------------------------------------------------ stage 3: tune + score (CPU, fresh process per job)
def _job(args):
    arm, rule, run = args
    import state_rules as SR
    global RUN
    RUN = run
    SR.RULES['imm'] = (SR.RULES['imm'][0], dict(SR.RULES['imm'][1], q_lo=[1e-4]))
    cache = {sp: json.load(open(os.path.join(RUN, f'{sp}_{arm}_tracks.json'))) for sp in SPLITS}
    full = SR.scenes_of
    SR.scenes_of = lambda split: full(split)[:len(cache[split])]
    SR.tracks_of = lambda split, src: cache[split]
    P, _ = SR.tune(rule, 'x')
    return (arm, rule), dict(params=P, rows={sp: SR.evaluate(rule, P, sp, 'x', rows_out=True) for sp in ('eval9s', 'eval8')})


def _both(rows):
    by = {}
    for r in rows:
        if r['differs']:
            by.setdefault((r['pair'], r['t'], r['qtype']), []).append(r['ok'])
    return {k: int(all(v)) for k, v in by.items() if len(v) == 2}


def _boot(a, b=None, n=2000):
    keys = sorted(set(a) & set(b)) if b is not None else sorted(a)
    val = {k: a[k] - (b[k] if b is not None else 0) for k in keys}
    pairs = sorted({k[0] for k in keys}); per = {p: [val[k] for k in keys if k[0] == p] for p in pairs}
    rng = np.random.default_rng(0)
    bs = [100 * np.mean([v for i in rng.integers(0, len(pairs), len(pairs)) for v in per[pairs[i]]]) for _ in range(n)]
    return round(100 * np.mean(list(val.values())), 2), [round(float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)]


def track_err(sp, arm):
    import state_rules as SR
    T = json.load(open(os.path.join(RUN, f'{sp}_{arm}_tracks.json'))); orc = SR.tracks_of(sp, 'orc')[:len(T)]
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


def score(quick):
    from multiprocessing import get_context
    jobs = [(a, r, RUN) for a in ARMS for r in ('ca', 'imm')]
    with get_context('spawn').Pool(min(6, os.cpu_count() or 2), maxtasksperchild=1) as p:
        res = dict(p.imap_unordered(_job, jobs))
    S = dict(quick=quick, acc={}, diff={}, err={})
    B = {k: {sp: _both(v['rows'][sp]) for sp in v['rows']} for k, v in res.items()}
    for (arm, rule), b in B.items():
        S['acc'][f'{arm}/{rule}'] = dict(params=res[(arm, rule)]['params'], **{sp: _boot(b[sp]) for sp in b})
    for rule in ('ca', 'imm'):
        for sp in ('eval9s', 'eval8'):
            S['diff'][f'{sp}/{rule}/crop-global'] = _boot(B[('crop', rule)][sp], B[('global', rule)][sp])
            S['diff'][f'{sp}/{rule}/crop-base'] = _boot(B[('crop', rule)][sp], B[('base', rule)][sp])
            S['diff'][f'{sp}/{rule}/global-base'] = _boot(B[('global', rule)][sp], B[('base', rule)][sp])
    for sp in ('eval9s', 'eval8'):
        S['err'][sp] = {arm: track_err(sp, arm) for arm in ARMS}
    # reference: the existing estimator's tracks, same rules (scored from the repository's own est tracks)
    json.dump(S, open(os.path.join(HERE, 'crop_ablation_quick.json' if quick else 'crop_ablation.json'), 'w'), indent=1, default=float)
    print(json.dumps(S['diff'], indent=1)); print(json.dumps(S['err'], indent=1))
    for k, v in S['acc'].items():
        print(k, 'eval9s', v['eval9s'], 'eval8', v['eval8'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['encode', 'train', 'score', 'all'])
    ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
    if a.quick:
        RUN = os.path.join(HERE, 'crop_runs_quick')
    os.makedirs(RUN, exist_ok=True)
    if a.stage in ('encode', 'all'):
        encode(a.quick)
    if a.stage in ('train', 'all'):
        train_heads(a.quick); make_tracks(a.quick)
    if a.stage in ('score', 'all'):
        score(a.quick)
