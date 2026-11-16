"""Round 7 (gates: idea-stage/GATES_round7_2026-11-08.md): the cheaper look, on the NEW sealed twins eval11.
Arms (all use the head-start formulation: S1 box in the query, output = box origin + side * (0.5 + r)):
  base_box   (V1h)  : 64 frame tokens only                                   (round-6 heads, reused)
  global_box (V2h)  : + 256 tokens of the whole frame at 448                  (round-6 heads, reused)
  crop       (V3)   : + 256 tokens of the 96 px square at 448                  (round-5 heads, reused)
  crop224    (V3c)  : + 64 tokens of the SAME 96 px square at 224 (1/4 of the extra tokens)      NEW, trained
  crop64     (V3s)  : + 64 tokens of a 64 px square at 224 (tighter square, more zoom per token) NEW, exploratory
Also scores the current system (A + B + C on the S1 estimates) on eval11, and measures the encoder time of one extra look.
Splits: train8, train9 (training + rule tuning), eval10 (development, secondary), eval11 (sealed, primary).
--quick: 16 scenes, 50 steps, eval9s replaces eval11 and eval10 (the smoke test never touches eval11).
Stages:  python crop_ablation_r7.py encode | train | score | all   [--quick] [--seeds 0,1,2]
Output:  crop_ablation_r7.json"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn
import crop_ablation as CA
import crop_ablation_r5 as R5
from crop_ablation_r6 import HeadBox

HERE = CA.HERE
ARMS = ('base_box', 'global_box', 'crop', 'crop224', 'crop64')
NEW = ('crop224', 'crop64')
FEAT = {'global_box': 'glob', 'crop': 'crop', 'crop224': 'crop224', 'crop64': 'crop64'}
BOXK = {'crop224': 'box', 'crop': 'box', 'global_box': 'box', 'base_box': 'box', 'crop64': 'box64'}
SMALL = 64


class HeadBox64(HeadBox):
    """HeadBox with 64 extra tokens (a 224x224 input gives 8x8 merged tokens)"""
    def __init__(self, extra):
        super().__init__(extra)
        self.pos_e = nn.Parameter(torch.randn(64, self.pos_b.shape[1]) * 0.02)


def make_head(arm):
    if arm in NEW:
        return HeadBox64(arm)
    if arm == 'base_box':
        return HeadBox(None)
    if arm == 'global_box':
        return HeadBox('global')
    return CA.Head('crop')


def setup(quick):
    ev, dev = ('eval9s', 'eval9s') if quick else ('eval11', 'eval10')
    CA.SPLITS = {'train8': 1200, 'train9': 600, dev: 600, ev: 600}
    CA.RUN = os.path.join(HERE, 'crop_runs_r7_quick' if quick else 'crop_runs_r7')
    os.makedirs(CA.RUN, exist_ok=True)
    return ev, dev


# ------------------------------------------------------------------ stage 1: encode (round-4 code + two 224 crops)
def encode(quick):
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import sts
    p224 = AutoProcessor.from_pretrained(CA.MODEL, min_pixels=224 * 224, max_pixels=224 * 224).image_processor
    p448 = AutoProcessor.from_pretrained(CA.MODEL, min_pixels=448 * 448, max_pixels=448 * 448).image_processor
    model = AutoModelForImageTextToText.from_pretrained(CA.MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    g = torch.Generator().manual_seed(0)
    W = torch.linalg.qr(torch.randn(2048, CA.PROJ, generator=g))[0].to('cuda', torch.bfloat16)     # same projection as rounds 4-6

    def feats(proc, imgs):
        enc = proc(images=imgs, return_tensors='pt')
        with torch.no_grad():
            out = model.model.get_image_features(enc['pixel_values'].to('cuda'), enc['image_grid_thw'].to('cuda'), return_dict=True)
        return [(o.to(torch.bfloat16) @ W).cpu() for o in out.pooler_output]

    # cost of ONE extra look (batch 1), vision encoder only, median over 50 runs after 10 warm-up runs
    if not os.path.exists(os.path.join(CA.RUN, 'timing.json')):
        img = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
        T = {}
        for name, proc, size in (('look448', p448, 448), ('look224', p224, 224)):
            im = img.resize((size, size)); tp, te = [], []
            for k in range(60):
                t0 = time.perf_counter(); enc = proc(images=[im], return_tensors='pt'); t1 = time.perf_counter()
                pv, th = enc['pixel_values'].to('cuda'), enc['image_grid_thw'].to('cuda'); torch.cuda.synchronize(); t2 = time.perf_counter()
                with torch.no_grad():
                    model.model.get_image_features(pv, th, return_dict=True)
                torch.cuda.synchronize(); t3 = time.perf_counter()
                if k >= 10:
                    tp.append(1000 * (t1 - t0)); te.append(1000 * (t3 - t2))
            T[name + '_preprocess_cpu'] = round(float(np.median(tp)), 2); T[name + '_encoder_gpu'] = round(float(np.median(te)), 2)
        T['gpu'] = torch.cuda.get_device_name(0)
        T['encoder_ratio_448_over_224'] = round(T['look448_encoder_gpu'] / T['look224_encoder_gpu'], 2)
        T['note'] = 'batch 1, median of 50 runs after 10 warm-up; a datacenter/laptop GPU, not an edge device'
        json.dump(T, open(os.path.join(CA.RUN, 'timing.json'), 'w')); print('timing', T, flush=True)

    rng = np.random.default_rng(0)
    for sp, n in CA.SPLITS.items():
        n = 16 if quick else n
        path = os.path.join(CA.RUN, f'{sp}_feats.pt')
        if os.path.exists(path):
            continue
        est = json.load(open(os.path.join(HERE, f'{sp}_est_tracks.json')))
        t0 = time.time(); S, BOX, BOX64, UV, COL = [], [], [], [], []
        B = torch.zeros(n, 8, 64, CA.PROJ, dtype=torch.bfloat16)
        G = torch.zeros(n, 8, 256, CA.PROJ, dtype=torch.bfloat16); C = torch.zeros_like(G)
        C2 = torch.zeros(n, 8, 64, CA.PROJ, dtype=torch.bfloat16); C3 = torch.zeros_like(C2)
        for i in range(n):
            sc = sts.load(sp, i); tr = sts.scene_tracks(sp, i, sc); e = est[i]
            sch = CA.schedule(e['vis']); fr = [Image.fromarray(f) for f in sc['frames']]
            B[i] = torch.stack(feats(p224, fr))
            glob, crops, crops224, crops64, boxes, boxes64, uvs = [], [], [], [], [], [], []
            for t in range(8):
                b = sch[t]
                if b < 0:
                    boxes.append([0, 0, 224]); boxes64.append([0, 0, 224]); uvs.append([np.nan, np.nan]); continue
                ce0 = CA.own_to_pix(CA.t7_to_own(np.array(e['pos'][t][b]), tr, t))
                if sp == 'train8':                       # same noise draw, same order as rounds 4-6 -> identical 96 boxes
                    ce0 = ce0 + rng.normal(0, 6, 2)
                ce = np.clip(ce0, CA.CROP / 2, 224 - CA.CROP / 2)
                x0, y0 = int(round(ce[0] - CA.CROP / 2)), int(round(ce[1] - CA.CROP / 2))
                boxes.append([x0, y0, CA.CROP])
                sq = fr[t].crop((x0, y0, x0 + CA.CROP, y0 + CA.CROP))
                crops.append(sq.resize((448, 448), Image.BICUBIC)); crops224.append(sq.resize((224, 224), Image.BICUBIC))
                cs = np.clip(ce0, SMALL / 2, 224 - SMALL / 2)
                a0, b0 = int(round(cs[0] - SMALL / 2)), int(round(cs[1] - SMALL / 2))
                boxes64.append([a0, b0, SMALL])
                crops64.append(fr[t].crop((a0, b0, a0 + SMALL, b0 + SMALL)).resize((224, 224), Image.BICUBIC))
                glob.append(fr[t].resize((448, 448), Image.BICUBIC))
                uvs.append(CA.own_to_pix(CA.t7_to_own(np.array(tr['pos'][t][b]), tr, t)) if tr['vis'][t][b] else [np.nan, np.nan])
            idx = [t for t in range(8) if sch[t] >= 0]
            if idx:
                for X, proc, L in ((G, p448, glob), (C, p448, crops), (C2, p224, crops224), (C3, p224, crops64)):
                    for t, f in zip(idx, feats(proc, L)):
                        X[i, t] = f
            S.append(sch); BOX.append(boxes); BOX64.append(boxes64); UV.append(uvs)
            COL.append([CA.COLORS.index(c) for c in sc['names']])
            if i % 100 == 0:
                print(sp, i, '/', n, f'{time.time() - t0:.0f}s', flush=True)
        torch.save(dict(base=B, glob=G, crop=C, crop224=C2, crop64=C3, sched=torch.tensor(np.array(S)),
                        box=torch.tensor(np.array(BOX), dtype=torch.float32), box64=torch.tensor(np.array(BOX64), dtype=torch.float32),
                        uv=torch.tensor(np.array(UV, float), dtype=torch.float32), col=torch.tensor(COL)), path)
        print(sp, 'encoded', n, f'{time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------ stage 2: heads + tracks
def extra_of(arm, D, s, t):
    return D[FEAT.get(arm, 'crop')][s, t]


def box_of(arm, D, s, t):
    return D[BOXK[arm]][s, t]


def train_heads(quick, seeds, steps=4000, bs=256):
    data = {sp: torch.load(os.path.join(CA.RUN, f'{sp}_feats.pt')) for sp in CA.TRAIN_SPLITS}
    IT = {sp: CA.items(D) for sp, D in data.items()}
    for seed in seeds:
        for arm in ARMS:
            path = os.path.join(CA.RUN, f'head_{arm}_s{seed}.pt')
            if os.path.exists(path):
                print('reuse', path, flush=True); continue
            if arm not in NEW and not quick:
                raise FileNotFoundError(f'{path}: heads of {arm} must be reused from rounds 5-6 (put them in {CA.RUN})')
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


# ------------------------------------------------------------------ stage 3: tune + score (CPU)
def _job(args):
    arm, rule, seed, run, splits, evs = args
    import state_rules as SR
    SR.RULES['imm'] = (SR.RULES['imm'][0], dict(SR.RULES['imm'][1], q_lo=[1e-4]))
    cache = {sp: json.load(open(os.path.join(run, f'{sp}_{arm}_s{seed}_tracks.json'))) for sp in splits}
    full = SR.scenes_of
    SR.scenes_of = lambda split: full(split)[:len(cache[split])]
    SR.tracks_of = lambda split, src: cache[split]
    P, _ = SR.tune(rule, 'x')
    return (arm, rule, seed), dict(params=P, rows={sp: SR.evaluate(rule, P, sp, 'x', rows_out=True) for sp in evs})


def _current(ev, n):
    """current system: A + B + C (rule CA) on the S1 estimates, parameters fixed earlier (state_rules_results.json)"""
    import state_rules as SR
    P = json.load(open(os.path.join(HERE, 'state_rules_results.json')))['ca/est']['params']
    full = SR.scenes_of; SR.scenes_of = lambda split: full(split)[:n]
    est = json.load(open(os.path.join(HERE, f'{ev}_est_tracks.json')))[:n]; SR.tracks_of = lambda split, src: est
    return CA._both(SR.evaluate('ca', P, ev, 'x', rows_out=True))


def score(quick, seeds, ev, dev):
    from multiprocessing import get_context
    evs = sorted({ev, dev})
    jobs = [(a, r, s, CA.RUN, list(CA.SPLITS), evs) for s in seeds for a in ARMS for r in ('ca', 'imm')]
    with get_context('spawn').Pool(min(6, os.cpu_count() or 2), maxtasksperchild=1) as p:
        res = dict(p.imap_unordered(_job, jobs))
    B = {k: {sp: CA._both(v['rows'][sp]) for sp in v['rows']} for k, v in res.items()}
    n_ev = len(json.load(open(os.path.join(CA.RUN, f'{ev}_crop_s{seeds[0]}_tracks.json'))))
    cur = _current(ev, n_ev)
    S = dict(quick=quick, eval_split=ev, dev_split=dev, seeds=seeds, acc={}, diff={}, per_seed={}, err={}, gates={})
    S['acc']['current/ca'] = {ev: CA._boot(cur)}
    for arm in ARMS:
        for rule in ('ca', 'imm'):
            S['acc'][f'{arm}/{rule}'] = {sp: R5._boot_seeds([B[(arm, rule, s)][sp] for s in seeds]) for sp in evs}
    pairs = (('crop', 'global_box'), ('crop224', 'crop'), ('crop224', 'base_box'), ('crop224', 'global_box'),
             ('crop64', 'crop224'), ('crop64', 'crop'), ('crop', 'base_box'), ('global_box', 'base_box'))
    for rule in ('ca', 'imm'):
        for sp in evs:
            for a, b in pairs:
                S['diff'][f'{sp}/{rule}/{a}-{b}'] = R5._boot_seeds([B[(a, rule, s)][sp] for s in seeds], [B[(b, rule, s)][sp] for s in seeds])
                for s in seeds:
                    S['per_seed'][f'{sp}/{rule}/{a}-{b}/s{s}'] = CA._boot(B[(a, rule, s)][sp], B[(b, rule, s)][sp])
    for arm in ('crop224', 'crop', 'global_box', 'base_box', 'crop64'):
        S['diff'][f'{ev}/ca/{arm}-current'] = R5._boot_seeds([B[(arm, 'ca', s)][ev] for s in seeds], [cur] * len(seeds))
    for sp in evs:
        S['err'][sp] = {arm: {f's{s}': R5.track_err(sp, arm, s) for s in seeds} for arm in ARMS}
    # gates (GATES_round7_2026-11-08.md), primary split, rule CA, 3-seed mean, bootstrap over scene pairs
    d = lambda k: S['diff'][f'{ev}/ca/{k}']
    S['gates'] = {
        'G0_confirm_round6 (crop - global_box: low > 0)': d('crop-global_box')[1][0] > 0,
        'G1_noninferior (crop224 - crop: low > -2)': d('crop224-crop')[1][0] > -2,
        'G2_cheap_look_helps (crop224 - base_box: low > 0)': d('crop224-base_box')[1][0] > 0,
        'G3_beats_current (crop224 - current: low > 0)': d('crop224-current')[1][0] > 0}
    S['cheap_look_pass'] = all(list(S['gates'].values())[1:])
    t_path = os.path.join(CA.RUN, 'timing.json')
    S['timing_ms'] = json.load(open(t_path)) if os.path.exists(t_path) else None
    out = os.path.join(HERE, 'crop_ablation_r7_quick.json' if quick else 'crop_ablation_r7.json')
    json.dump(S, open(out, 'w'), indent=1, default=float)
    for k, v in S['gates'].items():
        print(k, '->', 'PASS' if v else 'miss', flush=True)
    for a, b in pairs:
        print(f'{ev} CA {a} - {b}:', S['diff'][f'{ev}/ca/{a}-{b}'], flush=True)
    for k, v in S['acc'].items():
        print(k, v, flush=True)
    print('timing', S['timing_ms'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['encode', 'train', 'score', 'all'])
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(',')]
    ev, dev = setup(a.quick)
    if a.stage in ('encode', 'all'):
        encode(a.quick)
    if a.stage in ('train', 'all'):
        train_heads(a.quick, seeds); make_tracks(seeds)
    if a.stage in ('score', 'all'):
        score(a.quick, seeds, ev, dev)
