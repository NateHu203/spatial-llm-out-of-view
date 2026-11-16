"""Real-VLM MVE for NowRoPE on Qwen2.5-VL-3B-Instruct (v0.6), oracle(+noise) geometry, synthetic dynamic scenes.

Arms (identical data, LoRA config, steps, seed):
  zeroshot  : no training (reference)
  native    : LoRA only, native M-RoPE
  egogeo    : LoRA + geometric rotary on 16 low-freq pairs, tokens at OBSERVED positions expressed in the
              ego-frame at t* (camera motion compensated, objects assumed static = static-world geometry)
  feattrans : LoRA + native M-RoPE + MLP(transported coords, variance, validity) added to visual tokens
              (the decisive "same information as features" control)
  nowrope   : LoRA + geometric rotary on the same 16 pairs, tokens TRANSPORTED to t* by estimated velocity,
              per-token attenuation a = exp(-1/2 w^2 sigma^2) applied to cos/sin (=> a_i a_j on the kernel)
Usage:
  python mve_nowrope.py gen --split train --n 3000 ; python mve_nowrope.py gen --split eval --n 600
  python mve_nowrope.py run --arm nowrope --seed 0
"""
import argparse, json, math, os, time, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
DATA = os.path.join(HERE, 'mve_data')
MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
TRAIN_SPLIT, EVAL_SPLIT = 'train', 'eval'
SIG_P, SIG_V = 0.05, 0.05                     # "teacher" geometry noise (m, m/s)
LAMBDAS = [0.5, 1, 2, 4, 8, 16, 32, 64]          # metres
OMEGA = torch.tensor([2 * math.pi / l for l in LAMBDAS])
PAIRS_Z = list(range(32, 40))                    # lowest-frequency 8 pairs of the M-RoPE height section
PAIRS_X = list(range(56, 64))                    # lowest-frequency 8 pairs of the M-RoPE width section


# ------------------------------------------------------------------ data
def _gen_one(args):
    split, i = args
    import scenes
    if split in ('train2', 'eval2'):
        seed = {'eval2': 20_000_000, 'train2': 30_000_000}[split] + i
        it = scenes.sample_item_v2(i, split)
    else:
        seed = (0 if split == 'train' else 10_000_000) + i
        it = scenes.sample_item(seed)
    (cxz, yaw), tstar = it['cam_now'], it['tstar']
    rng = np.random.default_rng(seed + 7)
    ego, trn, var, valid = [], [], [], []
    for t, g in enumerate(it['geos']):
        w = g['world'].reshape(-1, 2) + SIG_P * rng.standard_normal((64, 2))
        v = g['vel'].reshape(-1, 2) + SIG_V * rng.standard_normal((64, 2)) * (g['kind'].reshape(-1, 1) == 2)
        kind = g['kind'].reshape(-1)
        dt = tstar - t
        ego.append(scenes.to_cam(w, cxz, yaw))
        trn.append(scenes.to_cam(w + v * dt, cxz, yaw))
        var.append(np.where(kind == 2, SIG_P ** 2 + dt ** 2 * SIG_V ** 2, SIG_P ** 2))
        valid.append(kind > 0)
    np.savez_compressed(os.path.join(DATA, split, f'{i:05d}.npz'), frames=np.stack(it['frames']),
                        ego=np.concatenate(ego).astype(np.float32), trn=np.concatenate(trn).astype(np.float32),
                        var=np.concatenate(var).astype(np.float32), valid=np.concatenate(valid),
                        meta=json.dumps(dict(question=it['question'], options=[str(o) for o in it['options']],
                                             label=int(it['label']), qtype=it['qtype'], hidden=bool(it['hidden']),
                                             future=bool(it['future']), stale=float(it['stale']),
                                             motion_required=bool(it.get('motion_required', not scenes.static_heuristic_ok(it))))))


def gen(split, n, workers=8):
    from multiprocessing import Pool
    os.makedirs(os.path.join(DATA, split), exist_ok=True)
    todo = [(split, i) for i in range(n) if not os.path.exists(os.path.join(DATA, split, f'{i:05d}.npz'))]
    with Pool(workers) as p:
        for k, _ in enumerate(p.imap_unordered(_gen_one, todo, chunksize=8)):
            if k % 200 == 0:
                print(split, k, '/', len(todo), flush=True)


def load(split, i):
    z = np.load(os.path.join(DATA, split, f'{i:05d}.npz'))
    return dict(frames=z['frames'], ego=z['ego'], trn=z['trn'], var=z['var'], valid=z['valid'], **json.loads(str(z['meta'])))


# ------------------------------------------------------------------ model pieces
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, dtype=torch.float32) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.A), self.B) * self.s).to(x.dtype)


class Ctx:
    """per-forward geometry context consumed by the rotary patch and the visual hook."""
    arm = 'native'; coords = None; scale = None; vis_mask = None; feat = None
    gate = None   # *_g arms: zero-init per-pair gate, angle = native + g (geo - native), scale = 1 + g (a - 1)


def patch_rotary(rot):
    orig = rot.forward
    om = OMEGA

    def fwd(x, position_ids):
        cos, sin = orig(x, position_ids)
        if Ctx.arm.endswith('_g') and Ctx.coords is not None:
            return gated(cos, sin)
        if Ctx.arm not in ('egogeo', 'nowrope') or Ctx.coords is None:
            return cos, sin
        cos, sin = cos.clone(), sin.clone()                       # B,L,128
        B, L, _ = cos.shape
        dev = cos.device
        ang_x = torch.zeros(B, L, 8, device=dev); ang_z = torch.zeros(B, L, 8, device=dev)
        scl = torch.ones(B, L, 8, device=dev)
        m = Ctx.vis_mask                                          # B,L bool
        c = Ctx.coords.to(dev)                                    # Nvis,2 (lateral x, forward z)
        ang_x[m] = c[:, :1] * om.to(dev)
        ang_z[m] = c[:, 1:] * om.to(dev)
        scl[m] = Ctx.scale.to(dev)                                # Nvis,8
        for pairs, ang in ((PAIRS_X, ang_x), (PAIRS_Z, ang_z)):
            idx = torch.tensor(pairs, device=dev)
            cv, sv = (ang.cos() * scl).to(cos.dtype), (ang.sin() * scl).to(sin.dtype)
            cos[..., idx] = cv; cos[..., idx + 64] = cv
            sin[..., idx] = sv; sin[..., idx + 64] = sv
        return cos, sin
    rot.forward = fwd


def gated(cos, sin):
    B, L, _ = cos.shape
    dev = cos.device
    om = OMEGA.to(dev)
    geo = torch.zeros(B, L, 16, device=dev); scl = torch.ones(B, L, 16, device=dev)
    m = Ctx.vis_mask
    c = Ctx.coords.to(dev)
    geo[m] = torch.cat([c[:, :1] * om, c[:, 1:] * om], -1)           # 8 x-pairs then 8 z-pairs
    scl[m] = torch.cat([Ctx.scale, Ctx.scale], -1).to(dev)
    idx = torch.tensor(PAIRS_X + PAIRS_Z, device=dev)
    nat = torch.atan2(sin[..., idx].float(), cos[..., idx].float())
    g = Ctx.gate
    ang = nat + g * (geo - nat)
    mag = 1 + g * (scl - 1)
    cv, sv = (ang.cos() * mag).to(cos.dtype), (ang.sin() * mag).to(sin.dtype)
    cos = torch.cat([cos[..., :64].index_copy(-1, idx, cv), cos[..., 64:].index_copy(-1, idx, cv)], -1)
    sin = torch.cat([sin[..., :64].index_copy(-1, idx, sv), sin[..., 64:].index_copy(-1, idx, sv)], -1)
    return cos, sin


def build(arm, seed):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.manual_seed(seed)
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=224 * 224, max_pixels=224 * 224)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    for p in model.parameters():
        p.requires_grad_(False)
    lm = model.model.language_model
    params = []
    if arm != 'zeroshot':
        for layer in lm.layers:
            at = layer.self_attn
            for nm in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                lo = LoRALinear(getattr(at, nm)).to('cuda'); setattr(at, nm, lo); params += [lo.A, lo.B]
    patch_rotary(lm.rotary_emb)
    if arm.endswith('_g'):
        Ctx.gate = nn.Parameter(torch.zeros(16, device='cuda')); params.append(Ctx.gate)
    feat_mlp = None
    if arm == 'feattrans':
        d = model.config.text_config.hidden_size
        feat_mlp = nn.Sequential(nn.Linear(36, 256), nn.GELU(), nn.Linear(256, d)).to('cuda')
        nn.init.zeros_(feat_mlp[2].weight); nn.init.zeros_(feat_mlp[2].bias)
        params += list(feat_mlp.parameters())

        def hook(mod, inp, out):
            if Ctx.feat is not None:
                out.pooler_output = out.pooler_output + feat_mlp(Ctx.feat.to('cuda')).to(out.pooler_output.dtype)
            return out
        model.model.visual.register_forward_hook(hook)
    return proc, model, params, feat_mlp


def fourier(c):
    om = OMEGA.to(c.device)
    a = torch.cat([c[:, :1] * om, c[:, 1:] * om], -1)
    return torch.cat([a.sin(), a.cos()], -1)                     # 32


def encode(proc, it, dev='cuda'):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in it['frames']]
    content = [{'type': 'image'} for _ in imgs] + [{'type': 'text', 'text': it['question']}]
    text = proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False)
    enc = proc(text=[text], images=imgs, return_tensors='pt')
    return {k: v.to(dev) for k, v in enc.items()}


def set_ctx(arm, it, enc, model):
    Ctx.arm = arm
    ids = enc['input_ids']
    vis = ids == model.config.image_token_id
    n = int(vis.sum())
    assert n == len(it['ego']), (n, len(it['ego']))
    valid = torch.tensor(it['valid'])
    var = torch.tensor(it['var'])
    base = arm[:-2] if arm.endswith('_g') else arm
    if base in ('egogeo', 'nowrope'):
        coords = torch.tensor(it['trn'] if base == 'nowrope' else it['ego'])
        coords = coords * valid[:, None]                        # sky -> origin ...
        if base == 'nowrope':
            scale = torch.exp(-0.5 * var[:, None] * OMEGA[None] ** 2)
        else:
            scale = torch.ones(n, 8)
        scale = scale * valid[:, None]                          # ... and zero weight
        Ctx.coords, Ctx.scale, Ctx.vis_mask = coords, scale, vis
    else:
        Ctx.coords = None
    if arm == 'feattrans':
        trn = torch.tensor(it['trn'])
        Ctx.feat = torch.cat([fourier(trn) * valid[:, None], var[:, None], valid[:, None].float(),
                              torch.zeros(n, 2)], -1)            # 36
    else:
        Ctx.feat = None


def option_ids(proc, opts):
    ids = [proc.tokenizer.encode(o, add_special_tokens=False)[0] for o in opts]
    assert ids[0] != ids[1], opts
    return ids


def evaluate(proc, model, arm, n_eval):
    model.eval(); rows = []
    with torch.no_grad():
        for i in range(n_eval):
            it = load(EVAL_SPLIT, i)
            enc = encode(proc, it); set_ctx(arm, it, enc, model)
            logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
            oid = option_ids(proc, it['options'])
            pred = int(logits[oid[1]] > logits[oid[0]])
            rows.append(dict(i=i, correct=int(pred == it['label']), qtype=it['qtype'], hidden=it['hidden'],
                             future=it['future'], stale=it['stale'],
                             margin=float(logits[oid[it['label']]] - logits[oid[1 - it['label']]])))
    return rows


def run(arm, seed, n_train, n_eval, lr, accum, out):
    proc, model, params, _ = build(arm, seed)
    t0 = time.time()
    if arm != 'zeroshot':
        gp = [p for p in params if p is not Ctx.gate]
        groups = [{'params': gp, 'lr': lr}] + ([{'params': [Ctx.gate], 'lr': lr * 100}] if arm.endswith('_g') else [])
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
        order = np.random.default_rng(seed).permutation(n_train)
        steps = len(order) // accum
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=steps, pct_start=0.05)
        model.train(); run_loss = 0.0
        for k, i in enumerate(order):
            it = load(TRAIN_SPLIT, int(i))
            enc = encode(proc, it); set_ctx(arm, it, enc, model)
            logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
            oid = option_ids(proc, it['options'])
            loss = F.cross_entropy(logits[None], torch.tensor([oid[it['label']]], device='cuda')) / accum
            loss.backward(); run_loss += loss.item()
            if (k + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); sched.step()
            if (k + 1) % 400 == 0:
                print(f'{arm} s{seed} {k+1}/{len(order)} loss {run_loss/ (400/accum):.3f} {time.time()-t0:.0f}s', flush=True); run_loss = 0.0
    train_sec = time.time() - t0
    rows = evaluate(proc, model, arm, n_eval)
    acc = float(np.mean([r['correct'] for r in rows]))
    res = dict(arm=arm, seed=seed, n_train=n_train if arm != 'zeroshot' else 0, acc=acc, train_sec=round(train_sec),
               gate=(Ctx.gate.detach().cpu().tolist() if arm.endswith('_g') else None),
               rows=rows)
    with open(out, 'a') as fh:
        fh.write(json.dumps(res) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--split', default='train'); ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--arm', default='nowrope'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_train', type=int, default=3000); ap.add_argument('--n_eval', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(HERE, 'results_mve.jsonl'))
    ap.add_argument('--train_split', default='train'); ap.add_argument('--eval_split', default='eval')
    a = ap.parse_args()
    TRAIN_SPLIT, EVAL_SPLIT = a.train_split, a.eval_split
    if a.cmd == 'gen':
        gen(a.split, a.n)
    else:
        run(a.arm, a.seed, a.n_train, a.n_eval, a.lr, a.accum, a.out)
