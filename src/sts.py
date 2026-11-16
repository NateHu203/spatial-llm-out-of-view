"""Shared-Trajectory Supervision (STS) pilot on Qwen2.5-VL-3B (v0.6).

Five-arm factorial (reviewer's design; identical QA exposures, LoRA, steps, head capacity):
  lora         : answer CE only
  indep_aux    : aux head predicts per-time positions r_k(t) independently (time-conditioned); state loss; no coupling
  shared_aux   : aux head predicts ONE state (p, v for A,B in last-frame camera axes + camera speed/yaw rate);
                 analytic transition gives r_k(t); state loss; no coupling
  indep_coupled: per-time head + coupling loss (answer log-odds <-> relation margins of the per-time predictions)
  sts          : shared state + transition + coupling loss
At evaluation the aux head is NOT used: answers come from the ordinary language head.
Usage:
  python sts.py gen --split train3 --n 1000 ; python sts.py gen --split eval3 --n 300
  python sts.py run --arm sts --seed 0
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
DATA = os.path.join(HERE, 'sts_data')
TRAIN_TIMES = (7.0, 8.0, 9.0)
TAU_LR, TAU_NEAR = 0.3, 1.0
HID_LAYER = 27


# ------------------------------------------------------------------ data
def _gen_one(args):
    split, i = args
    import scenes
    if split in ('train4', 'eval4', 'train5', 'eval5'):   # repaired task: twin pairs -> scenes 2i, 2i+1 (identical last frames); 5 = moving camera
        seed = {'train4': 60_000_000, 'eval4': 70_000_000, 'train5': 80_000_000, 'eval5': 90_000_000}[split] + i
        for tw in scenes.sample_twin_pair(seed, moving=split.endswith('5')):
            meta = dict(questions=tw['questions'], names=tw['names'], twin=tw['twin'], pair=i, tracks=tw['tracks'],
                        hidden_end=[False, False])
            np.savez_compressed(os.path.join(DATA, split, f'{2 * i + tw["twin"]:05d}.npz'), frames=np.stack(tw['frames']),
                                meta=json.dumps(meta))
        return
    if split in ('train6', 'train6d', 'eval6', 'train7', 'train7d', 'eval7'):   # natural-distribution history twins -> scenes 2i, 2i+1
        # train6: every pair; train6d / eval6: first seed whose twins disagree on >=1 future question (train: t=8,9)
        # train7 / train7d / eval7: salient-but-disguised twins (ordinary scene + the same scene with every velocity
        # reversed); same pair-selection rules as train6 / train6d / eval6
        base, need = {'train6': (110_000_000, None), 'train6d': (130_000_000, (8.0, 9.0)), 'eval6': (120_000_000, (8.0, 9.0, 10.0)),
                      'train7': (140_000_000, None), 'train7d': (160_000_000, (8.0, 9.0)), 'eval7': (150_000_000, (8.0, 9.0, 10.0))}[split]
        history = 'reflect' if '7' in split else 'posterior'
        for j in range(1000 if need else 1):
            out = scenes.sample_natural_twin(base + i * (1000 if need else 1) + j, history=history)
            if need is None or any(q['differs'] for q in out[0]['questions'] if q['t'] in need):
                break
        for tw in out:
            meta = dict(questions=tw['questions'], names=tw['names'], twin=tw['twin'], pair=i, tracks=tw['tracks'],
                        hidden_end=tw['hidden_end'])
            np.savez_compressed(os.path.join(DATA, split, f'{2 * i + tw["twin"]:05d}.npz'), frames=np.stack(tw['frames']),
                                meta=json.dumps(meta))
        return
    if split in ('train8', 'eval8'):   # exit twins: same last frame AND same last-seen position of the exited ball (§4t)
        for tw in scenes.sample_exit_twin({'train8': 180_000_000, 'eval8': 170_000_000}[split] + i):
            meta = dict(questions=tw['questions'], names=tw['names'], twin=tw['twin'], pair=i, tracks=tw['tracks'],
                        hidden_end=tw['hidden_end'], exit_obj=tw['exit_obj'])
            np.savez_compressed(os.path.join(DATA, split, f'{2 * i + tw["twin"]:05d}.npz'), frames=np.stack(tw['frames']),
                                meta=json.dumps(meta))
        return
    base = {'train3': 40_000_000, 'eval3': 50_000_000}[split]
    sc = scenes.sample_scene_multi(base + i)
    meta = dict(questions=sc['questions'], state=sc['state'], names=sc['names'], hidden_end=sc['hidden_end'])
    np.savez_compressed(os.path.join(DATA, split, f'{i:05d}.npz'), frames=np.stack(sc['frames']), meta=json.dumps(meta))


def gen(split, n, workers=8):
    from multiprocessing import Pool
    os.makedirs(os.path.join(DATA, split), exist_ok=True)
    todo = [(split, i) for i in range(n) if not os.path.exists(os.path.join(DATA, split, f'{(2 * i + 1) if split in TWIN_SPLITS else i:05d}.npz'))]
    with Pool(workers) as p:
        for k, _ in enumerate(p.imap_unordered(_gen_one, todo, chunksize=8)):
            if k % 200 == 0:
                print(split, k, '/', len(todo), flush=True)


TWIN_SPLITS = ('train4', 'eval4', 'train5', 'eval5', 'train6', 'train6d', 'eval6', 'train7', 'train7d', 'eval7', 'train8', 'eval8')
EVAL_SCENES = {'eval3': 300, 'eval4': 400, 'eval5': 400, 'eval6': 600, 'eval7': 600, 'eval8': 600, 'eval9': 600, 'eval9s': 600, 'eval10': 600, 'eval11': 600, 'eval12': 600, 'eval13': 600}


def load(split, i):
    z = np.load(os.path.join(DATA, split, f'{i:05d}.npz'))
    return dict(frames=z['frames'], **json.loads(str(z['meta'])))


# ------------------------------------------------------------------ dynamics
def transition(p, v, speed, yaw_rate, dt):
    """p, v: (2,2) [object, (lat, fwd)] in last-frame camera axes; returns positions in the camera frame at 7+dt."""
    w = yaw_rate
    small = w.abs() < 1e-4
    w_safe = torch.where(small, torch.full_like(w, 1e-4), w)
    cx = torch.where(small, torch.zeros_like(w), speed / w_safe * (1 - torch.cos(w * dt)))
    cz = torch.where(small, speed * dt, speed / w_safe * torch.sin(w * dt))
    d = p + v * dt - torch.stack([cx, cz])[None]
    a = w * dt
    return torch.stack([d[:, 0] * torch.cos(a) - d[:, 1] * torch.sin(a), d[:, 0] * torch.sin(a) + d[:, 1] * torch.cos(a)], -1)


def margins(r):
    """r: (2,2) positions of A,B at t. Returns (log-odds for 'right' / B-is-the-answer) teacher logits for both qtypes.
    leftright: label 1 ('right') when A is to the right of B, i.e. r_A.x > r_B.x
    closer   : label 1 (B closer) when |r_B|^2 < |r_A|^2"""
    lr = (r[0, 0] - r[1, 0]) / TAU_LR
    nr = ((r[0] ** 2).sum() - (r[1] ** 2).sum()) / TAU_NEAR
    return lr, nr


class AuxHead(nn.Module):
    """Cross-attention pooling over visual hidden states. shared: queries {A, B, camera} -> state.
    indep: queries {A, B} x time embedding -> per-time positions. Same width/depth in both modes."""
    def __init__(self, d_in=2048, d=256, mode='shared'):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(d_in, d)
        self.q = nn.Parameter(torch.randn(3, d) * 0.02)          # A, B, camera
        self.col = nn.Embedding(4, d)                            # object colour identity
        self.temb = nn.Linear(1, d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(self, h_vis, col_ids, t=None):
        kv = self.proj(h_vis.float())[None]                      # 1, N, d
        q = self.q.clone()
        q[:2] = q[:2] + self.col(col_ids)
        if self.mode == 'indep':
            q = q[:2] + self.temb(torch.tensor([[t - 7.0]], device=q.device))
        o, _ = self.attn(q[None], kv, kv)
        return self.mlp(o[0])                                    # rows x 4


# ------------------------------------------------------------------ main loop
def build(seed, aux_mode):
    import mve_nowrope as M
    proc, model, params, _ = M.build('native', seed)
    proc.tokenizer.padding_side = 'left'
    head = AuxHead(mode=aux_mode).to('cuda') if aux_mode else None
    return proc, model, params, head


INPUT = 'images'   # 'images': 8 separate images; 'video': native Qwen2.5-VL video path (2-frame temporal patches, timed M-RoPE)


def track_text(tr, names):
    """Oracle-perception text: visible-frame positions of the two balls in last-frame camera axes + camera kinematics."""
    parts = []
    for k, n in enumerate(names):
        obs = [f't={t} ({tr["pos"][t][k][0]:+.2f}, {tr["pos"][t][k][1]:+.2f})' for t in range(8) if tr['vis'][t][k]]
        parts.append(f'{n} ball: ' + '; '.join(obs))
    return ('Measured positions in metres (x = right, z = forward, in the camera axes of the last frame t=7 s; '
            'frames where a ball is not visible are omitted). ' + ' | '.join(parts) +
            f'. Camera forward speed {tr["cam_speed"]:.2f} m/s, yaw rate {tr["yaw_rate"]:+.3f} rad/s. ')


def encode_cached(proc, items, n_frames=8):
    """Cached-ViT path without any image preprocessing. items: [(feats (n_frames*k, d), text)], one sequence each,
    sequences may come from different scenes. The prompt is expanded to the processor's own placeholder layout
    (k <|image_pad|> per image, grid [1, 2*sqrt(k), 2*sqrt(k)]); input_ids / attention_mask / mm_token_type_ids /
    image_grid_thw were checked equal to the processor's output on Colab (2026-10-23)."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    if HIST['last1']:   # recurrent-state pilot (Stage R): only the newest frame's k tokens reach the LLM
        items = [(f[-(f.shape[0] // n_frames):], t) for f, t in items]; n_frames = 1
    pad = '<|image_pad|>'
    k = items[0][0].shape[0] // n_frames; side = 2 * int(round(k ** 0.5))
    texts = [proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}] * n_frames + [{'type': 'text', 'text': t}]}],
                                      add_generation_prompt=True, tokenize=False).replace(pad, pad * k) for _, t in items]
    enc = proc.tokenizer(texts, return_tensors='pt', padding=True)
    ids = enc['input_ids']
    enc = dict(input_ids=ids, attention_mask=enc['attention_mask'],
               mm_token_type_ids=(ids == proc.tokenizer.convert_tokens_to_ids(pad)).long(),
               image_grid_thw=torch.tensor([[1, side, side]] * (n_frames * len(items))))
    enc = {key: v.to('cuda') for key, v in enc.items()}
    vf = torch.cat([f for f, _ in items], 0)
    if SLOTS.get('vis') and SLOTS['feats'] is not None:   # visual-stream state: += mlp(state) on the last frame's k tokens
        sf = SLOTS['feats']
        assert sf.shape[0] == len(items), (sf.shape, len(items))
        delta = SLOTS['mlp'](sf.reshape(len(items), -1)).to(vf.dtype)                  # n_items, d
        add = torch.zeros(len(items), n_frames, k, 1, device=vf.device, dtype=vf.dtype)
        add[:, -1] = 1
        vf = (vf.view(len(items), n_frames, k, -1) + add * delta[:, None, None, :]).reshape(vf.shape)
    enc['mm_encoder_outputs'] = {'image': BaseModelOutputWithPooling(pooler_output=(vf,))}
    return enc


def encode_batch(proc, frames, questions, prefix='', feats=None):
    if feats is not None and INPUT == 'images':
        return encode_cached(proc, [(feats, prefix + q['question']) for q in questions], len(frames))
    from PIL import Image
    texts = []
    for q in questions:
        vis_part = [{'type': 'video'}] if INPUT == 'video' else [{'type': 'image'} for _ in frames]
        content = vis_part + [{'type': 'text', 'text': prefix + q['question']}]
        texts.append(proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False))
    if INPUT == 'video':
        enc = proc(text=texts, videos=[np.asarray(frames)] * len(questions), return_tensors='pt', padding=True,
                   do_sample_frames=False, fps=1.0)
    else:
        imgs = [Image.fromarray(f) for f in frames]
        enc = proc(text=texts, images=imgs * len(questions), return_tensors='pt', padding=True)
    enc = {k: v.to('cuda') for k, v in enc.items()}
    if feats is not None:   # cached ViT features replace pixel_values (same placeholder tokens, same grid_thw / M-RoPE)
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        enc.pop('pixel_values')
        enc['mm_encoder_outputs'] = {'image': BaseModelOutputWithPooling(pooler_output=(feats.repeat(len(questions), 1),))}
    return enc


def vis_feats(model, proc, split, i, frames):
    """Frozen-ViT output for the 8 frames of scene (split, i): (8*64, d) bf16, computed once and cached on disk.
    The images are encoded independently (per-image attention), so the 'lastframe' control is the last frame's
    64 rows repeated 8x and needs no extra ViT pass. Only for INPUT == 'images'."""
    path = os.path.join(DATA, 'vcache', split, f'{i:05d}.pt')
    if os.path.exists(path):
        return torch.load(path).to('cuda')
    from PIL import Image
    enc = proc.image_processor(images=[Image.fromarray(f) for f in frames], return_tensors='pt')
    with torch.no_grad():
        out = model.model.get_image_features(enc['pixel_values'].to('cuda'), enc['image_grid_thw'].to('cuda'), return_dict=True)
    f = torch.cat(out.pooler_output, 0).to(torch.bfloat16)
    os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(f.cpu(), path)
    return f


def lastframe_feats(f, n_frames=8):
    k = f.shape[0] // n_frames
    return f[-k:].repeat(n_frames, 1)


def opt_ids(proc, q):
    ids = [proc.tokenizer.encode(o, add_special_tokens=False)[0] for o in q['options']]
    assert ids[0] != ids[1]
    return ids


COLORS = ['red', 'blue', 'green', 'yellow']


def signed_margin(q):
    """Relation margin at the query time, positive toward option 1 (the label-1 answer): leftright -> A right of B;
    closer -> B closer than A. Uses the oracle camera-frame positions q['r'] = [[ax, az], [bx, bz]]."""
    (ax, az), (bx, bz) = q['r']
    return (ax - bx) if q['qtype'] == 'leftright' else (math.hypot(ax, az) - math.hypot(bx, bz))


_TRK = {}


def scene_tracks(split, i, sc):
    """Per-ball oracle tracks of the queried pair: pos (8 frames x 2 balls x (x, z)) in last-frame camera axes, vis, camera."""
    if 'tracks' in sc:
        return sc['tracks']
    if split not in _TRK:
        _TRK[split] = json.load(open(os.path.join(DATA, f'{split}_tracks.json')))
    return _TRK[split][i]


SLOT_DIM = 9


def slot_feats(tr, t, mode):
    """Oracle-stage memory-token features for the two queried balls (CHECKPOINT §4t), shape (2, SLOT_DIM).
    clock / clock_noprod (B1 and its control): last-seen frame k, dt = t* - k, velocity m at exit, dt*m (zeroed in the
    control), last-seen position, hidden-at-end flag. coast / coast_frozen (B2 ceiling and its control): the ball's
    position at t* in the t* camera frame, propagated from its last sighting with velocity m (coast) or with m = 0."""
    pos = np.array(tr['pos'], float); vis = np.array(tr['vis'], bool)
    rows = []
    for j in range(2):
        seen = np.nonzero(vis[:, j])[0]
        k = int(seen.max()) if len(seen) else 7
        # velocity from the last two distinct VISIBLE frames only (ladder audit: the old pos[k] - pos[k-1] could read a
        # hidden frame); fewer than two sightings -> m = 0
        m = (pos[k, j] - pos[seen[-2], j]) / (k - seen[-2]) if len(seen) >= 2 else np.zeros(2)
        dt = t - k
        if mode in ('clock', 'clock_noprod'):
            prod = dt * m if mode == 'clock' else np.zeros(2)
            rows.append([k / 7, dt / 10, *m, *prod, pos[k, j, 0] / 3, pos[k, j, 1] / 6, float(not vis[7, j])])
        else:
            mv = np.zeros(2) if mode == 'coast_frozen' else m   # coast / vis_coast: propagate; coast_frozen: stale
            r = transition(torch.tensor(pos[k, j] + (7 - k) * mv)[None].float(), torch.tensor(mv)[None].float(),
                           torch.tensor(float(tr['cam_speed'])), torch.tensor(float(tr['yaw_rate'])), torch.tensor(t - 7.0))[0]
            rows.append([r[0].item() / 3, r[1].item() / 6] + [0.0] * (SLOT_DIM - 2))
    return np.array(rows, np.float32)


HIST = dict(last1=False)   # --hist last1: drop the visual history, keep the last frame (Stage R)
SLOTS = dict(mlp=None, feats=None, id=None)   # memory tokens: embedding += mlp(feats) at the two '<|object_ref_start|>' slots
_NSC = {}


def shuffled_tracks(split, i):
    """Tracks of a different scene (i + 101 mod n) of the same split: the shuffled-state control."""
    if split not in _NSC:
        _NSC[split] = len([f for f in os.listdir(os.path.join(DATA, split)) if f.endswith('.npz')])
    j = (i + 101) % _NSC[split]
    has_json = os.path.exists(os.path.join(DATA, f'{split}_tracks.json'))
    return scene_tracks(split, j, {} if has_json else load(split, j))


def slot_text(names):
    return f'Object memory: the {names[0]} ball <|object_ref_start|>, the {names[1]} ball <|object_ref_start|>. '


def install_slots(model, proc, vis=False):
    """vis=False: memory tokens (embedding += mlp(feats) at the two text slots). vis=True (ladder Stage C, visual-stream
    state): no text slots; mlp(both balls' feats) is added to every visual token of the LAST frame (see encode_cached)."""
    d = model.config.text_config.hidden_size
    mlp = nn.Sequential(nn.Linear(SLOT_DIM * (2 if vis else 1), 256), nn.GELU(), nn.Linear(256, d)).to('cuda')
    nn.init.zeros_(mlp[2].weight); nn.init.zeros_(mlp[2].bias)   # starts as a no-op
    SLOTS.update(mlp=mlp, id=proc.tokenizer.convert_tokens_to_ids('<|object_ref_start|>'), vis=vis)
    if vis:
        return list(mlp.parameters())

    def hook(mod, inp, out):
        if SLOTS['feats'] is None:
            return out
        mask = inp[0] == SLOTS['id']
        assert int(mask.sum()) == 2 * SLOTS['feats'].shape[0], (int(mask.sum()), SLOTS['feats'].shape)
        out = out.clone()
        out[mask] = out[mask] + SLOTS['mlp'](SLOTS['feats'].reshape(-1, SLOT_DIM)).to(out.dtype)
        return out
    if SLOTS.get('handle') is not None:   # idempotent within one process
        SLOTS['handle'].remove()
    SLOTS['handle'] = model.get_input_embeddings().register_forward_hook(hook)
    return list(mlp.parameters())


def swap_lora(params, vals):
    """Load vals into the LoRA parameters in place; return the previous values (so the caller can swap back)."""
    old = [p.data.clone() for p in params]
    for p, v in zip(params, vals):
        p.data.copy_(v)
    return old


def pick_questions(qs, sample, rng):
    """3 training questions. uniform: at random. motion: static-heuristic-wrong first. differ: questions whose label
    differs between the scene's history twins first (natural twins), then at random."""
    if sample in ('motion', 'differ'):
        hit = (lambda q: not q['static_ok']) if sample == 'motion' else (lambda q: q.get('differs', False))
        mw = [q for q in qs if hit(q)]; ok = [q for q in qs if not hit(q)]
        mw = [mw[j] for j in rng.permutation(len(mw))]; ok = [ok[j] for j in rng.permutation(len(ok))]
        return (mw + ok)[:3]
    return [qs[j] for j in rng.choice(len(qs), min(3, len(qs)), replace=False)]


def make_order(tr_splits, n_train, pair, rng):
    """List of steps; each step is a list of (split, scene). pair='none': 1 scene per step (round-robin over splits).
    pair='twin': both history twins of one pair in the same step. pair='shuffle': the same scenes as 'twin' but
    re-dealt into random 2-scene steps (twins almost never share a step) -- the co-batching ablation."""
    if pair == 'none':
        return [[(tr_splits[k % len(tr_splits)], int(j))] for k, j in
                enumerate(np.stack([rng.permutation(-(-n_train // len(tr_splits))) for _ in tr_splits], 1).reshape(-1))][:n_train]
    n_pairs = {sp: len(os.listdir(os.path.join(DATA, sp))) // 2 for sp in tr_splits}
    per = -(-n_train // len(tr_splits))
    assert all(n >= per for n in n_pairs.values()), (n_pairs, per)
    steps = [[(tr_splits[k % len(tr_splits)], 2 * int(j)), (tr_splits[k % len(tr_splits)], 2 * int(j) + 1)] for k, j in
             enumerate(np.stack([rng.permutation(n_pairs[sp])[:per] for sp in tr_splits], 1).reshape(-1))][:n_train]
    if pair == 'shuffle':
        flat = [x for st in steps for x in st]
        flat = [flat[j] for j in rng.permutation(len(flat))]
        steps = [flat[k:k + 2] for k in range(0, len(flat), 2)]
    return steps


def run(arm, seed, n_train, n_eval, lr, out, sample='uniform', tag='', tracks='none', tr_split='train3', ev_split='eval3', micro=False,
        pair='none', cache=True, init_lora='', save_lora='', rm_weight=1.0, slots='none', swap='none', slot_shuffle=False):
    # ladder Stage C (pilot/LADDER_RESULTS.md): --slots vis_coast (state added to the last frame's visual tokens),
    # --slot_shuffle (state of another scene), --text_state (propagated positions as a text prefix, see __main__)
    # round-4 (CHECKPOINT §4t): --slots clock|clock_noprod|coast|coast_frozen adds two memory tokens (oracle stage);
    # --swap cross|self (B8, pair=twin on exit twins): extra items with the twin's motion code and the twin's labels
    # (cross) or plain duplicates (self, compute-matched control)
    # round-3 arms (CHECKPOINT §4q), all from a common warm start (--init_lora):
    #   anchor    : answer = z_B(C) + z_theta(V) - z_theta(C); B = frozen warm start, C = last frame repeated (train + test)
    #   poe       : train on log p_theta(V) + log p_B(C) (B frozen bias expert); test p_theta(V) alone
    #   respmatch : paired twins; CE + Huber(z_A - z_B, clip((m_A - m_B)/s_q, -3, 3)), z = option-1 log-odds
    aux_mode = {'lora': None, 'anchor': None, 'poe': None, 'respmatch': None,
                'indep_aux': 'indep', 'indep_coupled': 'indep', 'shared_aux': 'shared', 'sts': 'shared'}[arm]
    coupled = arm in ('indep_coupled', 'sts')
    proc, model, params, head = build(seed, aux_mode)
    if init_lora:
        swap_lora(params, [v.to('cuda') for v in torch.load(init_lora)])
    frozen = [p.detach().clone() for p in params] if arm in ('anchor', 'poe') else None

    def frozen_logits(enc):   # warm-start model B, no grad
        with torch.no_grad():
            old = swap_lora(params, frozen)
            lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            swap_lora(params, old)
        return lg
    groups = [{'params': params, 'lr': lr}] + ([{'params': list(head.parameters()), 'lr': 1e-3}] if head else [])
    if slots != 'none':
        assert cache and not micro and head is None and arm in ('lora',), 'memory tokens: cached batched lora path only'
        groups.append({'params': install_slots(model, proc, vis=slots.startswith('vis')), 'lr': 1e-3})
    assert swap == 'none' or (slots != 'none' and pair == 'twin')
    sfeat = lambda sp, i, sc, q: slot_feats(scene_tracks(sp, i, sc), q['t'], slots)
    if slot_shuffle:   # Stage C control: the state of ANOTHER scene of the same split (train and test)
        sfeat = lambda sp, i, sc, q: slot_feats(shuffled_tracks(sp, i), q['t'], slots)

    def set_slots(rows):   # rows: list of (2, SLOT_DIM) arrays, one per sequence, or None
        SLOTS['feats'] = None if slots == 'none' else torch.tensor(np.stack(rows), device='cuda')
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    # n_train = 0: evaluate-only (e.g. a saved --init_lora on a new test set)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=n_train, pct_start=0.05) if n_train else None
    rng = np.random.default_rng(seed)
    tr_splits, ev_splits = tr_split.split(','), ev_split.split(',')
    # mixture training: n_train steps drawn round-robin from the listed splits (each split's own scene order)
    order = make_order(tr_splits, n_train, pair, rng)
    cache = cache and INPUT == 'images'
    feats = (lambda sp, i, sc: vis_feats(model, proc, sp, i, sc['frames'])) if cache else (lambda sp, i, sc: None)
    if tracks == 'oracle' and tr_split == 'train3':
        TR = {sp: json.load(open(os.path.join(DATA, f'{sp}_tracks.json'))) for sp in (tr_split, ev_split)}
        pre = lambda sp, i, sc: track_text(TR[sp][i], sc['names'])
    elif tracks == 'oracle':                                   # repaired splits store tracks in the scene meta
        pre = lambda sp, i, sc: track_text(sc['tracks'], sc['names'])
    else:
        pre = lambda sp, i, sc: ''
    if slots != 'none' and not slots.startswith('vis'):
        pre0 = pre
        pre = lambda sp, i, sc: pre0(sp, i, sc) + slot_text(sc['names'])
    S_Q = {}
    if arm == 'respmatch':   # training-only margin-difference scale per question type (std over 150 pairs per split)
        d = {'leftright': [], 'closer': []}
        for sp in tr_splits:
            for j in range(min(150, len(os.listdir(os.path.join(DATA, sp))) // 2)):
                a, b = load(sp, 2 * j), load(sp, 2 * j + 1)
                for qa, qb in zip(a['questions'], b['questions']):
                    if qa['t'] in TRAIN_TIMES:
                        d[qa['qtype']].append(signed_margin(qa) - signed_margin(qb))
        S_Q = {k: float(np.std(v)) + 1e-6 for k, v in d.items()}
        print('respmatch margin-difference scale', S_Q, flush=True)
    t0 = time.time(); logs = []
    model.train()
    for step, st in enumerate(order):
        if pair != 'none':   # lora arm only: accumulate over the step's two scenes, one optimizer step
            assert head is None
            opt.zero_grad(); la = 0.0; shared = None; batch = []
            for tsp, i in st:
                sc = load(tsp, i)
                qs = [q for q in sc['questions'] if q['t'] in TRAIN_TIMES]
                if pair == 'twin':   # both twins get the SAME questions (labels may differ: that is the contrast)
                    if shared is None:
                        shared = [qs.index(q) for q in pick_questions(qs, sample, rng)]
                    qs = [qs[j] for j in shared]
                else:
                    qs = pick_questions(qs, sample, rng)
                oids = [opt_ids(proc, q) for q in qs]
                f = feats(tsp, i, sc)
                if f is not None and not micro:   # cached: the step's scenes go through ONE forward (below)
                    batch += [(f, pre(tsp, i, sc) + q['question'], o[q['label']], q, o,
                               sfeat(tsp, i, sc, q) if slots != 'none' else None) for q, o in zip(qs, oids)]
                    continue
                for ch in ([[k] for k in range(len(qs))] if micro else [list(range(len(qs)))]):
                    enc = encode_batch(proc, sc['frames'], [qs[k] for k in ch], pre(tsp, i, sc), f)
                    tgt = torch.tensor([oids[k][qs[k]['label']] for k in ch], device='cuda')
                    l = F.cross_entropy(model(**enc, logits_to_keep=1).logits[:, -1].float(), tgt, reduction='sum')
                    (l / (len(qs) * len(st))).backward(); la += l.item() / (len(qs) * len(st))
            if batch and swap != 'none':   # B8: twin 0 items, then twin 1 items, same question order
                n = len(batch) // 2
                if swap == 'cross':        # recipient frames + donor's motion code + donor's answer
                    batch += [(r[0], r[1], r[4][d[3]['label']], r[3], r[4], d[5])
                              for r, d in [(batch[k], batch[k + n]) for k in range(n)] + [(batch[k + n], batch[k]) for k in range(n)]]
                else:
                    batch += list(batch)
            if batch:
                set_slots([b[5] for b in batch])
                enc = encode_cached(proc, [(b[0], b[1]) for b in batch])
                L = model(**enc, logits_to_keep=1).logits[:, -1].float()
                l = F.cross_entropy(L, torch.tensor([b[2] for b in batch], device='cuda'))
                if arm == 'respmatch' and pair == 'twin':   # batch = twin 0's questions, then twin 1's, same order
                    n = len(batch) // 2
                    z = torch.stack([L[k, b[4][1]] - L[k, b[4][0]] for k, b in enumerate(batch)])
                    tgt_d = torch.tensor([max(-3.0, min(3.0, (signed_margin(batch[k][3]) - signed_margin(batch[k + n][3]))
                                                    / S_Q[batch[k][3]['qtype']])) for k in range(n)], device='cuda')
                    l = l + rm_weight * F.huber_loss(z[:n] - z[n:], tgt_d)
                l.backward(); la = l.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            logs.append([la, 0.0, 0.0])
            if (step + 1) % 100 == 0:
                print(f'{arm} s{seed} {step+1}/{n_train} ans {np.mean([x[0] for x in logs[-100:]]):.3f} {time.time()-t0:.0f}s', flush=True)
            continue
        tsp, i = st[0]
        sc = load(tsp, i)
        qs = [q for q in sc['questions'] if q['t'] in TRAIN_TIMES]
        qs = pick_questions(qs, sample, rng)   # 3 train questions, same rule for every arm
        oids = [opt_ids(proc, q) for q in qs]
        if micro and head is None:   # one question per forward/backward (long prompts overflow 24 GB as a batch of 3)
            opt.zero_grad(); la = 0.0
            for q, o in zip(qs, oids):
                enc = encode_batch(proc, sc['frames'], [q], pre(tsp, i, sc), feats(tsp, i, sc))
                l = F.cross_entropy(model(**enc, logits_to_keep=1).logits[:, -1].float(), torch.tensor([o[q['label']]], device='cuda'))
                (l / len(qs)).backward(); la += l.item() / len(qs)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            logs.append([la, 0.0, 0.0])
            if (step + 1) % 100 == 0:
                print(f'{arm} s{seed} {step+1}/{n_train} ans {np.mean([x[0] for x in logs[-100:]]):.3f} {time.time()-t0:.0f}s', flush=True)
            continue
        f = None if head is not None else feats(tsp, i, sc)
        if slots != 'none':
            set_slots([sfeat(tsp, i, sc, q) for q in qs])
        enc = encode_batch(proc, sc['frames'], qs, pre(tsp, i, sc), f)
        outp = model(**enc, logits_to_keep=1, output_hidden_states=head is not None)
        logits = outp.logits[:, -1].float()
        if arm in ('anchor', 'poe'):   # C = last frame repeated (free counterfactual); B = frozen warm start
            encC = encode_batch(proc, sc['frames'], qs, pre(tsp, i, sc), lastframe_feats(f))
            zBC = frozen_logits(encC)
            logits = (logits - model(**encC, logits_to_keep=1).logits[:, -1].float() + zBC) if arm == 'anchor' \
                else (F.log_softmax(logits, -1) + F.log_softmax(zBC, -1))
        tgt = torch.tensor([o[q['label']] for o, q in zip(oids, qs)], device='cuda')
        loss_ans = F.cross_entropy(logits, tgt)
        loss = loss_ans; ls = torch.zeros(()); lc = torch.zeros(())
        if head is not None:
            vis = enc['input_ids'][0] == model.config.image_token_id
            h = outp.hidden_states[HID_LAYER][0][vis]
            col = torch.tensor([COLORS.index(c) for c in sc['names']], device='cuda')
            st = sc['state']
            if aux_mode == 'shared':
                o = head(h, col)
                p, v = o[:2, :2], o[:2, 2:]
                speed, yaw = o[2, 0], o[2, 1] * 0.1
                ls = (F.mse_loss(p, torch.tensor(st['p'], device='cuda')) + F.mse_loss(v, torch.tensor(st['v'], device='cuda'))
                      + (speed - st['cam_speed']) ** 2 + ((yaw - st['yaw_rate']) / 0.1) ** 2)
                rs = {q['t']: transition(p, v, speed, yaw, torch.tensor(q['t'] - 7.0, device='cuda')) for q in qs}
            else:
                rs = {q['t']: head(h, col, q['t'])[:, :2] for q in qs}
                ls = sum(F.mse_loss(rs[q['t']], torch.tensor(q['r'], device='cuda')) for q in qs) / len(qs)
            loss = loss + 0.5 * ls
            if coupled:
                terms = []
                for k, (q, o) in enumerate(zip(qs, oids)):
                    lr_m, nr_m = margins(rs[q['t']])
                    teacher = lr_m if q['qtype'] == 'leftright' else nr_m
                    student = logits[k, o[1]] - logits[k, o[0]]
                    teacher = teacher.clamp(-15, 15)
                    qt = torch.sigmoid(teacher)
                    # KL(Ber(q_teacher) || Ber(p_student)) computed in logit space (stable); joint gradients into both branches
                    terms.append(F.binary_cross_entropy_with_logits(student, qt) - F.binary_cross_entropy_with_logits(teacher, qt))
                lc = torch.stack(terms).mean()
                loss = loss + 1.0 * lc
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if head is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        logs.append([loss_ans.item(), float(ls), float(lc)])
        if (step + 1) % 100 == 0:
            m = np.mean(logs[-100:], 0)
            print(f'{arm} s{seed} {step+1}/{n_train} ans {m[0]:.3f} state {m[1]:.3f} couple {m[2]:.3f} {time.time()-t0:.0f}s', flush=True)
    train_sec = time.time() - t0
    if save_lora:
        torch.save([p.detach().cpu() for p in params], save_lora)
    # ---- evaluation: ordinary language head only (aux head unused)
    # 'lastframe' repeats the final frame 8x: same token layout, zero motion information
    model.eval()
    for ev_split in ev_splits:
      n_ev = min(n_eval, EVAL_SCENES[ev_split])
      rows = []
      with torch.no_grad():
        for i in range(n_ev):   # one ViT pass per scene; 'lastframe' reuses the last frame's cached rows
            if i % 50 == 0:
                print(f'eval {ev_split} {i}/{n_ev} {time.time()-t0:.0f}s', flush=True)
            sc = load(ev_split, i)
            qs = sc['questions']
            fv = feats(ev_split, i, sc)
            modes = ('video', 'lastframe') if tag else ('video',)
            if fv is not None and not micro:   # cached: every question x mode of the scene in ONE forward
                fl = lastframe_feats(fv)
                items = [(m, q) for m in modes for q in qs]
                if slots != 'none':   # oracle stage: the same memory-token features in both modes (an interface ceiling)
                    set_slots([sfeat(ev_split, i, sc, q) for _, q in items])
                enc = encode_cached(proc, [(fv if m == 'video' else fl, pre(ev_split, i, sc) + q['question']) for m, q in items])
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
                if arm == 'anchor':   # deployed answer: z_B(C) + z_theta(V) - z_theta(C); on C itself it reduces to z_B(C)
                    assert modes == ('video', 'lastframe')
                    nq = len(qs)
                    zB = frozen_logits(encode_cached(proc, [(fl, pre(ev_split, i, sc) + q['question']) for q in qs]))
                    lg = torch.cat([zB + lg[:nq] - lg[nq:], zB])
                for k, (mode, q) in enumerate(items):
                    o = opt_ids(proc, q)
                    pred = int(lg[k, o[1]] > lg[k, o[0]])
                    rows.append(dict(mode=mode, scene=i, pair=sc.get('pair'), twin=sc.get('twin'), t=q['t'], qtype=q['qtype'], label=q['label'], correct=int(pred == q['label']),
                                     static_ok=q['static_ok'], differs=q.get('differs'), hidden_end=any(sc['hidden_end'])))
                continue
            for mode in modes:
              frames = sc['frames'] if mode == 'video' else np.repeat(sc['frames'][-1:], len(sc['frames']), 0)
              f = None if fv is None else (fv if mode == 'video' else lastframe_feats(fv))
              for chunk in ([[q] for q in qs] if micro else [c for c in (qs[:4], qs[4:]) if c]):
                  enc = encode_batch(proc, frames, chunk, pre(ev_split, i, sc), f)
                  lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
                  for k, q in enumerate(chunk):
                      o = opt_ids(proc, q)
                      pred = int(lg[k, o[1]] > lg[k, o[0]])
                      rows.append(dict(mode=mode, scene=i, pair=sc.get('pair'), twin=sc.get('twin'), t=q['t'], qtype=q['qtype'], label=q['label'], correct=int(pred == q['label']),
                                       static_ok=q['static_ok'], differs=q.get('differs'), hidden_end=any(sc['hidden_end'])))
      acc = float(np.mean([r['correct'] for r in rows if r['mode'] == 'video']))
      res = dict(arm=arm + tag, seed=seed, input=INPUT, tracks=tracks, split=ev_split, n_train=n_train, pair=pair, sample=sample, acc=acc, train_sec=round(train_sec), rows=rows)
      with open(out, 'a') as fh:
          fh.write(json.dumps(res) + '\n')
      print(json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--split', default='train3'); ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--arm', default='sts'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_train', type=int, default=1000); ap.add_argument('--n_eval', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--sample', default='uniform', choices=['uniform', 'motion', 'differ']); ap.add_argument('--tag', default='')
    ap.add_argument('--input', default='images', choices=['images', 'video']); ap.add_argument('--tracks', default='none', choices=['none', 'oracle'])
    ap.add_argument('--train_split', default='train3'); ap.add_argument('--eval_split', default='eval3')
    ap.add_argument('--micro', action='store_true')
    ap.add_argument('--pair', default='none', choices=['none', 'twin', 'shuffle']); ap.add_argument('--no_cache', action='store_true')
    ap.add_argument('--init_lora', default=''); ap.add_argument('--save_lora', default='')
    ap.add_argument('--rm_weight', type=float, default=1.0)
    ap.add_argument('--slots', default='none', choices=['none', 'clock', 'clock_noprod', 'coast', 'coast_frozen', 'vis_coast'])
    ap.add_argument('--swap', default='none', choices=['none', 'cross', 'self'])
    ap.add_argument('--slot_shuffle', action='store_true'); ap.add_argument('--text_state', action='store_true')
    ap.add_argument('--est_tracks', default='', help='splits whose text state comes from {split}_est_tracks.json '
                                                     '(learned estimator, Stage D), e.g. eval8,eval3; other splits use oracle tracks')
    ap.add_argument('--vel_fit', default='last2', choices=['last2', 'lsq'])
    ap.add_argument('--hist', default='all', choices=['all', 'last1'])
    ap.add_argument('--state_rule', default='', help='Stage M: text state from state_rules.<rule> at TEST time, '
                                                    'params from state_rules_results.json (est/orc by track source)')
    ap.add_argument('--est_file', default='', help='2026-11-06: tracks file for the --est_tracks split instead of {split}_est_tracks.json')
    ap.add_argument('--rule_params', default='', help='2026-11-06: JSON params for --state_rule instead of state_rules_results.json')
    ap.add_argument('--dist_text', default='none', choices=['none', 'append'],
                    help='2026-11-05 H8 diagnostic: append each ball distance from the camera to the Stage M state text')
    ap.add_argument('--out', default=os.path.join(HERE, 'results_sts.jsonl'))
    a = ap.parse_args()
    if a.cmd == 'gen':
        gen(a.split, a.n)
    else:
        INPUT = a.input
        HIST['last1'] = a.hist == 'last1'
        if a.text_state:   # every question gets the propagated positions at its own t* as a text prefix (train and test)
            from ladder_stage_b import propagated_text
            import ladder_stage_a
            ladder_stage_a.VEL['mode'] = a.vel_fit
            _load = load
            EST = {sp: json.load(open(a.est_file if a.est_file else os.path.join(HERE, f'{sp}_est_tracks.json'))) for sp in a.est_tracks.split(',') if sp}

            if a.state_rule:   # Stage M: test-time state from another update rule (same text format)
                import state_rules as SR
                PR = json.load(open(os.path.join(HERE, 'state_rules_results.json')))

                def rule_text(tr, names, q, split):
                    P = json.loads(a.rule_params) if a.rule_params else PR[f'{a.state_rule}/{"est" if split in EST else "orc"}']['params']
                    p7, v, _ = SR.scene_state(a.state_rule, P, tr)
                    _, r = SR.answer(p7, v, tr, q)
                    return ("Computed positions at the asked time, in the camera's own axes at that time (x = metres to the "
                            f'right, z = metres forward): the {names[0]} ball at ({r[0, 0]:+.2f}, {r[0, 1]:+.2f}), the {names[1]} '
                            f'ball at ({r[1, 0]:+.2f}, {r[1, 1]:+.2f}). ' +
                            (f'Distances from the camera at that time: the {names[0]} ball {np.linalg.norm(r[0]):.2f} m, '
                             f'the {names[1]} ball {np.linalg.norm(r[1]):.2f} m. ' if a.dist_text == 'append' else ''))

            def load(split, i):
                sc = _load(split, i)
                tr = EST[split][i] if split in EST else scene_tracks(split, i, sc)
                train = split not in EVAL_SCENES
                for q in sc['questions']:
                    txt = rule_text(tr, sc['names'], q, split) if a.state_rule and not train else propagated_text(tr, sc['names'], q)
                    q['question'] = txt + q['question']
                return sc
        run(a.arm, a.seed, a.n_train, a.n_eval, a.lr, a.out, a.sample, a.tag, a.tracks, a.train_split, a.eval_split, a.micro,
            a.pair, not a.no_cache, a.init_lora, a.save_lora, a.rm_weight, a.slots, a.swap, a.slot_shuffle)
