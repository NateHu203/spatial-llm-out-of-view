"""Distil the calculator into the language model: the student writes the state itself, then answers.

Teacher  = the change-aware calculator (state_rules 'ca', tuned parameters from state_rules_results.json) run on
           each ball's track: exact tracks (--teacher orc) or the measuring part's estimates (--teacher est).
Student  = Qwen2.5-VL-3B + LoRA (starts from model W). It sees all 8 frames and the question, and NO sentence.

Arms (--target):
  state  : the student is trained to write, as ordinary text, where the teacher puts the two balls at the asked time
           and then the answer:   "red (-1.20, +3.00) 3.23 m; blue (+1.50, +3.80) 4.09 m. Answer:" + answer word
           At test time it generates that line itself (greedy) and the answer is read after its own "Answer:".
  answer : control. Same data, steps and teacher answers, but the student is trained on the answer word only
           (no written state). This is model W's training recipe on the distillation data.
  record : the per-frame sightings first, then the state line, then the answer.
  motion : the camera's motion and each ball's position and velocity at the last frame first, then state and answer.
  locate : diagnostic, no question and no motion: write where the two balls are in the LAST frame, or that they are
           not visible. Measures how precisely the model can read a position as text, next to the measuring part.
           Use --eval_split eval3,eval9 (eval3 has many more balls visible in the last frame).

Reported on the eval split (twin accuracy = both twins right, questions whose answer differs between the twins):
  model    : the student's own answer
  written  : the answer a calculator gets from the positions the student wrote (state arm only)
  teacher  : the teacher's answer on the same questions (the thing being imitated)
and, for the state arm, the median distance between the written and the true positions.

Run from the workspace with PYTHONPATH=src and SPATIAL_LLM_WORK=<workspace> (as pipeline/runner.py does):
  python distill.py --target state  --save_lora d_state_s0.pt  --out res_d_state.jsonl
  python distill.py --target answer --save_lora d_answer_s0.pt --out res_d_answer.jsonl
  python distill.py --target state --n_train 0 --init_lora d_state_s0.pt --eval_split eval10   (evaluate only)
"""
import argparse, json, os, re, time
import numpy as np, torch, torch.nn.functional as F
import sts
import state_rules as SR

INSTR = (" First write where each of the two balls is at the asked time, as `colour (x, z) distance m` with x = metres "
         "to the right and z = metres forward in the camera's own axes at that time, then `Answer:` and the one-word answer.")
MARK = 'At the asked time:'
INSTR_REC = (" First list, for each of the two balls, the frames t0 to t7 in which it is visible, each with its position "
             "`(x, z)` in the camera's axes at the last frame (x = metres to the right, z = metres forward). Then write "
             f"`{MARK}` and each ball as `colour (x, z) distance m` in the camera's own axes at the asked time, then "
             "`Answer:` and the one-word answer.")
NUM = re.compile(r'\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)')
SIGHT = re.compile(r't(\d)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)')
INSTR_MOT = (" First write the camera's own motion as `Camera: speed S m/s, yaw Y rad/s`. Then, for each of the two balls, "
             "write where it is at the last frame and how it moves, as `colour: at (x, z) moving (vx, vz)` in the camera's "
             "axes at the last frame (x = metres to the right, z = metres forward; metres per second). Then write "
             f"`{MARK}` and each ball as `colour (x, z) distance m` in the camera's own axes at the asked time, then "
             "`Answer:` and the one-word answer.")
CAM = re.compile(r'speed\s*([+-]?\d+(?:\.\d+)?)\s*m/s,\s*yaw\s*([+-]?\d+(?:\.\d+)?)')
MOVE = re.compile(r'at\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*moving\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)')
# state line: about 45 tokens; record: up to 16 sightings of about 14 tokens; motion: camera + two balls + state
MAX_NEW = dict(state=72, record=330, motion=190)


def motion_line(names, tr, p7, v):
    """the calculator's own state, written out: the camera's motion, then each ball's position at the last frame and
    its velocity (both in the last frame's camera axes; a ball the rule treats as standing still has velocity 0)"""
    balls = '; '.join(f'{n}: at ({p7[j, 0]:+.2f}, {p7[j, 1]:+.2f}) moving ({v[j, 0]:+.2f}, {v[j, 1]:+.2f})' for j, n in enumerate(names))
    return f'Camera: speed {tr["cam_speed"]:.2f} m/s, yaw {tr["yaw_rate"]:+.3f} rad/s. {balls}. {MARK} '


def parse_motion(text):
    """(speed, yaw, p7 (2, 2), v (2, 2)) the student wrote before MARK, or None when malformed"""
    head = text.split(MARK)[0]
    c = CAM.search(head); m = MOVE.findall(head)
    if c is None or len(m) < 2:
        return None
    b = np.array(m[:2], float)
    return float(c.group(1)), float(c.group(2)), b[:, :2], b[:, 2:]


def odo_text(tr):
    """the camera's own motion, which the system treats as known (the calculator and the measuring part both use it)"""
    return f'Camera forward speed {tr["cam_speed"]:.2f} m/s, yaw rate {tr["yaw_rate"]:+.3f} rad/s. '


def record_line(names, tr):
    """the written record: every sighting of the two asked-about balls, in the last frame's camera axes"""
    pos = np.array(tr['pos'], float); vis = np.array(tr['vis'], bool)
    parts = []
    for j, n in enumerate(names):
        s = ', '.join(f't{t} ({pos[t, j, 0]:+.2f}, {pos[t, j, 1]:+.2f})' for t in range(len(vis)) if vis[t, j])
        parts.append(f'{n}: {s or "not seen"}')
    return '; '.join(parts) + f'. {MARK} '


def parse_record(text, n_frames=8):
    """(pos (n_frames, 2, 2), vis (n_frames, 2)) the student wrote before MARK, or None when the two balls cannot be told apart"""
    segs = text.split(MARK)[0].split(';')
    if len(segs) != 2:
        return None
    pos = np.zeros((n_frames, 2, 2)); vis = np.zeros((n_frames, 2), bool)
    for j, seg in enumerate(segs):
        for t, x, z in SIGHT.findall(seg):
            if int(t) < n_frames:
                pos[int(t), j] = (float(x), float(z)); vis[int(t), j] = True
    return pos, vis


# ------------------------------------------------------------------ teacher
class Teacher:
    def __init__(self, src, rule='ca'):
        self.src, self.rule = src, rule
        self.P = json.load(open(os.path.join(sts.HERE, 'state_rules_results.json')))[f'{rule}/{src}']['params']
        self.est = {}

    def tracks(self, split, i, sc):
        if self.src == 'orc':
            return sts.scene_tracks(split, i, sc)
        if split not in self.est:
            path = os.path.join(sts.HERE, f'{split}_est_tracks.json')
            assert os.path.exists(path), f'{path} missing: run est_infer.py {split}:<n> first, or use --teacher orc'
            self.est[split] = json.load(open(path))
        return self.est[split][i]

    def state(self, split, i, sc):
        """the calculator's state for the two asked-about balls: position at the last frame, velocity, and the tracks"""
        tr = self.tracks(split, i, sc)
        p7, v, _ = SR.scene_state(self.rule, self.P, tr)
        return p7, v, tr

    def scene(self, split, i, sc):
        """one calculator state per scene; answer(q) -> (label, positions (2, 2) at the asked time)"""
        p7, v, tr = self.state(split, i, sc)
        return lambda q: SR.answer(p7, v, tr, q)


def state_line(names, r):
    return '; '.join(f'{n} ({r[j, 0]:+.2f}, {r[j, 1]:+.2f}) {np.linalg.norm(r[j]):.2f} m' for j, n in enumerate(names)) + '. Answer:'


def parse_state(text):
    """positions (2, 2) the student wrote, or None when the line is malformed"""
    m = NUM.findall(text.split(MARK)[-1])   # record arm: the state comes after MARK
    return np.array(m[:2], float) if len(m) >= 2 else None


def relation(r, q):
    return int(r[0, 0] > r[1, 0]) if q['qtype'] == 'leftright' else int(np.linalg.norm(r[1]) < np.linalg.norm(r[0]))


# ------------------------------------------------------------------ student
def with_target(enc, tgt):
    """append the target tokens to a single prompt (teacher forcing)"""
    t = torch.tensor([tgt], device='cuda')
    out = dict(enc)
    out['input_ids'] = torch.cat([enc['input_ids'], t], 1)
    out['attention_mask'] = torch.cat([enc['attention_mask'], torch.ones_like(t)], 1)
    out['mm_token_type_ids'] = torch.cat([enc['mm_token_type_ids'], torch.zeros_like(t)], 1)
    return out


def step_inputs(model, mask, nxt, pkv):
    """one cached decoding step: only the new token goes in; its M-RoPE position is (real tokens so far) + rope delta"""
    pos = (mask.long().sum(-1, keepdim=True) - 1) + model.model.rope_deltas.to(mask.device).view(-1, 1)
    return dict(input_ids=nxt[:, None], attention_mask=mask, past_key_values=pkv, use_cache=True,
                position_ids=pos[None].expand(3, -1, -1), logits_to_keep=1)


@torch.no_grad()
def decode(model, proc, enc, opts, cached=True, max_new=MAX_NEW['state'], trace=None, stop='Answer:'):
    """Greedy decoding of a left-padded batch until every sequence has written 'Answer:'.
    Returns the generated texts and, per sequence, the two option logits read right after its 'Answer:' (None if it
    never wrote one). cached=False recomputes the whole sequence every step (slow; used to check the cache)."""
    tok = proc.tokenizer
    B = enc['input_ids'].shape[0]
    cur = dict(enc); gen = [[] for _ in range(B)]; ans = [None] * B
    out = model(**cur, use_cache=cached, logits_to_keep=1)
    pkv = out.past_key_values if cached else None
    for _ in range(max_new):
        lg = out.logits[:, -1].float()
        if trace is not None:
            trace.append(lg.cpu())
        for b in range(B):
            if ans[b] is None and tok.decode(gen[b]).rstrip().endswith(stop):
                ans[b] = (float(lg[b, opts[b][0]]), float(lg[b, opts[b][1]]))
        if all(a is not None for a in ans):
            break
        nxt = lg.argmax(-1)
        for b in range(B):
            gen[b].append(int(nxt[b]))
        one = torch.ones(B, 1, dtype=cur['attention_mask'].dtype, device='cuda')
        cur['attention_mask'] = torch.cat([cur['attention_mask'], one], 1)
        if cached:
            out = model(**step_inputs(model, cur['attention_mask'], nxt, pkv)); pkv = out.past_key_values
        else:
            cur['input_ids'] = torch.cat([cur['input_ids'], nxt[:, None]], 1)
            cur['mm_token_type_ids'] = torch.cat([cur['mm_token_type_ids'], torch.zeros_like(nxt[:, None])], 1)
            out = model(**cur, logits_to_keep=1)
    return [tok.decode(g) for g in gen], ans


def check_cache(model, proc, split, n_scenes=2):
    """Cached decoding must match full recomputation ON THE SAME TOKENS (free-running greedy paths may split at a
    near-tie, so they are not compared): every step's logits from the cache against one teacher-forced full forward."""
    tok = proc.tokenizer; diffs, agree = [], []
    for i in range(n_scenes):
        sc = sts.load(split, i); fv = sts.vis_feats(model, proc, split, i, sc['frames'])
        qs = sc['questions'][:4]
        enc = sts.encode_cached(proc, [(fv, q['question'] + INSTR) for q in qs])
        tr = []
        texts, _ = decode(model, proc, enc, [(0, 1)] * len(qs), cached=True, max_new=16, trace=tr)
        n = len(tr)                                                  # logits for steps 0..n-1; n-1 tokens were appended
        new = torch.stack([t.argmax(-1) for t in tr[:-1]], 1).to('cuda')   # (B, n-1)
        full = dict(enc)
        full['input_ids'] = torch.cat([enc['input_ids'], new], 1)
        full['attention_mask'] = torch.cat([enc['attention_mask'], torch.ones_like(new)], 1)
        full['mm_token_type_ids'] = torch.cat([enc['mm_token_type_ids'], torch.zeros_like(new)], 1)
        with torch.no_grad():
            ref = model(**full, logits_to_keep=n).logits.float().cpu()      # (B, n, V)
        got = torch.stack(tr, 1)
        diffs.append((ref - got).abs().amax(-1).flatten()); agree.append((ref.argmax(-1) == got.argmax(-1)).float().flatten())
    d = torch.cat(diffs); ag = float(torch.cat(agree).mean())
    print(f'cache check: argmax agreement {100 * ag:.1f}%, logit difference median {float(d.median()):.3f} '
          f'max {float(d.max()):.3f}', flush=True)
    # bf16 noise: median about 0.2, single logits occasionally above 1. A wrong position or mask shifts every step by
    # several units, so the median is the test; the maximum only guards against a gross error.
    return float(d.median()) < 0.5 and float(d.max()) < 4.0


# ------------------------------------------------------------------ scoring
def twin_acc(rows, key):
    """both-twins-correct over the questions whose answer differs between the twins, with a pair bootstrap (m_pairacc)"""
    by = {}
    for r in rows:
        if r.get('differs') and r.get('pair') is not None:
            by.setdefault((r['pair'], r['t'], r['qtype']), []).append(int(r[key]))
    both = {k: int(all(v)) for k, v in by.items() if len(v) == 2}
    if not both:
        return None
    pairs = sorted({k[0] for k in both}); per = {p: [v for k, v in both.items() if k[0] == p] for p in pairs}
    rng = np.random.default_rng(0)
    bs = [100 * np.mean([v for p in rng.choice(pairs, len(pairs)) for v in per[p]]) for _ in range(1000)]
    return dict(acc=round(100 * float(np.mean(list(both.values()))), 1),
                ci=[round(float(np.percentile(bs, q)), 1) for q in (2.5, 97.5)], n=len(both))


def summarize(rows, target):
    s = dict(model=twin_acc(rows, 'correct'), teacher=twin_acc(rows, 'teacher_ok'),
             plain_acc=round(100 * float(np.mean([r['correct'] for r in rows])), 1),
             agree_teacher=round(100 * float(np.mean([r['agree'] for r in rows])), 1))
    if target == 'record':
        rv = [r['rec_vis_acc'] for r in rows if r['rec_vis_acc'] is not None]
        re_ = [e for r in rows for e in (r['rec_errs'] or [])]
        s.update(from_record=twin_acc(rows, 'record_ok'), rec_malformed=int(sum(r['rec_vis_acc'] is None for r in rows)),
                 rec_vis_acc=round(100 * float(np.mean(rv)), 1) if rv else None,
                 rec_err_med_m=round(float(np.median(re_)), 3) if re_ else None,
                 rec_err_p90_m=round(float(np.percentile(re_, 90)), 3) if re_ else None)
    if target == 'motion':
        ok = [r for r in rows if r['cam_err'] is not None]
        med = lambda k: round(float(np.median([r[k] for r in ok])), 3) if ok else None
        s.update(from_motion=twin_acc(rows, 'motion_ok'), mot_malformed=len(rows) - len(ok), cam_speed_err_med=med('cam_err'),
                 yaw_err_med=med('yaw_err'), p7_err_med_m=med('p7_err'), vel_err_med=med('v_err'))
    if target in ('state', 'record', 'motion'):
        errs = [r['err'] for r in rows if r['err'] is not None]
        s.update(written=twin_acc(rows, 'written_ok'), no_answer=int(sum(r['no_answer'] for r in rows)),
                 malformed=int(sum(r['err'] is None for r in rows)),
                 err_med_m=round(float(np.median(errs)), 3) if errs else None,
                 err_p90_m=round(float(np.percentile(errs, 90)), 3) if errs else None,
                 follows_written=round(100 * float(np.mean([r['follows'] for r in rows if r['follows'] is not None] or [0])), 1))
    return s


# ------------------------------------------------------------------ localisation diagnostic
def locate_prompt(names):
    return ('These 8 frames are from a video taken by a moving camera, one frame per second. Where are the '
            f'{names[0]} ball and the {names[1]} ball in the last frame? Write each as `colour (x, z)` with x = metres to '
            "the right and z = metres forward in the camera's own axes at the last frame, or `colour not visible` if it "
            'is not in the last frame. Then write `Done`.')


def locate_line(names, tr):
    pos = np.array(tr['pos'], float)[-1]; vis = np.array(tr['vis'], bool)[-1]
    return '; '.join(f'{n} ({pos[j, 0]:+.2f}, {pos[j, 1]:+.2f})' if vis[j] else f'{n} not visible'
                     for j, n in enumerate(names)) + '. Done'


def parse_locate(text):
    """(pos (2, 2), vis (2,)) the student wrote, or None when the two balls cannot be told apart"""
    segs = text.split('Done')[0].split(';')
    if len(segs) != 2:
        return None
    pos = np.zeros((2, 2)); vis = np.zeros(2, bool)
    for j, seg in enumerate(segs):
        m = NUM.search(seg)
        if m and 'not visible' not in seg:
            pos[j] = (float(m.group(1)), float(m.group(2))); vis[j] = True
    return pos, vis


def run_locate(a):
    """How precisely can the language model READ a position, as text? No motion, no extrapolation, no question:
    the student writes where the two asked-about balls are in the LAST frame (or that they are not visible).
    Compared, on the same balls, with the measuring part (the 2.5M network outside the model)."""
    proc, model, params, _ = sts.build(a.seed, None)
    tok = proc.tokenizer; eos = tok.convert_tokens_to_ids('<|im_end|>')
    if a.init_lora:
        sts.swap_lora(params, [x.to('cuda') for x in torch.load(a.init_lora)])
    rng = np.random.default_rng(a.seed); t0 = time.time()
    if a.n_train:
        splits = a.train_split.split(',')
        n_sc = {sp: len([f for f in os.listdir(os.path.join(sts.DATA, sp)) if f.endswith('.npz')]) for sp in splits}
        perm = {sp: rng.permutation(n_sc[sp]) for sp in splits}
        opt = torch.optim.AdamW([{'params': params, 'lr': a.lr}], weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr], total_steps=a.n_train, pct_start=0.05)
        curve = a.out.replace('.jsonl', '') + '_train.jsonl'
        model.train(); logs = []
        for step in range(a.n_train):
            sp = splits[step % len(splits)]; i = int(perm[sp][(step // len(splits)) % n_sc[sp]])
            sc = sts.load(sp, i); f = sts.vis_feats(model, proc, sp, i, sc['frames'])
            tgt = tok.encode(locate_line(sc['names'], sts.scene_tracks(sp, i, sc)), add_special_tokens=False) + [eos]
            enc = with_target(sts.encode_cached(proc, [(f, locate_prompt(sc['names']))]), tgt[:-1])
            lg = model(**enc, logits_to_keep=len(tgt)).logits[0].float()
            loss = F.cross_entropy(lg, torch.tensor(tgt, device='cuda'))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            logs.append(loss.item())
            if (step + 1) % 100 == 0:
                print(f'locate s{a.seed} {step + 1}/{a.n_train} loss {np.mean(logs[-100:]):.3f} {time.time() - t0:.0f}s', flush=True)
                with open(curve, 'a') as fh:
                    fh.write(json.dumps(dict(step=step + 1, loss=round(float(np.mean(logs[-100:])), 4))) + '\n')
        if a.save_lora:
            torch.save([p.detach().cpu() for p in params], a.save_lora)
        del opt, sched, lg, loss, enc
        for p in params:
            p.grad = None
        torch.cuda.empty_cache()
    train_sec = time.time() - t0
    model.eval()
    for split in a.eval_split.split(','):
        n_ev = min(a.n_eval, sts.EVAL_SCENES[split]); rows = []
        est_path = os.path.join(sts.HERE, f'{split}_est_tracks.json')
        est = json.load(open(est_path)) if os.path.exists(est_path) else None
        for b0 in range(0, n_ev, 8):
            if b0 % 80 == 0:
                print(f'eval {split} {b0}/{n_ev} {time.time() - t0:.0f}s', flush=True)
            idx = list(range(b0, min(b0 + 8, n_ev))); scs = [sts.load(split, i) for i in idx]
            enc = sts.encode_cached(proc, [(sts.vis_feats(model, proc, split, i, sc['frames']), locate_prompt(sc['names']))
                                           for i, sc in zip(idx, scs)])
            texts, _ = decode(model, proc, enc, [(0, 1)] * len(idx), cached=not a.no_kv, max_new=40, stop='Done')
            for i, sc, text in zip(idx, scs, texts):
                text = text.split('Done')[0] + 'Done' if 'Done' in text else text   # drop what a finished sequence wrote while the batch went on
                tr = sts.scene_tracks(split, i, sc); tpos = np.array(tr['pos'], float)[-1]; tvis = np.array(tr['vis'], bool)[-1]
                w = parse_locate(text)
                for j in range(2):
                    row = dict(scene=i, ball=sc['names'][j], true_vis=bool(tvis[j]), text=text, wrote_vis=None, err=None, est_err=None)
                    if w is not None:
                        row['wrote_vis'] = bool(w[1][j])
                        if tvis[j] and w[1][j]:
                            row['err'] = float(np.linalg.norm(w[0][j] - tpos[j]))
                    if est is not None and tvis[j] and np.array(est[i]['vis'], bool)[-1][j]:
                        row['est_err'] = float(np.linalg.norm(np.array(est[i]['pos'], float)[-1][j] - tpos[j]))
                    rows.append(row)
        ok = [r for r in rows if r['wrote_vis'] is not None]
        errs = [r['err'] for r in rows if r['err'] is not None]; ee = [r['est_err'] for r in rows if r['est_err'] is not None]
        q = lambda v, p: round(float(np.percentile(v, p)), 3) if v else None
        summ = dict(balls=len(rows), visible=int(sum(r['true_vis'] for r in rows)), malformed=len(rows) - len(ok),
                    vis_acc=round(100 * float(np.mean([r['wrote_vis'] == r['true_vis'] for r in ok])), 1) if ok else None,
                    err_med_m=q(errs, 50), err_p90_m=q(errs, 90), n_err=len(errs),
                    measuring_part_err_med_m=q(ee, 50), measuring_part_err_p90_m=q(ee, 90))
        res = dict(arm='locate', seed=a.seed, split=split, train_split=a.train_split, n_train=a.n_train,
                   init_lora=a.init_lora, train_sec=round(train_sec), summary=summ, rows=rows)
        with open(a.out, 'a') as fh:
            fh.write(json.dumps(res) + '\n')
        print(json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
        for r in rows[:6:2]:
            print('  wrote:', repr(r['text']), flush=True)


# ------------------------------------------------------------------ main
def run(a):
    if a.target == 'locate':
        return run_locate(a)
    proc, model, params, _ = sts.build(a.seed, None)
    tok = proc.tokenizer
    if a.init_lora:
        sts.swap_lora(params, [v.to('cuda') for v in torch.load(a.init_lora)])
    eos = tok.convert_tokens_to_ids('<|im_end|>')
    teacher = Teacher(a.teacher)
    writes = a.target in ('state', 'record', 'motion')
    instr = dict(state=INSTR, record=INSTR_REC, motion=INSTR_MOT, answer='')[a.target]
    ask = lambda q, tr: (odo_text(tr) if a.odo else '') + q['question'] + instr
    P_rec = json.load(open(os.path.join(sts.HERE, 'state_rules_results.json')))['ca/est']['params']   # for a noisy written record

    def target_text(names, tr, r, p7, v):
        head = record_line(names, tr) if a.target == 'record' else motion_line(names, tr, p7, v) if a.target == 'motion' else ''
        return head + state_line(names, r)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    wb = None
    if a.wandb:   # optional: needs `pip install wandb` and `wandb login` done by the user
        import wandb
        wb = wandb.init(project=a.wandb, name=os.path.basename(a.out).replace('.jsonl', ''), config=vars(a))

    if a.n_train:
        splits = a.train_split.split(',')
        n_sc = {sp: len([f for f in os.listdir(os.path.join(sts.DATA, sp)) if f.endswith('.npz')]) for sp in splits}
        perm = {sp: rng.permutation(n_sc[sp]) for sp in splits}
        opt = torch.optim.AdamW([{'params': params, 'lr': a.lr}], weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr], total_steps=a.n_train, pct_start=0.05)
        model.train(); logs = []; t_ok = []; n_tgt = 0; start = 0; lg = ce = enc = None
        # nothing is lost when a run dies: a rolling checkpoint (LoRA + optimizer + schedule + sampling state) every
        # --ckpt_every steps, and the loss curve appended to <out>_train.jsonl every 100 steps. --resume continues.
        ck = (a.save_lora or a.out) + '.ckpt'; curve = a.out.replace('.jsonl', '') + '_train.jsonl'
        if a.resume and os.path.exists(ck):
            c = torch.load(ck, weights_only=False)
            sts.swap_lora(params, [x.to('cuda') for x in c['lora']]); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched'])
            rng.bit_generator.state = c['rng']; logs, t_ok, n_tgt, start = c['logs'], c['t_ok'], c['n_tgt'], c['step']
            print(f'resumed training from {ck} at step {start}', flush=True)
        for step in range(start, a.n_train):
            sp = splits[step % len(splits)]; i = int(perm[sp][(step // len(splits)) % n_sc[sp]])
            sc = sts.load(sp, i); f = sts.vis_feats(model, proc, sp, i, sc['frames'])
            qs = sts.pick_questions([q for q in sc['questions'] if q['t'] in sts.TRAIN_TIMES], 'uniform', rng)
            teach = teacher.scene(sp, i, sc); p7, v, tr = teacher.state(sp, i, sc)
            opt.zero_grad(); l_all = l_ans = 0.0
            for q in qs:   # one question per forward/backward: fits 16 GB (same averaged gradient as a batch)
                y, r = teach(q); oid = sts.opt_ids(proc, q); t_ok.append(int(y == q['label']))
                tgt = (tok.encode(target_text(sc['names'], tr, r, p7, v), add_special_tokens=False) if writes else []) + [oid[y]]
                tgt = tgt + [eos] if writes else tgt
                n_tgt = max(n_tgt, len(tgt))
                enc = with_target(sts.encode_cached(proc, [(f, ask(q, tr))]), tgt[:-1]) if len(tgt) > 1 else \
                    sts.encode_cached(proc, [(f, ask(q, tr))])
                lg = model(**enc, logits_to_keep=len(tgt)).logits[0].float()        # predicts tgt[0..]
                ce = F.cross_entropy(lg, torch.tensor(tgt, device='cuda'), reduction='none')
                k_ans = len(tgt) - 2 if writes else 0                               # position of the answer word
                (ce.mean() / len(qs)).backward()
                l_all += ce.mean().item() / len(qs); l_ans += ce[k_ans].item() / len(qs)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            logs.append([l_all, l_ans])
            if (step + 1) % 100 == 0:
                m = np.mean(logs[-100:], 0)
                print(f'{a.target} s{a.seed} {step + 1}/{a.n_train} loss {m[0]:.3f} ans {m[1]:.3f} '
                      f'teacher_ok {100 * np.mean(t_ok):.1f} longest_target {n_tgt} {time.time() - t0:.0f}s', flush=True)
                point = dict(step=step + 1, loss=round(float(m[0]), 4), ans_loss=round(float(m[1]), 4), lr=sched.get_last_lr()[0])
                with open(curve, 'a') as fh:
                    fh.write(json.dumps(point) + '\n')
                if wb:
                    wb.log(point, step=step + 1)
            if (step + 1) % a.ckpt_every == 0 or step + 1 == a.n_train:
                torch.save(dict(step=step + 1, lora=[p.detach().cpu() for p in params], opt=opt.state_dict(),
                                sched=sched.state_dict(), rng=rng.bit_generator.state, logs=logs, t_ok=t_ok, n_tgt=n_tgt), ck)
        if a.save_lora:
            torch.save([p.detach().cpu() for p in params], a.save_lora)
        del opt, sched, lg, ce, enc   # training memory must not stay reserved during evaluation: on 16 GB the two
        for p in params:              # together overflow into shared system memory and decoding slows to a crawl
            p.grad = None
        torch.cuda.empty_cache()
    train_sec = time.time() - t0

    model.eval()
    if writes and not check_cache(model, proc, a.eval_split):
        print('WARNING: cached decoding differs from full recomputation; evaluating without the cache (slow)', flush=True)
        a.no_kv = True
    n_ev = min(a.n_eval, sts.EVAL_SCENES[a.eval_split]); rows = []
    # every finished scene is appended to <out>.partial, so a stalled or killed evaluation continues with --resume
    part = a.out + '.partial'; done = set()
    sig = dict(target=a.target, odo=a.odo, teacher=a.teacher, split=a.eval_split, lora=a.save_lora or a.init_lora,
               n_train=a.n_train, differs_only=a.differs_only)
    if a.resume and os.path.exists(part):
        lines = [json.loads(l) for l in open(part) if l.strip()]
        if lines and lines[0] == sig:
            for l in lines[1:]:
                rows += l['rows']; done.add(l['scene'])
            print(f'resumed evaluation: {len(done)} scenes already done', flush=True)
    if not done:
        with open(part, 'w') as fh:
            fh.write(json.dumps(sig) + '\n')

    def keep(i, n0):
        with open(part, 'a') as fh:
            fh.write(json.dumps(dict(scene=i, rows=rows[n0:])) + '\n')
    for i in range(n_ev):
        if i in done:
            continue
        n0 = len(rows)
        if i % 50 == 0:
            print(f'eval {a.eval_split} {i}/{n_ev} {time.time() - t0:.0f}s '
                  f'gpu {torch.cuda.memory_reserved() / 2 ** 30:.1f} GB', flush=True)
            torch.cuda.empty_cache()
        sc = sts.load(a.eval_split, i); fv = sts.vis_feats(model, proc, a.eval_split, i, sc['frames'])
        qs = [q for q in sc['questions'] if q.get('differs')] if a.differs_only else sc['questions']
        if not qs:
            keep(i, n0)
            continue
        teach = teacher.scene(a.eval_split, i, sc); p7_t, v_t, tr = teacher.state(a.eval_split, i, sc)
        true = sts.scene_tracks(a.eval_split, i, sc); tpos = np.array(true['pos'], float); tvis = np.array(true['vis'], bool)
        opts = [sts.opt_ids(proc, q) for q in qs]
        enc = sts.encode_cached(proc, [(fv, ask(q, tr)) for q in qs])
        if writes:
            texts, ans = decode(model, proc, enc, opts, cached=not a.no_kv, max_new=MAX_NEW[a.target])
        else:
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            texts = [''] * len(qs); ans = [(float(lg[k, o[0]]), float(lg[k, o[1]])) for k, o in enumerate(opts)]
        for k, q in enumerate(qs):
            y_t, _ = teach(q)
            pred = None if ans[k] is None else int(ans[k][1] > ans[k][0])
            w = parse_state(texts[k]) if writes else None
            rec = parse_record(texts[k]) if a.target == 'record' else None
            rec_row = dict(rec_vis_acc=None, rec_errs=None, record_ok=0)
            if rec is not None:   # the record against the true tracks, and the calculator run on the student's own record
                both_seen = rec[1] & tvis
                y_rec, _, _ = SR.predict('ca', P_rec, dict(pos=rec[0], vis=rec[1], cam_speed=tr['cam_speed'], yaw_rate=tr['yaw_rate']), q)
                rec_row = dict(rec_vis_acc=float((rec[1] == tvis).mean()),
                               rec_errs=[round(float(e), 3) for e in np.linalg.norm(rec[0] - tpos, axis=-1)[both_seen]],
                               record_ok=int(y_rec == q['label']))
            mot = parse_motion(texts[k]) if a.target == 'motion' else None
            if a.target == 'motion':   # the written motion against the teacher's, and the calculator's propagation applied
                rec_row = dict(cam_err=None, yaw_err=None, p7_err=None, v_err=None, motion_ok=0)   # to the student's own numbers
                if mot is not None:
                    y_m, _ = SR.answer(mot[2], mot[3], dict(cam_speed=mot[0], yaw_rate=mot[1]), q)
                    rec_row = dict(cam_err=abs(mot[0] - tr['cam_speed']), yaw_err=abs(mot[1] - tr['yaw_rate']),
                                   p7_err=float(np.linalg.norm(mot[2] - p7_t, axis=1).mean()),
                                   v_err=float(np.linalg.norm(mot[3] - v_t, axis=1).mean()), motion_ok=int(y_m == q['label']))
            rows.append(dict(**rec_row, mode='video', scene=i, pair=sc.get('pair'), twin=sc.get('twin'), t=q['t'], qtype=q['qtype'],
                             label=q['label'], correct=int(pred == q['label']), differs=q.get('differs'),
                             static_ok=q['static_ok'], teacher_ok=int(y_t == q['label']), agree=int(pred == y_t),
                             no_answer=int(pred is None), text=texts[k],
                             written_ok=int(w is not None and relation(w, q) == q['label']),
                             follows=None if (w is None or pred is None) else int(pred == relation(w, q)),
                             err=None if w is None else float(np.linalg.norm(w - np.array(q['r']), axis=1).mean())))
        keep(i, n0)
    summ = summarize(rows, a.target)
    res = dict(arm=f'distill_{a.target}_{a.teacher}' + ('_odo' if a.odo else ''), seed=a.seed, split=a.eval_split, train_split=a.train_split,
               n_train=a.n_train, init_lora=a.init_lora, train_sec=round(train_sec), summary=summ, rows=rows)
    with open(a.out, 'a') as fh:
        fh.write(json.dumps(res) + '\n')
    os.remove(part)   # the full result is written; the partial file is no longer needed
    if wb:
        wb.summary.update({f'{k}_twin_acc': v['acc'] for k, v in summ.items() if isinstance(v, dict) and 'acc' in v})
        wb.summary.update({k: v for k, v in summ.items() if not isinstance(v, dict)}); wb.finish()
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
    for r in rows[:4] if writes else []:
        print('  wrote:', repr(r['text']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='state', choices=['state', 'answer', 'record', 'motion', 'locate'])
    ap.add_argument('--odo', action='store_true', help="put the camera's own speed and yaw rate in the prompt (known to the system)")
    ap.add_argument('--teacher', default='orc', choices=['orc', 'est'], help='tracks the calculator reads: exact or estimated')
    ap.add_argument('--train_split', default='train3,train8,train9'); ap.add_argument('--eval_split', default='eval9')
    ap.add_argument('--n_train', type=int, default=3000); ap.add_argument('--n_eval', type=int, default=600)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init_lora', default='w1k_s0.pt'); ap.add_argument('--save_lora', default='')
    ap.add_argument('--ckpt_every', type=int, default=250, help='steps between rolling training checkpoints')
    ap.add_argument('--resume', action='store_true', help='continue training from the checkpoint and evaluation from the partial file')
    ap.add_argument('--wandb', default='', help='Weights & Biases project name; empty = local files only')
    ap.add_argument('--differs_only', action='store_true',
                    help='evaluate only the questions whose answer differs between the twins (all that twin accuracy uses)')
    ap.add_argument('--no_kv', action='store_true', help='decode without the key-value cache (slow)')
    ap.add_argument('--out', default=os.path.join(sts.HERE, 'res_distill.jsonl'))
    run(ap.parse_args())
