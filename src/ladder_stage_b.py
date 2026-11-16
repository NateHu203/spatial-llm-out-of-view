"""Oracle-to-model ladder, Stage B: frozen-model interface ceiling on the exit twins (v0.2/next_steps_audit.md §6).
Same trained model W (w1k_s0.pt, evaluate-only), same 8 frames, same questions; only a text prefix changes:
  B0 video only
  B1 video + OBSERVED state as numeric text: each ball's last-seen time/position and its velocity estimated from the
     last two distinct VISIBLE frames (no hidden-frame access), plus camera speed/yaw rate (ego-motion provided)
  B2 video + the same observed estimate PROPAGATED to the asked time, in the camera frame at that time (numeric text)
  B3 video + the correct relation as text: an answer-bearing PARSER CEILING only, never an accuracy claim
  (B4 = deterministic executor, no VLM: reported by ladder_stage_a.py)
Usage: python ladder_stage_b.py --lora w1k_s0.pt [--split eval8] [--arms B0,B1,B2,B3]"""
import argparse, json, os, time
import numpy as np, torch
import sts
from ladder_stage_a import est_velocity


def state_text(tr, names):
    pos = np.array(tr['pos'], float); vis = np.array(tr['vis'], bool)
    parts = []
    for j, n in enumerate(names):
        v, k, nv = est_velocity(pos, vis, j, True)
        mv = f'moving at ({v[0]:+.2f}, {v[1]:+.2f}) m/s' if nv >= 2 else 'velocity unknown (seen in one frame only)'
        parts.append(f'the {n} ball was last seen at t={k} s at ({pos[k, j, 0]:+.2f}, {pos[k, j, 1]:+.2f}), {mv}')
    return ('Measurements from the visible frames (x = metres to the right, z = metres forward, in the camera axes of the '
            'last frame t=7 s): ' + '; '.join(parts) +
            f'. Camera forward speed {tr["cam_speed"]:.2f} m/s, yaw rate {tr["yaw_rate"]:+.3f} rad/s. ')


def propagated_text(tr, names, q):
    """Same observed-only estimator and propagation as the B4 executor (ladder_stage_a.predict)."""
    pos =np.array(tr['pos'], float); vis = np.array(tr['vis'], bool)
    p7, vv = [], []
    for j in range(2):
        v, k, _ = est_velocity(pos, vis, j, True)
        p7.append(pos[k, j] + (7 - k) * v); vv.append(v)
    r = sts.transition(torch.tensor(np.array(p7)).float(), torch.tensor(np.array(vv)).float(),
                       torch.tensor(float(tr['cam_speed'])), torch.tensor(float(tr['yaw_rate'])),
                       torch.tensor(q['t'] - 7.0)).numpy()
    return ('Computed positions at the asked time, in the camera\'s own axes at that time (x = metres to the right, '
            f'z = metres forward): the {names[0]} ball at ({r[0, 0]:+.2f}, {r[0, 1]:+.2f}), the {names[1]} ball at '
            f'({r[1, 0]:+.2f}, {r[1, 1]:+.2f}). ')


def relation_text(names, q):
    if q['qtype'] == 'leftright':
        return f'Hint: at that time the {names[0]} ball is to the {q["options"][q["label"]]} of the {names[1]} ball. '
    near = q['options'][q['label']]; far = q['options'][1 - q['label']]
    return f'Hint: at that time the {near} ball is closer to the camera than the {far} ball. '


def prefix(arm, sc, q):
    if arm == 'B0':
        return ''
    if arm == 'B1':
        return state_text(sc['tracks'], sc['names'])
    if arm == 'B2':
        return propagated_text(sc['tracks'], sc['names'], q)
    return relation_text(sc['names'], q)


def summarize(rows, obs):
    out = {}
    rng = np.random.default_rng(0)
    for arm in sorted({r['arm'] for r in rows}):
        for sub, keep in (('all_pairs', lambda r: True), ('observable', lambda r: obs[r['pair']])):
            rs = [r for r in rows if r['arm'] == arm and keep(r)]
            d = [r for r in rs if r['differs']]
            by = {}
            for r in d:
                by.setdefault((r['pair'], r['qi']), []).append(r['correct'])
            both = {k: int(all(v)) for k, v in by.items() if len(v) == 2}
            # pair-level bootstrap CI for twin both-correct
            pairs = sorted({k[0] for k in both})
            per = {p: [v for k, v in both.items() if k[0] == p] for p in pairs}
            bs = []
            for _ in range(1000):
                smp = rng.choice(pairs, len(pairs))
                vals = [v for p in smp for v in per[p]]
                bs.append(100 * np.mean(vals))
            out[f'{arm}/{sub}'] = dict(
                all=round(100 * np.mean([r['correct'] for r in rs]), 1),
                differ=round(100 * np.mean([r['correct'] for r in d]), 1),
                both=round(100 * np.mean(list(both.values())), 1), both_ci=[round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                static_ok=round(100 * np.mean([r['correct'] for r in rs if r['static_ok']]), 1),
                static_wrong=round(100 * np.mean([r['correct'] for r in rs if not r['static_ok']]), 1),
                n_q=len(rs), n_twin_q=len(both))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', default='w1k_s0.pt'); ap.add_argument('--split', default='eval8')
    ap.add_argument('--arms', default='B0,B1,B2,B3'); ap.add_argument('--out', default='ladder_stage_b')
    a = ap.parse_args()
    arms = a.arms.split(',')
    obs = {p['pair']: p['observable'] for p in json.load(open(os.path.join(sts.HERE, f'ladder_stage_a_{a.split}.json')))['pairs']}
    proc, model, params, _ = sts.build(0, None)
    if a.lora:
        sts.swap_lora(params, [v.to('cuda') for v in torch.load(a.lora)])
    model.eval()
    rows = []; t0 = time.time()
    n = sts.EVAL_SCENES[a.split]
    with torch.no_grad():
        for i in range(n):
            if i % 50 == 0:
                print(f'{a.split} {i}/{n} {time.time() - t0:.0f}s', flush=True)
            sc = sts.load(a.split, i)
            fv = sts.vis_feats(model, proc, a.split, i, sc['frames'])
            for arm in arms:
                qs = sc['questions']
                enc = sts.encode_cached(proc, [(fv, prefix(arm, sc, q) + q['question']) for q in qs])
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
                for qi, q in enumerate(qs):
                    o = sts.opt_ids(proc, q)
                    pred = int(lg[qi, o[1]] > lg[qi, o[0]])
                    rows.append(dict(arm=arm, scene=i, pair=sc['pair'], twin=sc['twin'], qi=qi, t=q['t'], qtype=q['qtype'],
                                     label=q['label'], correct=int(pred == q['label']), differs=q['differs'],
                                     static_ok=q['static_ok'], margin=float(lg[qi, o[1]] - lg[qi, o[0]])))
    summ = summarize(rows, obs)
    print(json.dumps(summ, indent=1))
    json.dump(dict(lora=a.lora, split=a.split, sec=round(time.time() - t0), summary=summ, rows=rows),
              open(os.path.join(sts.HERE, f'{a.out}_{a.split}.json'), 'w'))


if __name__ == '__main__':
    main()
