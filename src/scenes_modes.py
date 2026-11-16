"""Stage M (motion modes): exit twins whose exited ball changes behaviour before it leaves the view.
Twin 0 (mode scene): the exited ball a
  stop  : rolls with v1, stops at t_s and rests at p_a (>= 2 visible frames moving, >= 2 at rest);
  reloc : rests at P0, is picked up and placed at p_a between two frames (t_r), rests there (>= 1 / >= 2 visible frames);
  turn  : rolls with v1, turns at t_c to v2 (60-150 deg) and keeps rolling (>= 2 visible frames before / after).
Twin 1 (cv scene): a rolls at one constant velocity through the SAME last-seen position p_a at the same frame k.
Both: a visible at k, invisible at k+1..7; other balls identical => pixel-identical last frames. The future (t* = 7..10)
follows the last behaviour: stop / reloc stay at p_a, turn keeps v2, cv keeps its velocity. Changes that happen while
the ball is hidden are NOT generated (they are unobservable from the frames).
Strict splits (eval9s, eval10; 2026-11-05): additionally ZERO pixels of a in frames k+1..7 of both twins and every frame
after k pixel-identical between the twins (eval9 allowed <=30 px after the exit: 36/300 pairs leaked).
Usage: python scenes_modes.py train9 300 ; python scenes_modes.py eval9 300   (pairs)"""
import json, math, os, sys
import numpy as np
import scenes

_obj_pos_cv = scenes.obj_pos


def obj_pos(scn, t):
    """piecewise trajectories: scn['traj'][k] = [(t0, p0, v), ...] (sorted); objects without traj roll at constant v."""
    P = _obj_pos_cv(scn, t)
    for k, segs in (scn.get('traj') or {}).items():
        t0, p0, v = [s for s in segs if s[0] <= t + 1e-9][-1]
        P[k] = np.asarray(p0) + np.asarray(v) * (t - t0)
    return P


scenes.obj_pos = obj_pos   # render() and _multi_questions() look the function up at call time
MODES = ('stop', 'reloc', 'turn')
HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
DATA = os.path.join(HERE, 'sts_data')


def approx_vis(scn, k, t):
    """fast pinhole check (centre inside the image, in front), used before rendering"""
    cxz, yaw = scenes.cam_pose(scn, t)
    C = np.array([cxz[0], scenes.H_CAM, cxz[1]])
    fwd = np.array([math.sin(yaw) * math.cos(scenes.PITCH), -math.sin(scenes.PITCH), math.cos(yaw) * math.cos(scenes.PITCH)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)]); up = np.cross(fwd, right)
    p = obj_pos(scn, t)[k]; d = np.array([p[0], scenes.R_OBJ, p[1]]) - C
    z = d @ fwd
    if z < 0.4:
        return False
    u = scenes.FOCAL * (d @ right) / z + scenes.IMG / 2; v = -scenes.FOCAL * (d @ up) / z + scenes.IMG / 2
    return -2 < u < scenes.IMG + 2 and -2 < v < scenes.IMG + 2


def prior_v(rng):
    a = rng.uniform(0, 2 * math.pi); s = rng.uniform(0.15, 0.7)
    return np.array([math.cos(a) * s, math.sin(a) * s])


def mode_traj(mode, rng, pa, k):
    """trajectory of a (list of segments) and the change time"""
    tc = int(rng.integers(2, k))            # change frame: >= 2 frames before it (0..tc-1), >= 2 from tc..k
    if mode == 'stop':
        v1 = prior_v(rng)
        return [(0.0, pa - v1 * tc, v1), (float(tc), pa, np.zeros(2))], tc
    if mode == 'reloc':
        ang = rng.uniform(0, 2 * math.pi); d = rng.uniform(1.0, 2.5)
        P0 = pa + d * np.array([math.cos(ang), math.sin(ang)])
        return [(0.0, P0, np.zeros(2)), (float(tc), pa, np.zeros(2))], tc
    v1 = prior_v(rng)
    turn = rng.choice([-1, 1]) * rng.uniform(math.radians(60), math.radians(150))
    s2 = rng.uniform(0.15, 0.7)
    a1 = math.atan2(v1[1], v1[0]) + turn
    v2 = np.array([math.cos(a1) * s2, math.sin(a1) * s2])
    pc = pa - v2 * (k - tc)
    return [(0.0, pc - v1 * tc, v1), (float(tc), pc, v2)], tc


def counts_ok(mode, vis_a, tc, k):
    before, after = vis_a[:tc].sum(), vis_a[tc:k + 1].sum()
    return vis_a[k] and not vis_a[k + 1:].any() and after >= 2 and before >= (1 if mode == 'reloc' else 2)


def sample_mode_twin(seed, mode, times=(7.0, 8.0, 9.0, 10.0), strict=False):
    rng = np.random.default_rng(seed)
    for _ in range(20000):
        scn = dict(yaw0=0.0, yaw_rate=rng.uniform(-0.12, 0.12), cam_speed=rng.uniform(0.15, 0.45),
                   names=list(rng.permutation(scenes.NAMES)[:3]))
        n = 3
        p0 = np.stack([rng.uniform(-1.6, 1.6, n), rng.uniform(2.0, 5.5, n)], -1)
        scn.update(p0=p0, v=np.stack([prior_v(rng) for _ in range(n)]))
        a = int(rng.integers(n)); k = int(rng.integers(4, 7))
        ck, yk = scenes.cam_pose(scn, float(k))
        lat, fwd = rng.uniform(-2.2, 2.2), rng.uniform(1.2, 4.5)
        r_, f_ = np.array([math.cos(yk), -math.sin(yk)]), np.array([math.sin(yk), math.cos(yk)])
        pa = ck + lat * r_ + fwd * f_
        traj, tc = mode_traj(mode, rng, pa, k)
        sm = dict(scn, traj={a: traj})
        av = np.array([approx_vis(sm, a, float(t)) for t in range(8)])
        if not counts_ok(mode, av, tc, k):
            continue
        vc = prior_v(rng)
        sc = dict(scn, traj={a: [(0.0, pa - vc * k, vc)]})
        avc = np.array([approx_vis(sc, a, float(t)) for t in range(8)])
        if not (avc[k] and not avc[k + 1:].any() and avc.sum() >= 2):
            continue
        Pm = np.stack([obj_pos(sm, float(t)) for t in range(8)])
        if min(np.linalg.norm(Pm[t, i] - Pm[t, j]) for t in range(8) for i in range(n) for j in range(i + 1, n)) < 0.7:
            continue
        out = []
        for s in (sm, sc):
            fr, vi, npx = [], [], []
            for t in range(8):
                im, g = scenes.render(s, float(t)); fr.append(im); vi.append(g['visible']); npx.append(g['npix'][a])
            if strict and any(npx[k + 1:]):
                out = []
                break
            out.append((s, fr, np.array(vi)))
        if len(out) < 2:
            continue
        (_, f0, v0), (_, f1, v1) = out
        if strict and not all(np.array_equal(f0[t], f1[t]) for t in range(k + 1, 8)):
            continue
        if not (counts_ok(mode, v0[:, a], tc, k) and v1[k, a] and not v1[k + 1:, a].any() and v1[:, a].sum() >= 2
                and np.array_equal(f0[-1], f1[-1])):
            continue
        bs = [j for j in range(n) if j != a and v0[:, j].any() and v1[:, j].any()]
        if not bs:
            continue
        b = bs[int(rng.integers(len(bs)))]
        pair = (a, b) if rng.random() < 0.5 else (b, a)
        res = []
        for w, (s, fr, vi) in enumerate(out):
            qs, tracks = scenes._multi_questions(s, pair[0], pair[1], vi, times)
            res.append(dict(frames=fr, questions=qs, names=[s['names'][pair[0]], s['names'][pair[1]]], twin=w, tracks=tracks,
                            hidden_end=[bool(not vi[-1, pair[0]]), bool(not vi[-1, pair[1]])], exit_obj=int(a),
                            mode=mode if w == 0 else 'cv', change_t=tc if w == 0 else None, k_last=k,
                            exit_slot=pair.index(a)))
        for q0, q1 in zip(res[0]['questions'], res[1]['questions']):
            q0['differs'] = q1['differs'] = bool(q0['label'] != q1['label'])
        if not any(q['differs'] for q in res[0]['questions']):
            continue
        return res
    raise RuntimeError('no mode twin')


def _one(args):
    split, i = args
    base = {'train9': 210_000_000, 'eval9': 200_000_000, 'eval9s': 200_000_000, 'eval10': 230_000_000, 'eval11': 250_000_000, 'eval12': 270_000_000, 'eval13': 290_000_000}[split]  # eval11: sealed set for round 7 (2026-11-08)
    for tw in sample_mode_twin(base + i, MODES[i % 3], strict=split in ('eval9s', 'eval10', 'eval11', 'eval12', 'eval13')):
        meta = {k: v for k, v in tw.items() if k != 'frames'}
        meta['pair'] = i
        np.savez_compressed(os.path.join(DATA, split, f'{2 * i + tw["twin"]:05d}.npz'), frames=np.stack(tw['frames']),
                            meta=json.dumps(meta))


if __name__ == '__main__':
    from multiprocessing import Pool
    split, n = sys.argv[1], int(sys.argv[2])
    os.makedirs(os.path.join(DATA, split), exist_ok=True)
    todo = [(split, i) for i in range(n) if not os.path.exists(os.path.join(DATA, split, f'{2 * i + 1:05d}.npz'))]
    with Pool(20) as p:
        for c, _ in enumerate(p.imap_unordered(_one, todo, chunksize=4)):
            if c % 50 == 0:
                print(split, c, '/', len(todo), flush=True)
