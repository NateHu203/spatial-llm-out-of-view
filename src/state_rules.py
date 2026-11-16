"""Stage M: causal, bounded per-object state-update rules, all read by the same calculator (B4 propagation + relation).
Every rule sees, frame by frame, only (t, visible?, position in the fixed t=7 axes) of one object and keeps a bounded
state; at the end it outputs p7 (position at t=7) and v (velocity), propagated to t* exactly as in Stages A-R.
  anchor : last sighting, v = 0                                (last-seen anchor)
  last2  : velocity from the last two sightings                (Stages A-C)
  lsq    : least-squares line over all sightings (10 numbers)  (Stages D/R)
  kf     : constant-velocity Kalman filter (q, r)
  imm    : interacting multiple models: CV-low, stationary, CV-high (manoeuvre / jump) (q_lo, q_hi, r, p_stay)
  event  : rest/move event rule (OCC4M-style): v = last-two velocity if its speed > v_rest, else rest at the last sighting
  ca     : PROPOSED change-aware segmented least squares: sufficient statistics of the current motion segment + a
           3-observation buffer; a split test (residual reduction > kappa sigma^2) restarts the segment on the last W=2/3
           sightings; read-out = line of the current segment, 'settled' (v = 0, mean position) if |slope| < v_rest.
Usage: python state_rules.py   (tunes every rule on train8+train9, evaluates eval8, eval9, eval3; oracle and estimated
tracks) -> state_rules_results.json"""
import itertools, json, math, os
import numpy as np
import sts, ladder_stage_a as A

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace


# ------------------------------------------------------------------ rules: (pos (8,2), vis (8,), params) -> p7, v, info
def _fallback(pos, vis):
    return pos[7], np.zeros(2), dict(mode='unseen')


def r_anchor(pos, vis, P):
    s = np.nonzero(vis)[0]
    return (pos[s[-1]], np.zeros(2), dict(mode='anchor')) if len(s) else _fallback(pos, vis)


def r_last2(pos, vis, P):
    s = np.nonzero(vis)[0]
    if not len(s):
        return _fallback(pos, vis)
    k = s[-1]; v = (pos[k] - pos[s[-2]]) / (k - s[-2]) if len(s) >= 2 else np.zeros(2)
    return pos[k] + (7 - k) * v, v, dict(mode='moving')


def r_lsq(pos, vis, P):
    s = np.nonzero(vis)[0]
    if not len(s):
        return _fallback(pos, vis)
    k = s[-1]
    if len(s) < 2:
        return pos[k], np.zeros(2), dict(mode='unknown')
    t = s.astype(float); m = np.linalg.lstsq(np.stack([t, np.ones_like(t)], 1), pos[s], rcond=None)[0][0]
    return pos[k] + (7 - k) * m, m, dict(mode='moving')


def r_lsq_fit(pos, vis, P):
    """2026-11-05: one least-squares line over all sightings, read out at its FITTED value (same read-out as ca)"""
    s = np.nonzero(vis)[0]
    if not len(s):
        return _fallback(pos, vis)
    if len(s) < 2:
        return pos[s[-1]], np.zeros(2), dict(mode='unknown')
    t = s.astype(float); m, a0 = np.linalg.lstsq(np.stack([t, np.ones_like(t)], 1), pos[s], rcond=None)[0]
    return a0 + 7 * m, m, dict(mode='moving')


def _kf_mats(q, dt=1.0, stationary=False):
    I = np.eye(2); Z = np.zeros((2, 2))
    if stationary:
        F = np.block([[I, Z], [Z, Z]]); Q = np.block([[q * I, Z], [Z, 1e-6 * I]])
    else:
        F = np.block([[I, dt * I], [Z, I]])
        Q = q * np.block([[dt ** 3 / 3 * I, dt ** 2 / 2 * I], [dt ** 2 / 2 * I, dt * I]])
    return F, Q


H = np.hstack([np.eye(2), np.zeros((2, 2))])


def r_kf(pos, vis, P):
    s = np.nonzero(vis)[0]
    if not len(s):
        return _fallback(pos, vis)
    F, Q = _kf_mats(P['q']); R = P['r'] * np.eye(2)
    x = np.r_[pos[s[0]], 0, 0]; C = np.diag([P['r'], P['r'], 0.25, 0.25])
    for t in range(s[0] + 1, 8):
        x = F @ x; C = F @ C @ F.T + Q
        if vis[t]:
            S = H @ C @ H.T + R; K = C @ H.T @ np.linalg.inv(S)
            x = x + K @ (pos[t] - H @ x); C = (np.eye(4) - K @ H) @ C
    return x[:2], x[2:], dict(mode='moving')


def r_imm(pos, vis, P):
    s = np.nonzero(vis)[0]
    if not len(s):
        return _fallback(pos, vis)
    models = [_kf_mats(P['q_lo']), _kf_mats(P['q_lo'], stationary=True), _kf_mats(P['q_hi'])]
    M = len(models); ps = P['p_stay']
    PI = np.full((M, M), (1 - ps) / (M - 1)); np.fill_diagonal(PI, ps)
    R = P['r'] * np.eye(2)
    xs = [np.r_[pos[s[0]], 0, 0] for _ in range(M)]
    Cs = [np.diag([P['r'], P['r'], 0.25, 0.25]) for _ in range(M)]
    mu = np.full(M, 1.0 / M)
    for t in range(s[0] + 1, 8):
        cbar = PI.T @ mu                                            # mixing
        w = (PI * mu[:, None]) / np.maximum(cbar[None], 1e-300)    # w[i, j] = P(model i before | j now)
        x0 = [sum(w[i, j] * xs[i] for i in range(M)) for j in range(M)]
        C0 = [sum(w[i, j] * (Cs[i] + np.outer(xs[i] - x0[j], xs[i] - x0[j])) for i in range(M)) for j in range(M)]
        lik = np.ones(M)
        for j, (F, Q) in enumerate(models):
            x = F @ x0[j]; C = F @ C0[j] @ F.T + Q
            if vis[t]:
                S = H @ C @ H.T + R; e = pos[t] - H @ x
                lik[j] = math.exp(-0.5 * e @ np.linalg.solve(S, e)) / (2 * math.pi * math.sqrt(np.linalg.det(S)))
                K = C @ H.T @ np.linalg.inv(S); x = x + K @ e; C = (np.eye(4) - K @ H) @ C
            xs[j], Cs[j] = x, C
        mu = cbar * lik; mu = mu / mu.sum() if mu.sum() > 0 else np.full(M, 1.0 / M)
    x = sum(m * xx for m, xx in zip(mu, xs))
    return x[:2], x[2:], dict(mode=['moving', 'settled', 'manoeuvre'][int(np.argmax(mu))])


def r_event(pos, vis, P):
    p7, v, _ = r_last2(pos, vis, P)
    s = np.nonzero(vis)[0]
    if len(s) and np.linalg.norm(v) < P['v_rest']:
        return pos[s[-1]], np.zeros(2), dict(mode='settled')
    return p7, v, dict(mode='moving')


def seg_stats(obs):
    """8 sums of a set of sightings: n, St, Stt, Sx (2), Stx (2), Sxx (= sum |x|^2)"""
    z = np.zeros(8)
    for t, x in obs:
        z += np.r_[1, t, t * t, x, t * x, x @ x]
    return z


def line(z):
    """least-squares line x = a + m t through the sightings summarised by z -> (m, a)"""
    n, St, Stt, Sx, Stx = z[0], z[1], z[2], z[3:5], z[5:7]
    d = n * Stt - St * St
    m = (n * Stx - St * Sx) / d if d > 1e-9 else np.zeros(2)
    return m, (Sx - m * St) / max(n, 1)


def rss(z):
    """residual sum of squares of that line, from the sums only:
    sum |x - a - m t|^2 = Sxx - 2 a.Sx - 2 m.Stx + n |a|^2 + 2 St a.m + Stt |m|^2"""
    n, St, Stt, Sx, Stx, Sxx = z[0], z[1], z[2], z[3:5], z[5:7], z[7]
    if n <= 2:
        return 0.0
    m, a = line(z)
    return float(Sxx - 2 * a @ Sx - 2 * m @ Stx + n * a @ a + 2 * St * (a @ m) + Stt * (m @ m))


class CAState:
    """change-aware segmented least squares. Bounded state: 8 sums of the current motion segment, the last 3
    sightings, the mode. Per sighting: add it; split test 'do the last W sightings (W = 2, 3) fit a separate line much
    better?' (residual reduction > kappa * sigma^2); if so, the segment restarts on those W sightings (the obsolete
    trajectory is forgotten). Read-out: line of the current segment; 'settled' (v = 0, mean position) if |slope| < v_rest."""

    def __init__(self, P):
        self.P = P; self.L = np.zeros(8); self.buf = []; self.changes = []

    def update(self, t, vis, x):
        if not vis:
            return
        o = (float(t), np.asarray(x, float))
        self.buf = (self.buf + [o])[-3:]
        self.L = self.L + seg_stats([o])
        best, bw = 0.0, None
        for W in (2, 3):
            if len(self.buf) < W or self.L[0] < W + 1:
                continue
            new = seg_stats(self.buf[-W:]); old = self.L - new
            gain = rss(self.L) - rss(old) - rss(new)
            if gain > self.P['kappa'] * self.P['sigma'] ** 2 and gain > best:
                best, bw = gain, W
        if bw is not None:
            self.L = seg_stats(self.buf[-bw:]); self.changes.append(t)

    def readout(self):
        n = self.L[0]
        if n == 0:
            return None
        if n < 2:
            return self.buf[-1][1], np.zeros(2), 'unknown'
        m, a = line(self.L)
        if np.linalg.norm(m) < self.P['v_rest']:
            return self.L[3:5] / n, np.zeros(2), 'settled'
        return a + 7 * m, m, 'moving'


def r_ca(pos, vis, P, info=None):
    st = CAState(P)
    for t in range(8):
        st.update(t, vis[t], pos[t])
    out = st.readout()
    if out is None:
        return _fallback(pos, vis)
    return out[0], out[1], dict(mode=out[2], changes=st.changes)


RULES = dict(anchor=(r_anchor, {}), last2=(r_last2, {}), lsq=(r_lsq, {}), lsq_fit=(r_lsq_fit, {}),
             kf=(r_kf, dict(q=[1e-4, 1e-3, 1e-2, 3e-2, 1e-1], r=[1e-4, 1e-3, 1e-2, 3e-2])),
             imm=(r_imm, dict(q_lo=[1e-4, 1e-3], q_hi=[0.1, 0.5, 2.0], r=[1e-4, 1e-3, 1e-2, 3e-2], p_stay=[0.8, 0.9, 0.97])),
             event=(r_event, dict(v_rest=[0.03, 0.06, 0.1, 0.15])),
             ca=(r_ca, dict(sigma=[0.01, 0.05, 0.1, 0.15], kappa=[4, 9, 16, 25], v_rest=[0.0, 0.03, 0.06, 0.1])))


# ------------------------------------------------------------------ calculator
def transition_np(p, v, s, w, dt):
    """numpy copy of sts.transition: positions (2,2) in the camera frame at 7 + dt (camera arc with speed s, yaw rate w)"""
    if abs(w) < 1e-4:
        c = np.array([0.0, s * dt])
    else:
        c = np.array([s / w * (1 - math.cos(w * dt)), s / w * math.sin(w * dt)])
    d = p + v * dt - c[None]; a = w * dt
    return np.stack([d[:, 0] * math.cos(a) - d[:, 1] * math.sin(a), d[:, 0] * math.sin(a) + d[:, 1] * math.cos(a)], -1)


def scene_state(rule, P, tr):
    pos = np.array(tr['pos'], float); vis = np.array(tr['vis'], bool)
    outs = [RULES[rule][0](pos[:, j], vis[:, j], P) for j in range(2)]
    return np.array([o[0] for o in outs]), np.array([o[1] for o in outs]), [o[2] for o in outs]


def answer(p7, v, tr, q):
    r = transition_np(p7, v, float(tr['cam_speed']), float(tr['yaw_rate']), q['t'] - 7.0)
    return (int(r[0, 0] > r[1, 0]) if q['qtype'] == 'leftright' else int(np.linalg.norm(r[1]) < np.linalg.norm(r[0]))), r


def predict(rule, P, tr, q):
    p7, v, infos = scene_state(rule, P, tr)
    y, r = answer(p7, v, tr, q)
    return y, r, infos


_SC = {}


def scenes_of(split):
    if split not in _SC:
        n = {'eval3': 300}.get(split, 600 if split in ('eval8', 'eval9', 'eval9s', 'eval10', 'eval11', 'eval12', 'eval13', 'train9') else 1200)
        _SC[split] = [sts.load(split, i) for i in range(n)]
        for sc in _SC[split]:
            sc.pop('frames')
    return _SC[split]


def tracks_of(split, src):
    S = scenes_of(split)
    if src == 'orc':
        return [sts.scene_tracks(split, i, sc) for i, sc in enumerate(S)]
    return json.load(open(os.path.join(HERE, f'{split}_est_tracks.json')))


def evaluate(rule, P, split, src, rows_out=False):
    S = scenes_of(split); TR = tracks_of(split, src)
    rows = []
    for i, (sc, tr) in enumerate(zip(S, TR)):
        p7, v, infos = scene_state(rule, P, tr)
        for q in sc['questions']:
            y, _ = answer(p7, v, tr, q)
            rows.append(dict(pair=sc.get('pair', i), twin=sc.get('twin', 0), t=q['t'], qtype=q['qtype'], ok=int(y == q['label']),
                             differs=q.get('differs', False), static_ok=q['static_ok'], hidden=any(sc['hidden_end']),
                             mode=sc.get('mode', 'cv'), exit_mode=infos[sc['exit_slot']]['mode'] if 'exit_slot' in sc else None))
    return rows if rows_out else summarize(rows, split)


def summarize(rows, split):
    out = dict(all=100 * np.mean([r['ok'] for r in rows]))
    if split == 'eval3':
        h = [r['ok'] for r in rows if r['hidden'] and not r['static_ok']]
        out['hiddenSW'] = 100 * np.mean(h)
        return out
    by = {}
    for r in rows:
        if r['differs']:
            by.setdefault((r['pair'], r['t'], r['qtype']), []).append(r)
    both = {k: int(all(x['ok'] for x in v)) for k, v in by.items() if len(v) == 2}
    out['differ'] = 100 * np.mean([r['ok'] for r in rows if r['differs']])
    out['both'] = 100 * np.mean(list(both.values()))
    pairs = sorted({k[0] for k in both}); per = {p: [v for k, v in both.items() if k[0] == p] for p in pairs}
    rng = np.random.default_rng(0)
    bs = [100 * np.mean([v for p in rng.choice(pairs, len(pairs)) for v in per[p]]) for _ in range(1000)]
    out['both_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    if split in ('eval9', 'eval9s', 'eval10', 'eval11', 'eval12', 'eval13', 'train9'):
        mode_of = {r['pair']: r['mode'] for r in rows if r['twin'] == 0}
        for m in ('stop', 'reloc', 'turn'):
            vals = [v for k, v in both.items() if mode_of[k[0]] == m]
            out[f'both_{m}'] = 100 * np.mean(vals)
        for m in ('stop', 'reloc', 'turn', 'cv'):   # per-twin accuracy on differing questions
            out[f'acc_{m}'] = 100 * np.mean([r['ok'] for r in rows if r['differs'] and r['mode'] == m])
    return out


def tune(rule, src):
    fn, grid = RULES[rule]
    if not grid:
        return {}, None
    best = None
    for vals in itertools.product(*grid.values()):
        P = dict(zip(grid.keys(), vals))
        s = 0.5 * (evaluate(rule, P, 'train8', src)['both'] + evaluate(rule, P, 'train9', src)['both'])
        if best is None or s > best[1]:
            best = (P, s)
    return best


def main():
    res = {}
    for src in ('orc', 'est'):
        for rule in RULES:
            P, score = tune(rule, src)
            r = dict(params=P, train_score=score)
            for sp in ('eval8', 'eval9', 'eval3'):
                r[sp] = evaluate(rule, P, sp, src)
            res[f'{rule}/{src}'] = r
            print(rule, src, json.dumps({k: (v if not isinstance(v, dict) else {a: (round(b, 1) if isinstance(b, float) else b)
                                                                            for a, b in v.items()}) for k, v in r.items()}),
                  flush=True)
    json.dump(res, open(os.path.join(HERE, 'state_rules_results.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
