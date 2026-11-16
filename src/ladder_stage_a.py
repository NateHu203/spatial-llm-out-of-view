"""Oracle-to-model ladder, Stage A: CPU contract audit of the exit twins (eval8). See v0.2/next_steps_audit.md §6.
Checks the pair contract, whether the exited ball's velocity is observable from VISIBLE frames only, and whether a
deterministic executor (B4) reproduces the labels under the generator's motion model, using
  observed : velocity from the last two distinct visible timestamps (no hidden-frame access); <2 sightings -> v = 0
  omniscient: true constant velocity (uses hidden frames; a ceiling, not a method)
Camera kinematics (cam_speed, yaw_rate) are given to both (labelled 'ego-motion provided').
Writes ladder_stage_a.json (per-pair observability, used by Stage B) and prints a summary.
Usage: python ladder_stage_a.py [split]"""
import json, os, sys
import numpy as np, torch
import sts

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'eval8'


VEL = {'mode': 'last2'}   # 'last2': last two visible frames (Stages A-C); 'lsq': least-squares line over all visible frames (Stage D)


def est_velocity(pos, vis, j, observed):
    """(velocity (2,), last visible frame k, number of visible frames) for ball j."""
    seen = np.nonzero(vis[:, j])[0]
    k = int(seen.max()) if len(seen) else 7
    if not observed:
        return pos[1, j] - pos[0, j], k, len(seen)
    if len(seen) < 2:
        return np.zeros(2), k, len(seen)
    if VEL['mode'] == 'lsq':
        t = seen.astype(float)
        coef = np.linalg.lstsq(np.stack([t, np.ones_like(t)], 1), pos[seen, j], rcond=None)[0]
        return coef[0], k, len(seen)
    k0 = int(seen[-2])
    return (pos[k, j] - pos[k0, j]) / (k - k0), k, len(seen)


def predict(tr, q, observed):
    pos = np.array(tr['pos'], float); vis = np.array(tr['vis'], bool)
    p7, vv = [], []
    for j in range(2):
        v, k, _ = est_velocity(pos, vis, j, observed)
        p7.append(pos[k, j] + (7 - k) * v); vv.append(v)
    r = sts.transition(torch.tensor(np.array(p7)).float(), torch.tensor(np.array(vv)).float(),
                       torch.tensor(float(tr['cam_speed'])), torch.tensor(float(tr['yaw_rate'])), torch.tensor(q['t'] - 7.0))
    r = r.numpy()
    if q['qtype'] == 'leftright':
        return int(r[0, 0] > r[1, 0]), r
    return int(np.linalg.norm(r[1]) < np.linalg.norm(r[0])), r


def main():
    n_pairs = sts.EVAL_SCENES[SPLIT] // 2
    pairs, rows = [], []
    for i in range(n_pairs):
        s = [sts.load(SPLIT, 2 * i + w) for w in (0, 1)]
        tr = [x['tracks'] for x in s]
        pos = [np.array(t['pos'], float) for t in tr]; vis = [np.array(t['vis'], bool) for t in tr]
        # exited ball = the tracked ball whose trajectory differs between the twins
        diff = [float(np.abs(pos[0][:, j] - pos[1][:, j]).max()) for j in range(2)]
        a = int(np.argmax(diff))
        ka = [int(np.nonzero(v[:, a])[0].max()) for v in vis]
        nvis_a = [int(v[:, a].sum()) for v in vis]
        nvis_b = [int(v[:, 1 - a].sum()) for v in vis]
        verr = [float(np.linalg.norm(est_velocity(p, v, a, True)[0] - est_velocity(p, v, a, False)[0]))
                for p, v in zip(pos, vis)]
        rec = dict(pair=i, last_frame_equal=bool(np.array_equal(s[0]['frames'][-1], s[1]['frames'][-1])),
                   same_k=ka[0] == ka[1], last_seen_dist=float(np.linalg.norm(pos[0][ka[0], a] - pos[1][ka[1], a])),
                   nvis_a=nvis_a, nvis_b=nvis_b, vel_err_a=verr, hidden_end_a=[bool(not v[-1, a]) for v in vis],
                   b_hidden_end=[bool(not v[-1, 1 - a]) for v in vis],
                   observable=bool(min(nvis_a) >= 2 and min(nvis_b) >= 2))
        pairs.append(rec)
        for qi, (q0, q1) in enumerate(zip(s[0]['questions'], s[1]['questions'])):
            for w, (sc, q) in enumerate(zip(s, (q0, q1))):
                po, _ = predict(sc['tracks'], q, True); pn, rn = predict(sc['tracks'], q, False)
                rows.append(dict(pair=i, twin=w, qi=qi, t=q['t'], qtype=q['qtype'], label=q['label'], differs=q['differs'],
                                 static_ok=q['static_ok'], obs_ok=int(po == q['label']), omni_ok=int(pn == q['label']),
                                 omni_r_err=float(np.abs(rn - np.array(q['r'])).max()), observable=rec['observable']))

    def summ(rs, key):
        d = [r for r in rs if r['differs']]
        by = {}
        for r in d:
            by.setdefault((r['pair'], r['qi']), []).append(r[key])
        both = [int(all(v)) for v in by.values() if len(v) == 2]
        return dict(n_q=len(rs), all=round(100 * np.mean([r[key] for r in rs]), 1), n_differ=len(d),
                    differ=round(100 * np.mean([r[key] for r in d]), 1), n_twin_q=len(both), both=round(100 * np.mean(both), 1))

    P = pairs
    out = dict(split=SPLIT, n_pairs=len(P),
               contract=dict(last_frame_equal=sum(p['last_frame_equal'] for p in P), same_last_seen_frame=sum(p['same_k'] for p in P),
                             max_last_seen_dist_m=max(p['last_seen_dist'] for p in P),
                             exited_hidden_at_end_both=sum(all(p['hidden_end_a']) for p in P)),
               observability=dict(
                   a_seen_ge2_both=sum(min(p['nvis_a']) >= 2 for p in P),
                   a_seen_once_in_some_twin=sum(min(p['nvis_a']) < 2 for p in P),
                   a_seen_once_twin0=sum(p['nvis_a'][0] < 2 for p in P), a_seen_once_twin1=sum(p['nvis_a'][1] < 2 for p in P),
                   b_seen_once_in_some_twin=sum(min(p['nvis_b']) < 2 for p in P),
                   observable_pairs=sum(p['observable'] for p in P),
                   vel_err_a_observable_max=max(max(p['vel_err_a']) for p in P if p['observable'])),
               omni_label_reproduction_max_err_m=max(r['omni_r_err'] for r in rows),
               anchor_rule={'all_pairs': summ(rows, 'static_ok'), 'observable': summ([r for r in rows if r['observable']], 'static_ok')},
               B4_observed={'all_pairs': summ(rows, 'obs_ok'), 'observable': summ([r for r in rows if r['observable']], 'obs_ok')},
               B4_omniscient={'all_pairs': summ(rows, 'omni_ok'), 'observable': summ([r for r in rows if r['observable']], 'omni_ok')})
    print(json.dumps(out, indent=1))
    json.dump(dict(summary=out, pairs=P), open(os.path.join(sts.HERE, f'ladder_stage_a_{SPLIT}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
