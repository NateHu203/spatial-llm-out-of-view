"""Stage D pilot: a learned per-object state estimator over the VLM's own (cached, frozen) visual tokens.
For each frame t and each queried ball (identified by colour) it predicts [visible?, x_t, z_t] in frame t's OWN
camera axes. Known camera motion (cam_speed, yaw_rate: an odometry assumption) maps the positions into the t=7 axes;
the existing observed-only estimator (ladder_stage_a.est_velocity) and propagation then give the state. The output is
written as tracks files ({split}_est_tracks.json) in the same format as the oracle tracks, so the C1 interface
(sts.py --text_state --est_tracks ...) and the B4 executor can consume them unchanged.
Usage: python state_estimator.py [--train train3:3000,train8:1200] [--steps 3000]"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import sts, scenes
from ladder_stage_a import predict

COLORS = sts.COLORS
XS, ZS = 3.0, 6.0   # position scale (as in slot_feats)


def cam_axes(yaw):
    return np.array([math.cos(yaw), -math.sin(yaw)]), np.array([math.sin(yaw), math.cos(yaw)])   # right, forward


def t7_to_frame(tr):
    """(8, 2, 2) positions of the two balls in each frame's own camera axes, from the t=7-axes tracks."""
    scn = dict(yaw0=0.0, yaw_rate=tr['yaw_rate'], cam_speed=tr['cam_speed'])
    c7, y7 = scenes.cam_pose(scn, 7.0); r7, f7 = cam_axes(y7)
    pos = np.array(tr['pos'], float)
    world = c7 + pos[..., :1] * r7 + pos[..., 1:] * f7
    out = np.zeros_like(pos)
    for t in range(8):
        c, y = scenes.cam_pose(scn, float(t))
        out[t] = scenes.to_cam(world[t], c, y)
    return out


def frame_to_t7(own, tr):
    """inverse of t7_to_frame: own (8, 2, 2) -> t=7 axes."""
    scn = dict(yaw0=0.0, yaw_rate=tr['yaw_rate'], cam_speed=tr['cam_speed'])
    c7, y7 = scenes.cam_pose(scn, 7.0)
    out = np.zeros_like(own)
    for t in range(8):
        c, y = scenes.cam_pose(scn, float(t)); r, f = cam_axes(y)
        world = c + own[t, :, :1] * r + own[t, :, 1:] * f
        out[t] = scenes.to_cam(world, c7, y7)
    return out


class StateEstimator(nn.Module):
    def __init__(self, d_in=2048, d=256, layers=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(64, d) * 0.02)                 # 8x8 merged-token grid
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), layers)
        self.col = nn.Embedding(len(COLORS), d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, tok, col):
        """tok: (B, 64, 2048) one frame's visual tokens; col: (B,) colour id -> (B, 3) [vis logit, x/3, z/6]."""
        h = self.enc(self.proj(tok.float()) + self.pos)
        q = self.col(col)[:, None]
        o, _ = self.attn(q, h, h)
        return self.head(o[:, 0])


def load_split(split, n):
    """feats (n, 8, 64, 2048) bf16 on GPU, colour ids (n, 2), vis (n, 8, 2), own-frame positions (n, 8, 2, 2), tracks."""
    F_, C, V, P, TR = [], [], [], [], []
    for i in range(n):
        sc = sts.load(split, i)
        tr = sts.scene_tracks(split, i, sc)
        F_.append(torch.load(os.path.join(sts.DATA, 'vcache', split, f'{i:05d}.pt')).view(8, 64, -1))
        C.append([COLORS.index(c) for c in sc['names']])
        V.append(np.array(tr['vis'], bool)); P.append(t7_to_frame(tr)); TR.append(tr)
    return (torch.stack(F_).cuda(), torch.tensor(C).cuda(), torch.tensor(np.array(V)).cuda(),
            torch.tensor(np.array(P), dtype=torch.float32).cuda(), TR)


def flat(feats, cols, vis, pos):
    """(scene, frame, ball) samples -> index tensors"""
    n = feats.shape[0]
    s, t, b = torch.meshgrid(torch.arange(n), torch.arange(8), torch.arange(2), indexing='ij')
    return s.reshape(-1).cuda(), t.reshape(-1).cuda(), b.reshape(-1).cuda()


def run_model(model, feats, cols, s, t, b, bs=1024):
    out = []
    with torch.no_grad():
        for k in range(0, len(s), bs):
            out.append(model(feats[s[k:k + bs], t[k:k + bs]], cols[s[k:k + bs], b[k:k + bs]]))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='train3:3000,train8:1200'); ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    t0 = time.time()
    TRN = {sp: load_split(sp, int(n)) for sp, n in (x.split(':') for x in a.train.split(','))}
    print('loaded train', {k: v[0].shape[0] for k, v in TRN.items()}, f'{time.time() - t0:.0f}s', flush=True)
    samples = []
    for sp, (feats, cols, vis, pos, _) in TRN.items():
        s, t, b = flat(feats, cols, vis, pos)
        samples += [(sp, s, t, b)]
    model = StateEstimator().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    for step in range(a.steps):
        sp, s, t, b = samples[step % len(samples)]
        feats, cols, vis, pos, _ = TRN[sp]
        idx = torch.tensor(rng.choice(len(s), a.bs, replace=False)).cuda()
        si, ti, bi = s[idx], t[idx], b[idx]
        out = model(feats[si, ti], cols[si, bi])
        v = vis[si, ti, bi].float(); p = pos[si, ti, bi] / torch.tensor([XS, ZS], device='cuda')
        lv = F.binary_cross_entropy_with_logits(out[:, 0], v)
        lp = (((out[:, 1:] - p) ** 2).sum(-1) * v).sum() / v.sum().clamp(min=1)
        loss = lv + lp
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 250 == 0:
            print(f'step {step + 1} vis_bce {lv.item():.3f} pos_mse {lp.item():.4f} {time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), os.path.join(sts.HERE, 'state_estimator.pt'))
    del TRN
    model.eval()
    report = {}
    for sp, n in (('eval8', 600), ('eval3', 300)):
        feats, cols, vis, pos, TR = load_split(sp, n)
        s, t, b = flat(feats, cols, vis, pos)
        out = run_model(model, feats, cols, s, t, b).view(n, 8, 2, 3).cpu().numpy()
        vh = out[..., 0] > 0; ph = out[..., 1:] * np.array([XS, ZS])
        vt = vis.cpu().numpy(); pt = pos.cpu().numpy()
        err = np.linalg.norm(ph - pt, axis=-1)[vt]
        est = []
        for i in range(n):
            tr = TR[i]
            p7 = frame_to_t7(ph[i], tr)
            est.append(dict(vis=vh[i].tolist(), pos=np.round(p7, 2).tolist(), cam_speed=tr['cam_speed'], yaw_rate=tr['yaw_rate']))
        json.dump(est, open(os.path.join(sts.HERE, f'{sp}_est_tracks.json'), 'w'))
        # B4 executor on estimated vs oracle tracks
        rows = []
        for i in range(n):
            sc = sts.load(sp, i)
            for q in sc['questions']:
                rows.append(dict(pair=sc.get('pair', i), twin=sc.get('twin', 0), t=q['t'], qtype=q['qtype'], differs=q.get('differs', False),
                                 hidden=any(sc['hidden_end']), static_ok=q['static_ok'],
                                 est=int(predict(est[i], q, True)[0] == q['label']), orc=int(predict(TR[i], q, True)[0] == q['label'])))
        r = dict(vis_acc=round(100 * float((vh == vt).mean()), 1), vis_recall=round(100 * float(vh[vt].mean()), 1),
                 vis_precision=round(100 * float(vt[vh].mean()), 1), pos_err_med_m=round(float(np.median(err)), 3),
                 pos_err_p90_m=round(float(np.percentile(err, 90)), 3))
        for key in ('est', 'orc'):
            r[f'B4_{key}_all'] = round(100 * np.mean([x[key] for x in rows]), 1)
            if sp == 'eval8':
                d = [x for x in rows if x['differs']]
                by = {}
                for x in d:
                    by.setdefault((x['pair'], x['t'], x['qtype']), []).append(x[key])
                r[f'B4_{key}_differ'] = round(100 * np.mean([x[key] for x in d]), 1)
                r[f'B4_{key}_both'] = round(100 * np.mean([all(v) for v in by.values() if len(v) == 2]), 1)
            else:
                h = [x for x in rows if x['hidden'] and not x['static_ok']]
                r[f'B4_{key}_hiddenSW'] = round(100 * np.mean([x[key] for x in h]), 1)
        report[sp] = r
        print(sp, json.dumps(r), flush=True)
    json.dump(report, open(os.path.join(sts.HERE, 'state_estimator_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
