"""Run the trained Stage-D state estimator (state_estimator.pt) on cached splits and write {split}_est_tracks.json.
Usage: python est_infer.py eval9:600 train9:600"""
import json, os, sys
import numpy as np, torch
import sts
from state_estimator import StateEstimator, load_split, flat, run_model, frame_to_t7, XS, ZS

model = StateEstimator().cuda()
model.load_state_dict(torch.load(os.path.join(sts.HERE, 'state_estimator.pt'))); model.eval()
for arg in sys.argv[1:]:
    sp, n = arg.split(':'); n = int(n)
    feats, cols, vis, pos, TR = load_split(sp, n)
    s, t, b = flat(feats, cols, vis, pos)
    out = run_model(model, feats, cols, s, t, b).view(n, 8, 2, 3).cpu().numpy()
    vh = out[..., 0] > 0; ph = out[..., 1:] * np.array([XS, ZS])
    vt = vis.cpu().numpy(); err = np.linalg.norm(ph - pos.cpu().numpy(), axis=-1)[vt]
    est = [dict(vis=vh[i].tolist(), pos=np.round(frame_to_t7(ph[i], TR[i]), 2).tolist(), cam_speed=TR[i]['cam_speed'],
                yaw_rate=TR[i]['yaw_rate']) for i in range(n)]
    json.dump(est, open(os.path.join(sts.HERE, f'{sp}_est_tracks.json'), 'w'))
    print(sp, 'vis_acc', round(100 * float((vh == vt).mean()), 1), 'pos_err_med', round(float(np.median(err)), 3),
          'p90', round(float(np.percentile(err, 90)), 3), flush=True)
