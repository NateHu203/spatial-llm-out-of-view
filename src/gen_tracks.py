"""Replay sample_scene_multi for train3/eval3 and save per-frame oracle tracks of the queried pair:
positions at frames 0..7 in the LAST-frame camera axes (lateral x, forward z), per-frame visibility, camera kinematics.
Output: sts_data/<split>_tracks.json  (list indexed by scene id). Used by sts.py --tracks oracle."""
import json, os, sys
from multiprocessing import Pool
import numpy as np

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
BASE = {'train3': 40_000_000, 'eval3': 50_000_000}


def one(args):
    split, i = args
    import scenes
    calls = []
    _render = scenes.render
    def rec(scn, t):
        im, g = _render(scn, t); calls.append((scn, g['visible'])); return im, g
    scenes.render = rec
    sc = scenes.sample_scene_multi(BASE[split] + i)
    scn = calls[-1][0]
    vis = np.array([c[1] for c in calls[-8:]])
    names = list(scn['names']); ab = [names.index(n) for n in sc['names']]
    cxz7, yaw7 = scenes.cam_pose(scn, 7.0)
    pos = np.stack([scenes.to_cam(scenes.obj_pos(scn, float(t))[ab], cxz7, yaw7) for t in range(8)])   # 8,2,2
    return dict(vis=vis[:, ab].tolist(), pos=np.round(pos, 2).tolist(),
                cam_speed=round(float(scn['cam_speed']), 2), yaw_rate=round(float(scn['yaw_rate']), 3))


if __name__ == '__main__':
    for split, n in (('eval3', 300), ('train3', int(sys.argv[1]) if len(sys.argv) > 1 else 5000)):
        with Pool(6) as p:
            out = p.map(one, [(split, i) for i in range(n)], chunksize=16)
        json.dump(out, open(os.path.join(HERE, 'sts_data', f'{split}_tracks.json'), 'w'))
        print(split, len(out), flush=True)
