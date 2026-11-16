"""Fill the ViT feature cache (sts_data/vcache) once, before several runs share it in parallel.
Usage: python precache.py eval3:300 eval6:600 train6:2400 train6d:2400 train3:4000   (split:number of scenes)"""
import os, sys, time
import torch
import sts

proc, model, _, _ = sts.build(0, None)
model.eval()
t0 = time.time()
for arg in sys.argv[1:]:
    split, n = arg.split(':')
    for i in range(int(n)):
        if not os.path.exists(os.path.join(sts.DATA, 'vcache', split, f'{i:05d}.pt')):
            sts.vis_feats(model, proc, split, i, sts.load(split, i)['frames'])
    print(split, n, f'{time.time() - t0:.0f}s', flush=True)
