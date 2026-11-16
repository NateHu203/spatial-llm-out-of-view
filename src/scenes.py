"""Procedural dynamic tabletop-floor scenes for the NowRoPE real-VLM MVE (v0.6).

A pinhole camera at height H moves forward and yaws at a constant rate over a checkered floor while
coloured spheres roll with constant velocity. Frames are 1 s apart. Every merged Qwen2.5-VL token
(28x28 px block) gets oracle geometry: object id (or floor / sky), world ground-plane position, and
world velocity. The question asks for a relation at query time t* (last frame, or 2 s later assuming
the same motion continues), in the camera frame at t*. Items are sampled so that most of them involve
an object that is NOT visible at the last frame, which forces a model of motion.
"""
import math
import numpy as np

IMG = 224
TOK = 28                      # merged token = 2x2 patches of 14 px
G = IMG // TOK                # 8x8 tokens per frame
T_FR = 8
H_CAM, PITCH, FOV = 1.4, math.radians(20), math.radians(70)
FOCAL = (IMG / 2) / math.tan(FOV / 2)
R_OBJ = 0.3
COLORS = {'red': (220, 40, 40), 'blue': (40, 80, 230), 'green': (40, 180, 60), 'yellow': (235, 210, 40)}
NAMES = list(COLORS)


def cam_pose(scn, t):
    """camera position (x,z) on the plane and yaw at time t (constant forward speed + yaw rate)."""
    yaw = scn['yaw0'] + scn['yaw_rate'] * t
    # integrate forward motion along changing yaw (closed form for constant yaw rate)
    w, s = scn['yaw_rate'], scn['cam_speed']
    if abs(w) < 1e-6:
        dx, dz = s * t * math.sin(scn['yaw0']), s * t * math.cos(scn['yaw0'])
    else:
        dx = s / w * (math.cos(scn['yaw0']) - math.cos(yaw))
        dz = s / w * (math.sin(yaw) - math.sin(scn['yaw0']))
    return np.array([dx, dz]), yaw


def to_cam(pxz, cxz, yaw):
    """world ground-plane point(s) -> camera-at-yaw lateral x (right +) and forward z."""
    d = pxz - cxz
    fx, fz = math.sin(yaw), math.cos(yaw)          # forward
    rx, rz = math.cos(yaw), -math.sin(yaw)         # right
    return np.stack([d[..., 0] * rx + d[..., 1] * rz, d[..., 0] * fx + d[..., 1] * fz], -1)


def obj_pos(scn, t):
    return scn['p0'] + scn['v'] * t


def render(scn, t):
    cxz, yaw = cam_pose(scn, t)
    C = np.array([cxz[0], H_CAM, cxz[1]])
    fwd = np.array([math.sin(yaw) * math.cos(PITCH), -math.sin(PITCH), math.cos(yaw) * math.cos(PITCH)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    up = np.cross(fwd, right)
    u = (np.arange(IMG) + 0.5 - IMG / 2) / FOCAL
    uu, vv = np.meshgrid(u, u)
    D = fwd[None, None] + uu[..., None] * right[None, None] - vv[..., None] * up[None, None]
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    depth = np.full((IMG, IMG), np.inf)
    ident = np.full((IMG, IMG), -2, np.int16)          # -2 sky, -1 floor, k object
    tfl = np.where(D[..., 1] < -1e-6, -C[1] / np.minimum(D[..., 1], -1e-6), np.inf)
    floor = np.isfinite(tfl)
    depth[floor] = tfl[floor]; ident[floor] = -1
    P = obj_pos(scn, t)
    for k in range(len(P)):
        c = np.array([P[k, 0], R_OBJ, P[k, 1]])
        oc = C - c
        b = (D * oc).sum(-1); cc = (oc * oc).sum() - R_OBJ ** 2
        disc = b * b - cc
        hit = disc > 0
        tk = np.where(hit, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        better = hit & (tk > 0) & (tk < depth)
        depth[better] = tk[better]; ident[better] = k
    img = np.zeros((IMG, IMG, 3), np.float32)
    img[...] = (150, 200, 245)                          # sky
    hitp = C[None, None] + D * np.where(np.isfinite(depth), depth, 0)[..., None]
    chk = ((np.floor(hitp[..., 0]) + np.floor(hitp[..., 2])) % 2 == 0)
    fl = ident == -1
    img[fl] = np.where(chk[fl][:, None], (205, 205, 205), (120, 120, 120))
    light = np.array([0.3, 0.9, -0.3]); light /= np.linalg.norm(light)
    for k in range(len(P)):
        m = ident == k
        if not m.any():
            continue
        c = np.array([P[k, 0], R_OBJ, P[k, 1]])
        n = (hitp[m] - c) / R_OBJ
        shade = 0.35 + 0.65 * np.clip((n * light).sum(-1), 0, 1)
        img[m] = np.array(COLORS[scn['names'][k]])[None] * shade[:, None]
    # per-token oracle geometry (block centre pixel)
    cidx = np.arange(G) * TOK + TOK // 2
    tid = ident[np.ix_(cidx, cidx)]                     # G,G
    tp = hitp[np.ix_(cidx, cidx)][..., [0, 2]]          # world x,z of floor hit
    tok_world = np.zeros((G, G, 2)); tok_vel = np.zeros((G, G, 2)); tok_kind = np.zeros((G, G), np.int8)
    for i in range(G):
        for j in range(G):
            k = tid[i, j]
            if k >= 0:
                tok_world[i, j] = P[k]; tok_vel[i, j] = scn['v'][k]; tok_kind[i, j] = 2
            elif k == -1:
                tok_world[i, j] = tp[i, j]; tok_kind[i, j] = 1
    npix = [int((ident == k).sum()) for k in range(len(P))]
    visible = [n > 30 for n in npix]
    return img.clip(0, 255).astype(np.uint8), dict(world=tok_world, vel=tok_vel, kind=tok_kind, visible=visible, npix=npix)


def sample_item(seed):
    """Returns frames (list of uint8 HxWx3), per-frame token geometry, question, answer options, label, meta."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        scn = dict(yaw0=0.0, yaw_rate=rng.uniform(-0.12, 0.12), cam_speed=rng.uniform(0.15, 0.45),
                   names=list(rng.permutation(NAMES)[:3]))
        n = 3
        # objects start in front of the camera, moving with random velocities
        p0 = np.stack([rng.uniform(-1.6, 1.6, n), rng.uniform(2.0, 5.5, n)], -1)
        ang = rng.uniform(0, 2 * math.pi, n); spd = rng.uniform(0.15, 0.7, n)
        v = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], -1)
        scn.update(p0=p0, v=v)
        if min(np.linalg.norm(p0[a] - p0[b]) for a in range(n) for b in range(a + 1, n)) < 0.8:
            continue
        frames, geos = [], []
        for t in range(T_FR):
            im, g = render(scn, float(t)); frames.append(im); geos.append(g)
        vis = np.array([g['visible'] for g in geos])     # T,n
        seen = vis.any(0)
        future = rng.random() < 0.3
        tstar = T_FR - 1 + (2.0 if future else 0.0)
        cxz, yaw = cam_pose(scn, tstar)
        P = obj_pos(scn, tstar)
        cam = to_cam(P, cxz, yaw)                           # n,2 (lateral, forward)
        pairs = [(a, b) for a in range(n) for b in range(n) if a < b and seen[a] and seen[b]]
        want_hidden = rng.random() < 0.75
        cand = [(a, b) for a, b in pairs if (not vis[-1, a] or not vis[-1, b]) == want_hidden]
        if not cand:
            continue
        a, b = cand[rng.integers(len(cand))]
        if rng.random() < 0.5:
            a, b = b, a
        qtype = 'leftright' if rng.random() < 0.5 else 'closer'
        if qtype == 'leftright':
            if abs(cam[a, 0] - cam[b, 0]) < 0.4:
                continue
            label = 0 if cam[a, 0] < cam[b, 0] else 1
            opts = ['left', 'right']
            when = 'at the moment of the last frame' if not future else 'two seconds after the last frame (assume everything keeps moving exactly as before, including you)'
            q = (f"These 8 frames are from a video taken by a moving camera, one frame per second. "
                 f"Objects may move and may leave the view. Think about where things are {when}, "
                 f"seen from the camera's own position and heading at that time. "
                 f"Is the {scn['names'][a]} ball to the left or to the right of the {scn['names'][b]} ball? "
                 f"Answer with one word: left or right.")
        else:
            da, db = np.linalg.norm(cam[a]), np.linalg.norm(cam[b])
            if abs(da - db) < 0.4:
                continue
            label = 0 if da < db else 1
            opts = [scn['names'][a], scn['names'][b]]
            when = 'at the moment of the last frame' if not future else 'two seconds after the last frame (assume everything keeps moving exactly as before, including you)'
            q = (f"These 8 frames are from a video taken by a moving camera, one frame per second. "
                 f"Objects may move and may leave the view. Think about where things are {when}. "
                 f"Which ball is closer to the camera then: the {scn['names'][a]} ball or the {scn['names'][b]} ball? "
                 f"Answer with one word: {scn['names'][a]} or {scn['names'][b]}.")
        last_seen = [int(np.nonzero(vis[:, k])[0].max()) for k in (a, b)]
        stale = max(float(np.linalg.norm(scn['v'][k]) * (tstar - ls)) for k, ls in zip((a, b), last_seen))
        cams = [cam_pose(scn, float(t)) for t in range(T_FR)]
        return dict(frames=frames, geos=geos, question=q, options=opts, label=label, tstar=tstar, future=future,
                    qtype=qtype, hidden=want_hidden, stale=stale, cam_now=(cxz, yaw), cams=cams,
                    scn=scn, pair=(a, b), vis=vis)
    raise RuntimeError('could not sample item')


def static_heuristic_ok(it):
    """True if 'everything stays where last seen, camera frozen at last frame' answers the item correctly."""
    scn, (a, b), vis = it['scn'], it['pair'], it['vis']
    cam7 = cam_pose(scn, float(T_FR - 1))
    pos = {}
    for k in (a, b):
        last = int(np.nonzero(vis[:, k])[0].max())
        pos[k] = to_cam(obj_pos(scn, float(last))[k], *cam7)
    if it['qtype'] == 'leftright':
        h = 0 if pos[a][0] < pos[b][0] else 1
    else:
        h = 0 if np.linalg.norm(pos[a]) < np.linalg.norm(pos[b]) else 1
    return h == it['label']


def sample_item_v2(i, split):
    """MVE v2 sampler. eval2: every item defeats the static heuristic. train2: even i defeats it, odd i unconstrained."""
    base = {'eval2': 20_000_000, 'train2': 30_000_000}[split]
    need_motion = split == 'eval2' or i % 2 == 0
    for j in range(5000):
        it = sample_item(base + i * 5000 + j)
        if not need_motion or not static_heuristic_ok(it):
            it['motion_required'] = not static_heuristic_ok(it)
            return it
    raise RuntimeError('no motion-required item found')


# ---------------------------------------------------------------- STS (shared-trajectory supervision) data
WHEN = {7.0: 'at the moment of the last frame', 8.0: 'one second after the last frame',
        9.0: 'two seconds after the last frame', 10.0: 'three seconds after the last frame'}


def question_text(qtype, na, nb, t):
    when = WHEN[t] + ('' if t == 7.0 else ' (assume everything keeps moving exactly as before, including you)')
    head = ("These 8 frames are from a video taken by a moving camera, one frame per second. "
            "Objects may move and may leave the view. ")
    if qtype == 'leftright':
        return head + (f"Think about where things are {when}, seen from the camera's own position and heading at that time. "
                       f"Is the {na} ball to the left or to the right of the {nb} ball? Answer with one word: left or right.")
    return head + (f"Think about where things are {when}. Which ball is closer to the camera then: the {na} ball or the {nb} ball? "
                   f"Answer with one word: {na} or {nb}.")


def sample_scene_multi(seed, times=(7.0, 8.0, 9.0, 10.0)):
    """One scene + one object pair (a,b), with left/right and closer questions at every query time.
    Oracle state is expressed in the LAST-FRAME camera frame (lateral x, forward z): object positions at t=7 and
    world velocities rotated into that frame; camera forward speed and yaw rate."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        scn = dict(yaw0=0.0, yaw_rate=rng.uniform(-0.12, 0.12), cam_speed=rng.uniform(0.15, 0.45),
                   names=list(rng.permutation(NAMES)[:3]))
        n = 3
        p0 = np.stack([rng.uniform(-1.6, 1.6, n), rng.uniform(2.0, 5.5, n)], -1)
        ang = rng.uniform(0, 2 * math.pi, n); spd = rng.uniform(0.15, 0.7, n)
        scn.update(p0=p0, v=np.stack([np.cos(ang) * spd, np.sin(ang) * spd], -1))
        if min(np.linalg.norm(p0[i] - p0[j]) for i in range(n) for j in range(i + 1, n)) < 0.8:
            continue
        frames, vis = [], []
        for t in range(T_FR):
            im, g = render(scn, float(t)); frames.append(im); vis.append(g['visible'])
        vis = np.array(vis)
        seen = vis.any(0)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j and seen[i] and seen[j]]
        if not pairs:
            continue
        a, b = pairs[rng.integers(len(pairs))]
        cxz7, yaw7 = cam_pose(scn, 7.0)
        qs = []
        for t in times:
            cxz, yaw = cam_pose(scn, t)
            cam = to_cam(obj_pos(scn, t), cxz, yaw)
            lr = 0 if cam[a, 0] < cam[b, 0] else 1
            nr = 0 if np.linalg.norm(cam[a]) < np.linalg.norm(cam[b]) else 1
            qs.append(dict(t=t, qtype='leftright', question=question_text('leftright', scn['names'][a], scn['names'][b], t),
                           options=['left', 'right'], label=lr, margin=float(cam[b, 0] - cam[a, 0]), r=cam[[a, b]].tolist()))
            qs.append(dict(t=t, qtype='closer', question=question_text('closer', scn['names'][a], scn['names'][b], t),
                           options=[scn['names'][a], scn['names'][b]], label=nr,
                           margin=float(np.sum(cam[b] ** 2) - np.sum(cam[a] ** 2)), r=cam[[a, b]].tolist()))
        # oracle state in last-frame camera frame
        R = np.array([[math.cos(yaw7), -math.sin(yaw7)], [math.sin(yaw7), math.cos(yaw7)]])  # world (x,z) -> cam (lat,fwd)
        def rot(w):  # world displacement -> last-frame camera axes
            return np.array([w[0] * math.cos(yaw7) - w[1] * math.sin(yaw7), w[0] * math.sin(yaw7) + w[1] * math.cos(yaw7)])
        p7 = to_cam(obj_pos(scn, 7.0)[[a, b]], cxz7, yaw7)
        v7 = np.stack([rot(scn['v'][k]) for k in (a, b)])
        state = dict(p=p7.tolist(), v=v7.tolist(), cam_speed=scn['cam_speed'], yaw_rate=scn['yaw_rate'])
        last = [int(np.nonzero(vis[:, k])[0].max()) for k in (a, b)]
        # static 'last-seen' heuristic for each question (camera frozen at last frame)
        for q in qs:
            pos = [to_cam(obj_pos(scn, float(last[i]))[k], cxz7, yaw7) for i, k in enumerate((a, b))]
            if q['qtype'] == 'leftright':
                h = 0 if pos[0][0] < pos[1][0] else 1
            else:
                h = 0 if np.linalg.norm(pos[0]) < np.linalg.norm(pos[1]) else 1
            q['static_ok'] = bool(h == q['label'])
        return dict(frames=frames, questions=qs, state=state, names=[scn['names'][a], scn['names'][b]],
                    hidden_end=[bool(not vis[-1, a]), bool(not vis[-1, b])])
    raise RuntimeError('no scene')


# ---------------------------------------------------------------- repaired task (v0.6, 2026-10-23)
def question_text_fixed(qtype, na, nb, t):
    when = WHEN[t] + ('' if t == 7.0 else ' (every ball keeps rolling exactly as before)')
    head = ("These 8 frames are from a video taken by a fixed camera that never moves, one frame per second. "
            "The balls roll in straight lines at constant speed and may leave the view. ")
    if qtype == 'leftright':
        return head + (f"Think about where the balls are {when}. As seen from the camera, is the {na} ball to the left or "
                       f"to the right of the {nb} ball? Answer with one word: left or right.")
    return head + (f"Think about where the balls are {when}. Which ball is closer to the camera then: the {na} ball or "
                   f"the {nb} ball? Answer with one word: {na} or {nb}.")


def sample_twin_pair(seed, times=(7.0, 8.0, 9.0, 10.0), min_vis=4, moving=False):
    """Twin scenes with a FIXED camera and IDENTICAL last frames (same positions at t=7) but opposite object
    velocities, so their histories differ. Future questions are kept only where the twins' labels differ: the
    last frame (static rule) is then right on exactly one twin, so any accuracy above 50% must come from motion.
    t=7 questions are controls (same label in both twins). Returns [twin0, twin1] with frames, questions, tracks."""
    rng = np.random.default_rng(seed)
    for _ in range(500):
        names = list(rng.permutation(NAMES)[:3]); n = 3
        cam = dict(yaw0=0.0, yaw_rate=0.0, cam_speed=0.0)
        if moving:   # same camera trajectory in both twins (moves forward and turns), so the last frames still match
            cam = dict(yaw0=0.0, yaw_rate=float(rng.uniform(-0.12, 0.12)), cam_speed=float(rng.uniform(0.15, 0.45)))
        cxz7, yaw7 = cam_pose(cam, 7.0)
        p7 = np.stack([rng.uniform(-1.4, 1.4, n), rng.uniform(2.6, 6.0, n)], -1)      # in the t=7 camera frame
        R = np.array([[math.cos(yaw7), -math.sin(yaw7)], [math.sin(yaw7), math.cos(yaw7)]])
        p7 = cxz7 + p7 @ R                                                            # -> world
        if min(np.linalg.norm(p7[i] - p7[j]) for i in range(n) for j in range(i + 1, n)) < 0.8:
            continue
        ang = rng.uniform(0, 2 * math.pi, n); spd = rng.uniform(0.2, 0.6, n)
        v = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], -1)
        twins = []
        for sgn in (1.0, -1.0):
            scn = dict(cam, names=names, v=sgn * v, p0=p7 - sgn * v * 7.0)
            frames, vis = [], []
            for t in range(T_FR):
                im, g = render(scn, float(t)); frames.append(im); vis.append(g['visible'])
            twins.append(dict(scn=scn, frames=frames, vis=np.array(vis)))
        ok = [k for k in range(n) if all(tw['vis'][:, k].sum() >= min_vis and tw['vis'][-1, k] for tw in twins)]
        pairs = [(a, b) for a in ok for b in ok if a != b]
        if not pairs:
            continue
        a, b = pairs[rng.integers(len(pairs))]

        def labels(scn, t):
            cxz, yaw = cam_pose(scn, t)
            c = to_cam(obj_pos(scn, t), cxz, yaw)   # camera frame at the query time
            return (0 if c[a, 0] < c[b, 0] else 1), (0 if np.linalg.norm(c[a]) < np.linalg.norm(c[b]) else 1), c[[a, b]]
        keep = []
        for t in times:
            L = [labels(tw['scn'], t) for tw in twins]
            for qi, qtype in enumerate(('leftright', 'closer')):
                if t == 7.0 or L[0][qi] != L[1][qi]:
                    keep.append((t, qi, qtype))
        if sum(1 for t, _, _ in keep if t > 7) < 2:
            continue
        out = []
        for w, tw in enumerate(twins):
            qs = []
            for t, qi, qtype in keep:
                lab = labels(tw['scn'], t)
                opts = ['left', 'right'] if qtype == 'leftright' else [names[a], names[b]]
                qtxt = (question_text if moving else question_text_fixed)(qtype, names[a], names[b], t)
                qs.append(dict(t=t, qtype=qtype, question=qtxt, options=opts,
                               label=lab[qi], r=lab[2].tolist(), static_ok=(t == 7.0) or bool(lab[qi] == labels(tw['scn'], 7.0)[qi])))
            pos = np.stack([to_cam(obj_pos(tw['scn'], float(t))[[a, b]], cxz7, yaw7) for t in range(T_FR)])   # last-frame camera axes
            out.append(dict(frames=tw['frames'], questions=qs, names=[names[a], names[b]], twin=w,
                            tracks=dict(vis=tw['vis'][:, [a, b]].tolist(), pos=np.round(pos, 2).tolist(),
                                        cam_speed=round(cam['cam_speed'], 2), yaw_rate=round(cam['yaw_rate'], 3))))
        assert np.array_equal(out[0]['frames'][-1], out[1]['frames'][-1])
        return out
    raise RuntimeError('no twin pair')


# ---------------------------------------------------------------- natural-distribution history twins (v0.6, 2026-10-23)
def _multi_questions(scn, a, b, vis, times):
    """Same question/label/static_ok construction as sample_scene_multi, for an already-rendered scene."""
    cxz7, yaw7 = cam_pose(scn, 7.0)
    qs = []
    for t in times:
        cxz, yaw = cam_pose(scn, t)
        cam = to_cam(obj_pos(scn, t), cxz, yaw)
        lr = 0 if cam[a, 0] < cam[b, 0] else 1
        nr = 0 if np.linalg.norm(cam[a]) < np.linalg.norm(cam[b]) else 1
        qs.append(dict(t=t, qtype='leftright', question=question_text('leftright', scn['names'][a], scn['names'][b], t),
                       options=['left', 'right'], label=lr, r=cam[[a, b]].tolist()))
        qs.append(dict(t=t, qtype='closer', question=question_text('closer', scn['names'][a], scn['names'][b], t),
                       options=[scn['names'][a], scn['names'][b]], label=nr, r=cam[[a, b]].tolist()))
    last = [int(np.nonzero(vis[:, k])[0].max()) for k in (a, b)]
    for q in qs:
        pos = [to_cam(obj_pos(scn, float(last[i]))[k], cxz7, yaw7) for i, k in enumerate((a, b))]
        h = (0 if pos[0][0] < pos[1][0] else 1) if q['qtype'] == 'leftright' else \
            (0 if np.linalg.norm(pos[0]) < np.linalg.norm(pos[1]) else 1)
        q['static_ok'] = bool(h == q['label'])
    pos = np.stack([to_cam(obj_pos(scn, float(t))[[a, b]], cxz7, yaw7) for t in range(T_FR)])
    tracks = dict(vis=vis[:, [a, b]].tolist(), pos=np.round(pos, 2).tolist(),
                  cam_speed=round(float(scn['cam_speed']), 2), yaw_rate=round(float(scn['yaw_rate']), 3))
    return qs, tracks


def sample_natural_twin(seed, times=(7.0, 8.0, 9.0, 10.0), tries=200_000, history='posterior'):
    """Twin 0 is an ordinary sample_scene_multi-prior scene. Twin 1 keeps the camera and every object's position at
    t=7 (so the last frames are pixel-identical) and redraws the history from the SAME prior conditioned on those
    positions: velocities from the prior, p0 = p7 - 7 v, accepted only if p0 lies in the prior's start box and keeps
    the prior's 0.8 m start separation (rejection sampling = exact posterior over histories given the present).
    Both twins are therefore marginally ordinary scenes; nothing about either one marks it as a twin. All 8
    questions are kept in both; labels may or may not differ between the twins.
    history='reflect' (salient but disguised twins): twin 1 instead gets every velocity reversed (v -> -v,
    p0 = p7 + 7 v), a large history contrast with no question/visibility selection; twin 0 is still an ordinary scene."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        scn = dict(yaw0=0.0, yaw_rate=rng.uniform(-0.12, 0.12), cam_speed=rng.uniform(0.15, 0.45),
                   names=list(rng.permutation(NAMES)[:3]))
        n = 3
        p0 = np.stack([rng.uniform(-1.6, 1.6, n), rng.uniform(2.0, 5.5, n)], -1)
        ang = rng.uniform(0, 2 * math.pi, n); spd = rng.uniform(0.15, 0.7, n)
        scn.update(p0=p0, v=np.stack([np.cos(ang) * spd, np.sin(ang) * spd], -1))
        if min(np.linalg.norm(p0[i] - p0[j]) for i in range(n) for j in range(i + 1, n)) < 0.8:
            continue
        p7 = obj_pos(scn, 7.0)
        if history == 'reflect':
            tries = 1
        # posterior history: vectorised rejection over `tries` joint draws
        ang2 = rng.uniform(0, 2 * math.pi, (tries, n)); spd2 = rng.uniform(0.15, 0.7, (tries, n))
        v2 = np.stack([np.cos(ang2) * spd2, np.sin(ang2) * spd2], -1)                  # tries,n,2
        q0 = p7[None] - 7.0 * v2
        inbox = ((np.abs(q0[..., 0]) <= 1.6) & (q0[..., 1] >= 2.0) & (q0[..., 1] <= 5.5)).all(1)
        sep = np.min(np.stack([np.linalg.norm(q0[:, i] - q0[:, j], axis=-1) for i in range(n) for j in range(i + 1, n)], 1), 1)
        acc = np.nonzero(inbox & (sep >= 0.8))[0]
        if history == 'reflect':
            twins = [scn, dict(scn, p0=p7 + 7.0 * scn['v'], v=-scn['v'])]
        elif len(acc) == 0:
            continue
        else:
            k = acc[0]
            twins = [scn, dict(scn, p0=q0[k], v=v2[k])]
        rend = []
        for s in twins:
            frames, vis = [], []
            for t in range(T_FR):
                im, g = render(s, float(t)); frames.append(im); vis.append(g['visible'])
            rend.append((frames, np.array(vis)))
        seen = rend[0][1].any(0) & rend[1][1].any(0)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j and seen[i] and seen[j]]
        if not pairs:
            continue
        a, b = pairs[rng.integers(len(pairs))]
        out = []
        for w, (s, (frames, vis)) in enumerate(zip(twins, rend)):
            qs, tracks = _multi_questions(s, a, b, vis, times)
            out.append(dict(frames=frames, questions=qs, names=[s['names'][a], s['names'][b]], twin=w, tracks=tracks,
                            hidden_end=[bool(not vis[-1, a]), bool(not vis[-1, b])]))
        assert np.array_equal(out[0]['frames'][-1], out[1]['frames'][-1])
        for q0_, q1_ in zip(out[0]['questions'], out[1]['questions']):
            q0_['differs'] = q1_['differs'] = bool(q0_['label'] != q1_['label'])
        return out
    raise RuntimeError('no natural twin')


# ---------------------------------------------------------------- exit twins: out-of-view dead reckoning (v0.6 §4t)
def sample_exit_twin(seed, times=(7.0, 8.0, 9.0, 10.0), tries=4000):
    """Twin 0 is an ordinary scene in which a queried ball `a` is seen (>= 2 frames) and then leaves the view before the
    last frame. Twin 1 changes ONLY a's velocity (drawn from the prior) while keeping a's position at its last-seen
    frame k: a is still seen at frame k, must be invisible at frames k+1..7, and every other ball is unchanged, so the
    two last frames are pixel-identical AND a's last-seen position is identical. The 'last-seen anchor' rule therefore
    answers both twins the same way; on questions whose labels differ ('differs') it is right on exactly one twin, and
    only extrapolating a's exit motion gets both. The pair (a, b) uses another ball b seen in both twins."""
    rng = np.random.default_rng(seed)
    for _ in range(400):
        scn = dict(yaw0=0.0, yaw_rate=rng.uniform(-0.12, 0.12), cam_speed=rng.uniform(0.15, 0.45),
                   names=list(rng.permutation(NAMES)[:3]))
        n = 3
        p0 = np.stack([rng.uniform(-1.6, 1.6, n), rng.uniform(2.0, 5.5, n)], -1)
        ang = rng.uniform(0, 2 * math.pi, n); spd = rng.uniform(0.15, 0.7, n)
        scn.update(p0=p0, v=np.stack([np.cos(ang) * spd, np.sin(ang) * spd], -1))
        if min(np.linalg.norm(p0[i] - p0[j]) for i in range(n) for j in range(i + 1, n)) < 0.8:
            continue
        frames, vis = [], []
        for t in range(T_FR):
            im, g = render(scn, float(t)); frames.append(im); vis.append(g['visible'])
        vis = np.array(vis)
        exited = [k for k in range(n) if vis[:, k].sum() >= 2 and not vis[-1, k] and vis[:, k].any()]
        if not exited:
            continue
        a = exited[rng.integers(len(exited))]
        k_last = int(np.nonzero(vis[:, a])[0].max())
        pa = obj_pos(scn, float(k_last))[a]
        for _ in range(tries):   # twin 1: new velocity for a only, same position at its last-seen frame
            an2 = rng.uniform(0, 2 * math.pi); sp2 = rng.uniform(0.15, 0.7)
            v2 = scn['v'].copy(); v2[a] = [math.cos(an2) * sp2, math.sin(an2) * sp2]
            p02 = scn['p0'].copy(); p02[a] = pa - k_last * v2[a]
            s1 = dict(scn, p0=p02, v=v2)
            fr1, vi1 = [], []
            for t in range(T_FR):
                im, g = render(s1, float(t)); fr1.append(im); vi1.append(g['visible'])
            vi1 = np.array(vi1)
            if not (vi1[k_last, a] and not vi1[k_last + 1:, a].any() and np.array_equal(fr1[-1], frames[-1])):
                continue
            bs = [j for j in range(n) if j != a and vis[:, j].any() and vi1[:, j].any()]
            if not bs:
                break
            b = bs[rng.integers(len(bs))]
            if rng.random() < 0.5:
                pair = (a, b)
            else:
                pair = (b, a)
            out = []
            for w, (s, fr, vi) in enumerate(((scn, frames, vis), (s1, fr1, vi1))):
                qs, tracks = _multi_questions(s, pair[0], pair[1], vi, times)
                out.append(dict(frames=fr, questions=qs, names=[s['names'][pair[0]], s['names'][pair[1]]], twin=w,
                                tracks=tracks, hidden_end=[bool(not vi[-1, pair[0]]), bool(not vi[-1, pair[1]])], exit_obj=int(a)))
            for q0_, q1_ in zip(out[0]['questions'], out[1]['questions']):
                q0_['differs'] = q1_['differs'] = bool(q0_['label'] != q1_['label'])
            if not any(q['differs'] for q in out[0]['questions']):
                break
            return out
    raise RuntimeError('no exit twin')
