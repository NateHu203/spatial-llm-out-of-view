"""Collect the headline numbers from the result files and print them next to ours (reference/expected_results.json).

Every number is "twin accuracy" (%) unless stated: both twins answered correctly, on questions whose answer differs
between the twins; [low, high] = 95% range from resampling twin pairs (bootstrap).

collect(path_of) works on any folder layout: path_of(name) returns the path of a result file by its workspace name.
The authors' tools/make_reference.py calls the same function on the original files, so both sides are computed by
identical code.
"""
import json, os, sys
from . import config

# (key, plain description, workspace file, how to read it)
#   reader 'pairacc'      : a res_*.jsonl file from sts.py, scored with m_pairacc (as the slides)
#   reader 'json:<a>/<b>' : a value inside a JSON file (path of keys separated by '/'); value = [mean, [low, high]]
ITEMS = [
    # ---- table 1: the A / B / C ladder on eval9 (695 question pairs), model reading the record
    ('ladder/model_only', 'Model only: W sees all 8 frames, no record', 'res_m_w.jsonl', 'pairacc'),
    ('ladder/A', 'A: written record, straight-line calculator', 'res_m_r_lsq.jsonl', 'pairacc'),
    ('ladder/A+B', 'A + B: + distances in the sentence', 'res_flow_lsq_est_dist.jsonl', 'pairacc'),
    ('ladder/A+C', 'A + C: change-aware rule, no distances', 'res_m_r_ca.jsonl', 'pairacc'),
    ('ladder/A+B+C', 'A + B + C = current system', 'res_m_r_ca_dist.jsonl', 'pairacc'),
    ('ladder/A+B+C_exact', 'side test: A + B + C with exact positions', 'res_m_r_ca_orc_dist.jsonl', 'pairacc'),
    ('calc/lsq_est', 'calculator alone, straight line, estimated positions', 'calc_ladder.json', 'json:lsq/est/all'),
    ('calc/ca_est', 'calculator alone, change-aware, estimated positions', 'calc_ladder.json', 'json:ca/est/all'),
    ('calc/imm_est', 'calculator alone, IMM (usual tracking method), estimated', 'calc_ladder.json', 'json:imm/est/all'),
    ('calc/ca_orc', 'calculator alone, change-aware, exact positions', 'calc_ladder.json', 'json:ca/orc/all'),
    # ---- the measuring part
    ('estimator/vis_acc', 'measuring part: visibility accuracy (%), eval8', 'state_estimator_report.json', 'json:eval8/vis_acc'),
    ('estimator/pos_err_med_m', 'measuring part: median position error (m), eval8', 'state_estimator_report.json', 'json:eval8/pos_err_med_m'),
    # ---- table 2: extra looks (rule C, 3 trained heads), fresh twins eval10 (rounds 5-6) and eval11 (round 7)
    ('versions/eval10/V1', 'Version 1: frame tokens only', 'crop_ablation_r5.json', 'json:acc/base/ca/eval10'),
    ('versions/eval10/V2', 'Version 2: + whole frame at 448 (256 tokens)', 'crop_ablation_r5.json', 'json:acc/global/ca/eval10'),
    ('versions/eval10/V3', 'Version 3: + 96 px square at 448 (256 tokens)', 'crop_ablation_r5.json', 'json:acc/crop/ca/eval10'),
    ('versions/eval10/V2-V1', 'Version 2 - Version 1', 'crop_ablation_r5.json', 'json:diff/eval10/ca/global-base'),
    ('versions/eval10/V3-V2', 'Version 3 - Version 2', 'crop_ablation_r5.json', 'json:diff/eval10/ca/crop-global'),
    ('versions/eval10/V1h', 'Version 1h: frame tokens + head start', 'crop_ablation_r6.json', 'json:acc/base_box/ca/eval10'),
    ('versions/eval10/V2h', 'Version 2h: whole frame + head start', 'crop_ablation_r6.json', 'json:acc/global_box/ca/eval10'),
    ('versions/eval10/V3-V2h', 'Version 3 - Version 2h', 'crop_ablation_r6.json', 'json:diff/eval10/ca/crop-global_box'),
    ('versions/eval11/current', 'current system (no extra look)', 'crop_ablation_r7.json', 'json:acc/current/ca/eval11'),
    ('versions/eval11/V1h', 'Version 1h', 'crop_ablation_r7.json', 'json:acc/base_box/ca/eval11'),
    ('versions/eval11/V2h', 'Version 2h', 'crop_ablation_r7.json', 'json:acc/global_box/ca/eval11'),
    ('versions/eval11/V3', 'Version 3', 'crop_ablation_r7.json', 'json:acc/crop/ca/eval11'),
    ('versions/eval11/V3c', 'Version 3c: same square at 224 (64 tokens)', 'crop_ablation_r7.json', 'json:acc/crop224/ca/eval11'),
    ('versions/eval11/V3c-V3', 'Version 3c - Version 3 (cheaper look as good?)', 'crop_ablation_r7.json', 'json:diff/eval11/ca/crop224-crop'),
    ('versions/eval11/V3c-current', 'Version 3c - current system', 'crop_ablation_r7.json', 'json:diff/eval11/ca/crop224-current'),
    # ---- table 3: which frames to zoom on (Version 3's look, 3 heads, rule C re-tuned), eval12 (round 8), eval13 (round 9)
    ('look/eval12/none', 'no zoomed look (current system)', 'look_choice_r8.json', 'json:acc/eval12/none'),
    ('look/eval12/fixed2', "schedule's last 2 looks", 'look_choice_r8.json', 'json:acc/eval12/fixed2'),
    ('look/eval12/last_each2', 'last sighting of each ball (2 looks)', 'look_choice_r8.json', 'json:acc/eval12/last_each2'),
    ('look/eval12/fixed8', "Version 3's full schedule (~6.8 looks)", 'look_choice_r8.json', 'json:acc/eval12/fixed8'),
    ('look/eval12/last-fixed2', 'last sighting - schedule, 2 looks each', 'look_choice_r8.json', 'json:diff/eval12/last_each2-fixed2'),
    ('look/eval10/last-fixed2', 'same, eval10', 'look_choice_r8.json', 'json:diff/eval10/last_each2-fixed2'),
    ('look/eval13/fixed2', "schedule's last 2 looks", 'look_choice_r9.json', 'json:acc/eval13/fixed2'),
    ('look/eval13/last_each2', 'last sighting of each ball (2 looks)', 'look_choice_r9.json', 'json:acc/eval13/last_each2'),
    ('look/eval13/last-fixed2', 'last sighting - schedule, 2 looks each', 'look_choice_r9.json', 'json:diff/eval13/last_each2-fixed2'),
]


def _pairacc(path):
    sys.path.insert(0, str(config.SRC))
    from m_pairacc import by_key, stats
    rows = json.loads(open(path).readline())['rows']
    m, ci, n = stats(by_key(rows))
    return [float(m), [float(ci[0]), float(ci[1])]]


def _json(path, keys):
    v = json.load(open(path))
    # keys may themselves contain '/', e.g. 'eval10/ca/crop-global' inside 'diff': try the longest key first
    parts = keys.split('/')
    while parts:
        for k in range(len(parts), 0, -1):
            cand = '/'.join(parts[:k])
            if isinstance(v, dict) and cand in v:
                v = v[cand]; parts = parts[k:]; break
        else:
            raise KeyError(keys)
    if isinstance(v, (int, float)):
        return [float(v), None]
    return [float(v[0]), [float(v[1][0]), float(v[1][1])]]


def collect(path_of):
    out = {}
    for key, _, name, reader in ITEMS:
        p = path_of(name)
        if p is None or not os.path.exists(p):
            continue
        try:
            out[key] = _pairacc(p) if reader == 'pairacc' else _json(p, reader[5:])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f'    could not read {key} from {name}: {e!r}')
    return out


def verdict(mine, ref):
    """'match' = inside our 95% range; 'close' = within 2x its half-width (or 2 points); 'CHECK' otherwise"""
    m, r = mine[0], ref[0]
    if ref[1] is None:
        tol = max(abs(r) * 0.05, 1e-9)
        return 'match' if abs(m - r) <= tol else 'CHECK'
    lo, hi = ref[1]
    if lo <= m <= hi:
        return 'match'
    half = max((hi - lo) / 2, 2.0)
    return 'close' if abs(m - r) <= 2 * half else 'CHECK'


def main(quick=False):
    ref = json.load(open(config.REFERENCE / 'expected_results.json'))['values']
    def path_of(name):   # quick mode: rounds 5-8 write <name>_quick.json
        q = config.WORK / name.replace('.json', '_quick.json')
        return str(q if quick and name.endswith('.json') and q.exists() else config.WORK / name)
    mine = collect(path_of)
    json.dump(mine, open(config.WORK / 'my_results.json', 'w'), indent=1)
    desc = {k: d for k, d, _, _ in ITEMS}
    lines = ['| key | what | yours | ours [95% range] | verdict |', '|---|---|---|---|---|']
    for k, _, _, _ in ITEMS:
        r = ref.get(k)
        if k not in mine:
            lines.append(f'| {k} | {desc[k]} | (not run) | {_fmt(r)} | |'); continue
        lines.append(f'| {k} | {desc[k]} | {mine[k][0]:.2f} | {_fmt(r)} | {verdict(mine[k], r) if r and not quick else "-"} |')
    text = '\n'.join(lines)
    open(config.WORK / 'my_results.md', 'w', encoding='utf-8').write(
        ('QUICK MODE: numbers are meaningless (short training); this only shows the pipeline ran.\n\n' if quick else '') + text + '\n')
    print(text)
    print(f'\n    written: {config.WORK / "my_results.md"} and my_results.json')


def _fmt(r):
    if not r:
        return '-'
    return f'{r[0]:.2f}' + (f' [{r[1][0]:.1f}, {r[1][1]:.1f}]' if r[1] else '')
