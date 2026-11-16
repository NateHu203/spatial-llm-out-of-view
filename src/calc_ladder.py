"""Calculator-only ladder on eval9 (no language model): twin accuracy of the calculator answering directly from the
record, for the straight-line rule (lsq), the usual tracking method (imm) and the change-aware rule (ca), on estimated
('est') and exact ('orc') positions. Same scoring as m_pairacc.py (pair bootstrap, 1000 draws, seed 0).
Repo version of the authors' flow_calc_qtype.py, writing JSON instead of a log.
Usage: python calc_ladder.py   -> calc_ladder.json (in the workspace)"""
import json, os
import state_rules as SR
from m_pairacc import by_key, stats

if __name__ == '__main__':
    PR = json.load(open(os.path.join(SR.HERE, 'state_rules_results.json')))
    out = {}
    for rule in ('lsq', 'imm', 'ca'):
        for src in ('est', 'orc'):
            both = by_key(SR.evaluate(rule, PR[f'{rule}/{src}']['params'], 'eval9', src, rows_out=True))
            out[f'{rule}/{src}'] = {q or 'all': stats(both, q) for q in (None, 'leftright', 'closer')}
            print(rule, src, out[f'{rule}/{src}'], flush=True)
    json.dump(out, open(os.path.join(SR.HERE, 'calc_ladder.json'), 'w'), indent=1, default=float)
