| key | what | yours | ours [95% range] | verdict |
|---|---|---|---|---|
| ladder/model_only | Model only: W sees all 8 frames, no record | 4.90 | 5.30 [2.7, 8.4] | match |
| ladder/A | A: written record, straight-line calculator | 51.40 | 51.90 [46.0, 57.8] | match |
| ladder/A+B | A + B: + distances in the sentence | 54.20 | 53.70 [47.6, 59.5] | match |
| ladder/A+C | A + C: change-aware rule, no distances | 61.20 | 63.00 [57.2, 68.6] | match |
| ladder/A+B+C | A + B + C = current system | 64.70 | 66.80 [61.0, 72.0] | match |
| ladder/A+B+C_exact | side test: A + B + C with exact positions | 97.30 | 97.00 [95.2, 98.5] | match |
| calc/lsq_est | calculator alone, straight line, estimated positions | 54.10 | 53.50 [47.4, 59.4] | match |
| calc/ca_est | calculator alone, change-aware, estimated positions | 65.30 | 66.80 [61.1, 72.0] | match |
| calc/imm_est | calculator alone, IMM (usual tracking method), estimated | 65.00 | 66.00 [60.9, 71.5] | match |
| calc/ca_orc | calculator alone, change-aware, exact positions | 97.80 | 97.80 [96.1, 99.2] | match |
| estimator/vis_acc | measuring part: visibility accuracy (%), eval8 | 99.90 | 99.90 | match |
| estimator/pos_err_med_m | measuring part: median position error (m), eval8 | 0.10 | 0.10 | match |
| versions/eval10/V1 | Version 1: frame tokens only | (not run) | 61.82 [56.6, 66.6] | |
| versions/eval10/V2 | Version 2: + whole frame at 448 (256 tokens) | (not run) | 71.38 [66.8, 75.4] | |
| versions/eval10/V3 | Version 3: + 96 px square at 448 (256 tokens) | (not run) | 74.50 [69.9, 78.2] | |
| versions/eval10/V2-V1 | Version 2 - Version 1 | (not run) | 9.56 [5.6, 13.7] | |
| versions/eval10/V3-V2 | Version 3 - Version 2 | (not run) | 3.12 [-0.3, 6.5] | |
| versions/eval10/V1h | Version 1h: frame tokens + head start | (not run) | 64.63 [59.6, 69.4] | |
| versions/eval10/V2h | Version 2h: whole frame + head start | (not run) | 70.70 [66.2, 74.6] | |
| versions/eval10/V3-V2h | Version 3 - Version 2h | (not run) | 3.80 [0.4, 7.2] | |
| versions/eval11/current | current system (no extra look) | (not run) | 62.52 [56.9, 67.9] | |
| versions/eval11/V1h | Version 1h | (not run) | 61.24 [56.5, 65.8] | |
| versions/eval11/V2h | Version 2h | (not run) | 66.76 [62.3, 71.0] | |
| versions/eval11/V3 | Version 3 | (not run) | 70.04 [65.5, 74.3] | |
| versions/eval11/V3c | Version 3c: same square at 224 (64 tokens) | (not run) | 71.45 [67.3, 75.5] | |
| versions/eval11/V3c-V3 | Version 3c - Version 3 (cheaper look as good?) | (not run) | 1.41 [-2.0, 5.0] | |
| versions/eval11/V3c-current | Version 3c - current system | (not run) | 8.94 [3.6, 14.2] | |
| look/eval12/none | no zoomed look (current system) | (not run) | 61.83 [56.6, 67.3] | |
| look/eval12/fixed2 | schedule's last 2 looks | (not run) | 68.67 [63.8, 73.3] | |
| look/eval12/last_each2 | last sighting of each ball (2 looks) | (not run) | 72.64 [68.1, 76.9] | |
| look/eval12/fixed8 | Version 3's full schedule (~6.8 looks) | (not run) | 74.64 [70.3, 78.7] | |
| look/eval12/last-fixed2 | last sighting - schedule, 2 looks each | (not run) | 3.97 [0.5, 7.6] | |
| look/eval10/last-fixed2 | same, eval10 | (not run) | 4.17 [1.0, 7.5] | |
| look/eval13/fixed2 | schedule's last 2 looks | (not run) | 70.86 [66.3, 75.0] | |
| look/eval13/last_each2 | last sighting of each ball (2 looks) | (not run) | 75.35 [71.1, 79.0] | |
| look/eval13/last-fixed2 | last sighting - schedule, 2 looks each | (not run) | 4.49 [1.3, 7.8] | |
