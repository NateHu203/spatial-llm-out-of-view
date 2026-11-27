# Larger models on the twin test (API, zero-shot, 2026-11-27)

Does a bigger model pass the twin test without any of our tools? Each model gets the same input as our 3B model: the 8
frames (224 px) and the stored question text. Nothing is trained. Run with `src/api_twin_eval.py` through OpenRouter.

| Model | Easy control (guessing 50) | Twin accuracy on eval9 (guessing 25) | Questions |
|---|---|---|---|
| Qwen2.5-VL-72B | 75.2 [69.8, 80.2] | 3.3 [1.5, 5.3] | 695 (all of eval9) |
| Qwen3-VL-235B | 84.1 [79.5, 88.8] | 3.7 [2.3, 5.6] | 695 (all of eval9) |
| GPT-6.2 Sol (reasoning on) | 92.5 [85.0, 100.0] | 34.1 [20.0, 47.7] | 82 (first 40 pairs) |
| Our 3B model W (fine-tuned), for reference | – | 5.3 [2.7, 8.4] | 695 |

- **Twin accuracy:** a question counts only if both twins are answered correctly; 95% range from resampling pairs.
- **Easy control:** 129 ordinary scenes (seeds 50,000,000 + i, i < 1000, kept when both asked balls are visible in the
  last frame), 2 questions each about the last frame itself. GPT-6.2 Sol: the first 20 scenes only (40 questions).
- On the 40 pairs GPT-6.2 Sol was run on, the two Qwen models score 1.2 and 8.5.
- Both Qwen models give the same answer to both twins on about 90% of the questions (GPT-6.2 Sol: 58.5%).

**Limits.** Three models, one run each, zero-shot, 224 px frames. The GPT-6.2 Sol numbers come from 82 twin questions
and 40 control questions, so their ranges are wide; its range on the twin test includes 25. The data was regenerated
on a Mac with numpy 2.0.2: the fingerprint differs from `reference/data_fingerprints.json`, but all 4800 eval9 labels
equal those in `results/desktop_4080s/res_d_state.jsonl` and every pair's last frames are pixel-identical.

Files: `res_api_<model>.jsonl` (one row per call, twin test), `res_api_<model>_easy3.jsonl` (control), and a
`_summary.json` next to each.
