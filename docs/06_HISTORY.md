# 6. What we tried that did not hold (not in this repository's code)

These ideas were tested with pass rules written before running. They are listed so that nobody repeats them without a new reason. A result that missed its pass rule but still showed a gain is marked "informative", not "failed". The code and logs remain in the authors' project folder (`v0.6/pilot/`) and the slides (`ladder_slides/`).

| When | Idea | Plain summary | Result |
|---|---|---|---|
| Oct 19–22 | **NowRoPE**: shift the model's rotary position codes to "now" | Tell the model where things are now by changing how it numbers frames | No gain in the pilot |
| Oct 23 | **Shared-trajectory supervision (STS)** | Teach the model a ball's path with an extra training target | Stopped by its own kill rule |
| Oct 23 | **History twins, natural or disguised** | Train on twin videos so the model must use earlier frames | The model learned to spot which data source a video came from instead of using motion |
| Oct 23 | **Round 3: anchor (A1), product of experts (C1), response matching (A2)** | Three ways to push the model off "answer from the last frame" | All three missed their pass rules |
| Oct 26 | **Stage C: memory tokens (C2), visual-stream state (C3)** | Give the state to the model as extra tokens instead of a sentence | The model ignored them (5–10% vs 81% for the sentence) |
| Nov 5 | **Reference picture (D)** | Measure motion relative to a saved picture of the ball, to cancel shared error | Frame-to-frame error −25%, but twin accuracy barely changed; dropped |
| Nov 5 | **Round 4: zoom vs whole frame, first test** | The zoomed look on the development twins | Became rounds 5–7 (kept: block 6) |
| Nov 8 | **Version 3s: a tighter 64 px square** | More zoom for the same tokens | +1.3 on eval11, −3.2 on eval10: not reliably better |
| Nov 9 | **Round 8: residual rule (4 looks) vs 8 fixed looks** | Look where the estimate strays most from its path | −2.8 on eval12: the frames it picked barely move the forecast. Informative |
| Nov 9 | **Question-first looks, search over sampled worlds, depth-2 adaptive looks** (simulation) | Choose looks after reading the question; plan two looks ahead | No gain: every question asks about the same two balls and depends on the same state |
| Nov 9 | **Round 9: weighted ends (2:1) and one adaptive agent step** | Spend 6 looks by forecast weight; let the first readings steer the rest | +1.1 and +1.4, not significant: with 4+ looks, placement barely matters |
| Nov 9 | **Re-centring the zoom square** | Look again centred on the first reading | Dropped before running: an off-centre square does not hurt (heads were trained with 6 px jitter) |
| Nov 9 | **Simulate-and-check, ceiling only** | Give each ball its true motion story, to see how much room there is | +6.6 on eval13, so there is room. A simple automatic picker is 5–10 points worse than rule C. Open; see [05_STATUS_AND_NEXT_STEPS.md](05_STATUS_AND_NEXT_STEPS.md) |
