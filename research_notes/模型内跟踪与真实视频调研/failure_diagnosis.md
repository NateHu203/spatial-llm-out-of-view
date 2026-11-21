# Failure diagnosis: why video-language models ignore earlier frames and fail at object permanence, motion extrapolation and state tracking

Notes compiled 2026-11-19. Verification labels used below:
- **[page]** = the number or claim was read on the paper's own arXiv page (abstract or HTML full text) during this research.
- **[snippet]** = seen only in a search-result excerpt; treat as uncertain until the paper is opened.
Several HTML pages were longer than the reader's window (noted per item), so appendices were not read.

---

## 1. Do video LLMs rely on single frames / the last frame, ignore order, or exploit static bias?

### Takeaway
Yes, and the effect is large: on order-controlled tests most models sit near chance while humans are at 85-94%, and fine-tuned VideoQA models can score above 90% on shortcut items and below 4% on their counterfactual twins, the same below-chance signature the project sees (5% vs 25%). Newer work (2025-2026) shows the picture is model-specific: for Qwen-family models the temporal signal comes from the causal order of frame tokens and is aggregated into the last frame's tokens, not from the rotary position IDs.

### Cited Findings

**Benchmarks with counterfactual or order controls**
- Vinoground (Zhang, Cai, Lee; 3 Oct 2024): 1000 short video-caption pairs differing only in event order. Chance is 25.00 / 25.00 / 16.67 (text / video / group). Best model (GPT-4o + CoT, 32 frames) 59.20 / 51.00 / 35.00; GPT-4o without CoT 54.00 / 38.20 / 24.60; humans 93.40 / 94.00 / 90.00 (91.40 / 90.80 / 85.20 with 32 frames, no audio). Most open-source and CLIP-based models are at or below chance. [page] — [Vinoground](https://arxiv.org/html/2410.02763v1)
- Vinoground frame-count ablation: GPT-4o with 1 frame scores 28.20 / 28.00 / 10.00, rising to 54.00 / 38.20 / 24.60 at 32 frames and falling to 49.00 / 34.80 / 19.00 at 64 frames. The authors state that on EgoSchema, ActivityNet-QA, MSVD and MSRVTT performance barely changes with the number of frames ("single-frame bias", citing Lei et al.). [page] — [Vinoground](https://arxiv.org/html/2410.02763v1)
- TOMATO (Shangguan et al., Yale; arXiv Oct 2024, ICLR 2025): defines Multi-Frame Gain, Frame Order Sensitivity and Frame Information Disparity; 1,484 questions over 1,417 videos; reports a 57.3% human-model gap with the best model and concludes that models recognise events in isolated frames but fail to interpret them as a continuous sequence; argues temporal reasoning is overestimated because many existing questions are solvable with single, few, or out-of-order frames. [snippet, consistent across ICLR proceedings abstract and arXiv record] — [TOMATO](https://arxiv.org/abs/2410.23266v1)
- TempCompass (Liu et al., Mar 2024): uses conflicting video pairs/triplets with the same static content but different temporal properties; reports weak temporal perception in SOTA video LLMs. [snippet] — [TempCompass](https://arxiv.org/html/2403.00476v3)
- CLAVI (Rawal et al., A*STAR; v2 30 Sep 2023): after fine-tuning, consistent accuracy on shortcut ("control") items vs counterfactual items: JustAsk 98.0 vs 3.6 (video-counterfactual) / 2.4 (text-counterfactual); Singularity-Temporal 92.7 vs 1.7 / 0.5; All-In-One+ 98.1 vs 1.2 / 0.8; FrozenBiLM 93.2 vs 54.1 / 57.2. Balanced accuracy is 72-80% for all four, so the aggregate hides the failure. [page] — [CLAVI / "Illusion of Joint Multimodal Understanding"](https://arxiv.org/html/2306.08889v2)
- TOC-Bench (Chen et al.; 11 May 2026): object identity/state through occlusion, disappearance and reappearance; a three-layer temporal-necessity filter removes 60.7% of candidate QA pairs (i.e. most candidate items were answerable without temporal evidence); final benchmark 2,323 QA pairs over 1,951 videos; authors conclude "object-centric temporal coherence is a key bottleneck". Per-model numbers not on the abstract page. [page, abstract only] — [TOC-Bench](https://arxiv.org/abs/2605.09904)

**Channel decomposition (which signal carries "time")**
- Rahman, "What Does a Temporal Benchmark Score Measure?" (14 Jul 2026, v3 19 Nov 2026, WACV 2027): on TempCompass, multi-frame gain (full minus single frame) is +0.22 (Molmo2) and +0.21 (Qwen3-VL) overall; order gain (full minus shuffled) is +0.17 and +0.15. By aspect, order gain is +0.37 for attribute change, about +0.30 for event order, but +0.01 for "action" and +0.01 to +0.05 for "speed", so those sub-scores do not measure order use. [page] — [Rahman 2026](https://arxiv.org/html/2607.12304v3)
- Same paper: flipping only the RoPE indices (pixels unchanged) makes Molmo2 answer the forward events (TVBench moving_direction: forward-label rate 0.62-0.67, reversed-label 0.05-0.08), whereas Qwen3-VL follows the pixels it sees (reversed-label 0.61-0.65, forward-label 0.05-0.06). Molmo2 is "position-dominant", Qwen3-VL is "visual-sequence-dominant", although their aggregate TempCompass scores are similar. [page] — [Rahman 2026](https://arxiv.org/html/2607.12304v3)
- Same paper, arrow-of-time prior: reversed clips lower accuracy by about 0.19-0.33 (e.g. 0.79 to 0.56); six of seven models give the forward-content answer for 59-75% of their reversed-clip errors versus a random baseline of about 37%. [page] — [Rahman 2026](https://arxiv.org/html/2607.12304v3)
- Same paper, probe: in Qwen3-VL-4B frame-order decodability rises from 0.27 at layer 3 to 0.84 at layer 20; under reversed RoPE this rise does not occur. [page] — [Rahman 2026](https://arxiv.org/html/2607.12304v3)

**Qwen2.5-VL specifically**
- Shi et al., "Causality Matters: How Temporal Information Emerges in Video Language Models" (NTU; v2 15 Nov 2025), Qwen2.5-VL-7B main, Qwen2.5-VL-3B and LLaVA-OneVision-7B in appendix, TempCompass: reversing position IDs in all layers changes ground-truth probability by only 1.13-1.76%, while reversing frame order with original IDs drops multiple-choice probability by 15.45%. Shuffling video position IDs in the first layer has negligible effect (-0.02% / +0.07%); shuffling query position IDs drops it by 53.09%. [page] — [Shi et al. 2025](https://arxiv.org/html/2508.11576v2)
- Same paper, Qwen2.5-VL-3B: reversing frame order drops probability by up to 8.93%; reversing position encodings changes it by under 1.4%; inter-frame effects are weaker than in the 7B model. [page] — [Shi et al. 2025](https://arxiv.org/html/2508.11576v2)
- Same paper, last-frame aggregation: restricting the query's attention to frame 1 costs over 8% on multiple choice, restricting it to the last frame (frame 4) costs under 2%; blocking inter-frame attention has its peak effect near layer 10 (4-6% drops) and blocking frame-to-query attention near layer 15 (over 10% for captioning). The authors' reading: temporal cues are built by inter-frame attention, aggregated in the final frame, then passed to the query tokens. [page] — [Shi et al. 2025](https://arxiv.org/html/2508.11576v2)

**Other LLaVA-family mechanism studies**
- Kim, Kim, Han, "Map the Flow" (arXiv 2510.13251, v2 3 Mar 2026; ICLR 2026): in LLaVA-NeXT-7B-Video-FT, blocking cross-frame attention in layers 1-16 cuts accuracy by 18.0-60.8% across five TVBench tasks (Moving Direction -44.8%, Object Count -60.8%). The image-only LLaVA-NeXT-7B is mostly unaffected by the same knockout, i.e. the cross-frame dependency is created by video instruction tuning. [page] — [Map the Flow](https://arxiv.org/html/2510.13251v2)
- Same paper: temporal concepts appear from the middle layers (logit lens), video-to-question flow happens in early-to-middle layers, question-to-last-token in middle-to-late layers, answer probability rises from about layer 20. Failed samples show the same routing as successful ones, so the authors place the errors in early spatiotemporal representation building; with cross-frame attention disabled, the model falls back on static-scene bias. Note an internal inconsistency in the source on the fraction of attention edges (abstract: 58% suppressed; Table 3: 42% retained for this model). [page] — [Map the Flow](https://arxiv.org/html/2510.13251v2)
- Gou et al., "An Empirical Study on How Video-LLMs Answer Video Questions" (21 Aug 2025; LongVA, InternVideo2.5, LLaVA-Video-7B, LLaVA-OneVision-7B; Qwen-VL not studied): knocking out video-temporal or video-spatial self-attention in all layers causes minimal degradation; knocking out language-to-video attention retains at most 70% (as little as 48%) of performance; language-to-video effects concentrate around layers 12-16 and blocking beyond 60% depth has little effect. [page] — [Gou et al. 2025](https://arxiv.org/html/2508.15360v1)

### Inferences
- Gou et al. (temporal attention among video tokens barely matters on MVBench/EgoSchema/Video-MME) and Map the Flow / Shi et al. (cross-frame attention matters on TVBench/TempCompass) do not contradict each other once the benchmark is taken into account: the first set of benchmarks is largely answerable from single frames, so cross-frame computation is not exercised.
- For Qwen-family models, temporal order is carried by the causal token sequence, not by the rotary temporal index. This is a direct candidate explanation for why shifting rotary position codes did nothing in the project.
- "Last frame acts as an aggregation point" (Shi et al.) means a model that reads only last-frame tokens is using the normal read-out route; the failure is that the last-frame tokens were never loaded with trajectory information for the out-of-view case (upstream cross-frame computation in roughly layers 1-16), not that the query "forgets" to look at earlier frames.
- The project's 5% on twins (below 25% chance) matches the CLAVI pattern (1-4% on counterfactuals, >90% on controls): a deterministic shortcut keyed on the shared final frame gives the twin's answer systematically, which is worse than guessing.

### Gaps
- No paper found that measures a "last-frame-only" answer rate for out-of-view object localisation specifically; the last-frame result above is about aggregation, measured on TempCompass with 4 frames.
- TOMATO and TempCompass per-model numbers were not read on the source pages.
- VideoMME-specific shuffling analyses were not located in this pass.

---

## 2. Object permanence and intuitive physics: do self-supervised video encoders represent hidden objects while VLMs do not?

### Takeaway
On easy synthetic tests (IntPhys) V-JEPA reaches 98% while multimodal LLMs are near chance, but on the harder IntPhys 2 every model class, including V-JEPA 2, is at 50-58% against 96% for humans; so "video SSL encoders have object permanence" holds only for short, simple occlusions. For VLMs, tracking identical objects through swaps is at chance (about 1/3) even for frontier models, and training a Qwen2.5-VL-3B to answer directly does not move the loss.

### Cited Findings
- Garrido et al., "Intuitive physics understanding emerges from self-supervised pretraining on natural videos" (FAIR; 17 Feb 2025): V-JEPA zero-shot pairwise accuracy 98% on IntPhys (95% CI 95-99), 66% on GRASP, 62% on InfLevel-lab. Qwen2-VL, Gemini 1.5 Pro and VideoMAEv2 are reported only as "marginally above" randomly initialised networks (no exact figures in the text). [page] — [Garrido et al. 2025](https://arxiv.org/html/2502.11831v1)
- Same paper, object permanence (V-JEPA-L trained on HowTo100M vs untrained): IntPhys 85.7% vs 51.4%; GRASP 70.7% vs 54.1%; InfLevel 72.1% vs 52.5%. No significant gain for colour constancy, solidity or collision. [page; the last sentence from snippet] — [Garrido et al. 2025](https://arxiv.org/html/2502.11831v1)
- Same paper, ablations: prediction in representation space matters more than the masking scheme (random masking costs about 5 points on IntPhys); training data matters (SSv2 alone is near chance, HowTo100M best, 128 hours of it still keeps >70% on all properties); a 115M-parameter model still exceeds 85%. Authors' explanation for MLLMs: they are trained to predict text with video interleaved afterwards and have no video-prediction objective. [page] — [Garrido et al. 2025](https://arxiv.org/html/2502.11831v1)
- IntPhys 2 (Bordes et al., FAIR; 11 Jun 2025), overall accuracy (chance 50%): GPT-4o 53.75, Qwen-VL 2.5 52.27, Gemini 1.5 Pro 52.27, Gemini 2.5 Flash 55.63, VideoMAEv2-g 53.75, Cosmos-4B 49.41, V-JEPA-h 53.75, V-JEPA 2-h 57.51, human 96.44. Object permanence (fixed / moving camera): Qwen-VL 2.5 53.85 / 54.41, Gemini 2.5 Flash 64.42 / 58.82, V-JEPA-h 63.46 / 67.65, V-JEPA 2-h 59.62 / 57.35, human 100.0 / 99.26. Most models get worse as more frames are added. Authors attribute the gap to missing physical world models and weak short-term memory, since IntPhys 2 needs longer memory of occluded objects. [page] — [IntPhys 2](https://arxiv.org/html/2506.09849v1)
- VET-Bench, "Can Vision-Language Models Solve the Shell Game?" (Liu and Lee, NUS; 9 Mar 2026): with 3 identical objects and 5 swaps, Gemini-3-Pro 0.34, Gemini-3-Flash 0.30, Qwen3-VL-8B 0.30, GLM-4.6V-Flash 0.34, Doubao-Seed-2.0 0.30 (random 0.33). Zero swaps is near-perfect for most models; one swap already drops to chance. [page] — [VET-Bench](https://arxiv.org/html/2603.08436)
- Same paper, shortcuts in an existing benchmark: on the Perception Test cups-game subset, distinct/transparent cups and informative final frames allow bypassing tracking; after filtering (107 of 189 clips kept) Gemini-3-Pro falls from 0.80 to 0.31. [page] — [VET-Bench](https://arxiv.org/html/2603.08436)
- Same paper, theory and training: the authors prove tracking with 5 or more objects is NC1-complete, so (assuming TC0 != NC1) a fixed-depth transformer cannot do it without intermediate computation. Qwen2.5-VL-3B trained to give direct answers on 500 videos had its loss stuck at chance for 60 epochs. Making the model first write object trajectories as text (timestamps every 0.5 s plus normalised coordinates, "Spatiotemporal Grounded CoT") and fine-tuning Molmo2 on 300 samples for one epoch raises accuracy from near-random to 91%. [page] — [VET-Bench](https://arxiv.org/html/2603.08436)
- VSI-Bench, "Thinking in Space" (Yang et al.; v2 2 Jul 2025): best model Gemini-1.5 Pro 45.4% vs humans 79.2% (tiny subset); about 71% of its errors are attributed to spatial reasoning rather than perception or language; cognitive-map accuracy is 0.64 for the nearest distance bin and 0.00 for the farthest; zero-shot CoT and Tree-of-Thoughts reduce average performance by about 4%; prompting the model to produce a cognitive map raises relative-distance accuracy from 46.0% to 56.0% (66.0% with a ground-truth map). [page] — [VSI-Bench](https://arxiv.org/html/2412.14171v2)
- VSI-Bench shortcut analysis, "Benchmark Designers Should 'Train on the Test Set'" (arXiv 2511.04655, Nov 2025): a blind LLaVA-Video-7B rises from 25.9% to 44.7% after fine-tuning on VSI-Bench-style data; on the debiased split the blind model drops from 44.7% to 32.0%. [snippet] — [Train on the Test Set](https://arxiv.org/html/2511.04655v1)

### Inferences
- The clean statement is: a representation-space prediction objective gives violation-of-expectation sensitivity to short occlusions; it does not by itself give long-horizon memory of hidden objects (IntPhys 2), and nobody has shown that such an encoder attached to an LLM yields a VLM that can report a hidden object's position.
- The VET-Bench result is the closest published analogue of the project's finding that state given as text is used while the same state in a non-text channel is not: an explicit textual trajectory is what makes tracking trainable in a 3B-class model, whereas direct-answer supervision fails.
- The NC1 argument applies to permutation tracking of 5 or more objects; extrapolating one ball's motion is not covered by that proof, so depth limits are a hypothesis for the project, not an established cause.

### Gaps
- No verified numbers for MLLMs on GRASP, InfLevel, Physion or PhysBench were read in this pass.
- No study found that probes a VLM's own vision tower (as opposed to V-JEPA) for the position of an object that has left the field of view.

---

## 3. Is the information present in the vision features but unused by the language model?

### Takeaway
Both failure types exist and published probes separate them: for static spatial concepts the information is often linearly decodable inside the LLM while the answer is still wrong ("cognitive" or read-out failure); for motion, one probe study of the Qwen2.5-VL vision encoder finds camera-motion cues peak in an early block and decay with depth, i.e. a perceptual loss before the LLM. Which applies to object trajectory in the project is untested.

### Cited Findings
- Theodoridis et al., "Probing Visual Concepts in Lightweight VLMs for Automated Driving" (v2 6 Aug 2026; Ovis2.5-2B, InternVL3.5-2B, Qwen3-VL-2B, VST-3B, all under 4B): linear probes at vision encoder, projector and LLM on counterfactual CARLA image pairs. They define perceptual failure (not linearly encoded, probe and model both fail) and cognitive failure (probe succeeds, model answers wrongly); e.g. InternVL3.5 on a spatial-relation task at 5 m is near chance while the probe is near perfect. The projector is described as a bottleneck for orientation (InternVL3.5) and count (Ovis2.5, InternVL3.5). Adding or subtracting probe directions during generation changes answers in the expected direction for presence, count and spatial concepts (qualitative evidence). [page] — [Theodoridis et al. 2026](https://arxiv.org/html/2603.06054)
- Same paper: spatial relations are not linearly decodable from average-pooled encoder features but are from region-pooled ones (about 85% and 73% for Ovis2.5 and Qwen3-VL at 5 m), and in the LLM a spatial concept jumps from roughly 10% to 90% decodability within one or two middle layers. [page] — [Theodoridis et al. 2026](https://arxiv.org/html/2603.06054)
- Feng et al., "Geometry-Guided Camera Motion Understanding in VideoLLMs" (v2 24 Mar 2026): most off-the-shelf video LLMs are near the 25% random rate on 4-way camera-motion VQA; probing the Qwen2.5-VL vision encoder, performance peaks at the first full-attention block (index 7) and declines in later blocks; a Q-Former probe on its features reaches 0.450 instance accuracy versus 0.738 for a classifier on a 3D foundation model (VGGT). Loss through token compression is hypothesised, not measured; the LLM stage was not analysed. Their fix is to write the predicted motion labels as text in the prompt, with model weights unchanged. [page] — [Feng et al. 2026](https://arxiv.org/html/2603.13119)
- Zhang et al., "Anchored, Not Graded" (4 Jun 2026; ECCV 2026): VLMs predict surface slant only at a few anchor values; supervised fine-tuning partially reduces this but residual anchoring persists; the authors interpret it as a failure at the representation-to-output interface, not necessarily an absence of geometric encoding. Probes on Qwen2.5-VL-3B vision tokens are mentioned only in a search excerpt. [page for abstract claims; probe detail is snippet] — [Zhang et al. 2026](https://arxiv.org/abs/2606.06714)
- Attention to visual tokens: FastV reports image tokens are on average 64% of input tokens yet receive far less attention relative to their number in deep layers; a LLaVA-NeXT-7B analysis reports visual tokens are 80-90% of tokens but receive less attention than system and text tokens, with the top 20% of visual tokens taking 60% of the visual attention. [snippet] — [FastV](https://arxiv.org/pdf/2403.06764), [HiRED](https://arxiv.org/pdf/2408.10945)
- Text dominance (arXiv 2508.10552, Aug 2025): measured across image, video, audio, time-series and graph inputs; attributed to attention dilution from redundant non-text tokens, fusion design and task formulations that favour text. [snippet] — [When Language Overrules](https://arxiv.org/html/2508.10552v1)
- Assouel et al., "Visual symbolic mechanisms" (Jun 2025, v2 12 Dec 2025; includes Qwen2.5-VL 3B/7B/32B): VLMs bind objects through content-independent spatial "position IDs" stored in the keys of each object's image patches; three head groups in Qwen2-VL: ID retrieval (layers 12-16), ID selection (18-19), feature retrieval (20-27). The Qwen family uses relative rather than absolute spatial coding. Patching ID-selection head activations from clean trials raises accuracy on binding-error trials by 11.1 points for Qwen2.5-VL-3B. [page] — [Assouel et al. 2025](https://arxiv.org/html/2506.15871v2)

### Inferences
- The binding result implies the LLM addresses an object through the image patches where it currently is. An object that is not in the last frame has no patches there to act as an address, so any read-out must go through an earlier frame's patches plus an extrapolation step that the model was never trained to perform. This is a plausible mechanistic reason for "answers from where the ball was last visible".
- Qwen2.5-VL merges adjacent frame pairs and 2x2 patches before the LLM; Feng et al. suggest but do not show that compression attenuates motion cues. This stage is unprobed in the literature found here, which makes it a cheap, novel probe target for the project.

### Gaps
- No published linear probe of object velocity or out-of-view position across vision encoder, merger and LLM layers of Qwen2.5-VL was found.
- The widely discussed "modality gap" literature for CLIP-style encoders was not read on source pages in this pass.

---

## 4. Why are new non-text channels (soft tokens, memory tokens, injected embeddings) ignored after light fine-tuning while text is used?

### Takeaway
The best-supported explanation is a decoder-side one: the LLM's read-out is learned from text and extracts information only along text-aligned directions, so information can be present in the residual stream at every layer and still not affect the output; what changes this is a training objective that makes the output depend on that information, not which adapter is trained. Direct evidence on soft/memory tokens specifically is thin.

### Cited Findings
- Billa, "Modality Collapse as Mismatched Decoding" (26 Feb 2026; single-author preprint; Ultravox, Qwen2-Audio, LLaVA-1.5-7B, Prismatic DINOv2/SigLIP): non-text attributes stay 3-55x above chance at the final LLM layer (e.g. Prismatic-DINOv2 object category 0.792 vs 0.014 chance) yet are not used. Ablating 53 modality-specific modes (71% of adapter variance) reduces cross-entropy loss by 11.1%, whereas ablating a matched set of text-aligned modes changes it by -0.07%: the extra variance acts as noise to the decoder. [page] — [Billa 2026](https://arxiv.org/html/2602.23136v1)
- Same paper, encoders: with a text-aligned encoder (SigLIP) visual attribute decodability rises slightly through the LLM (+1.9% object category) versus no change for DINOv2; the author argues text-aligned encoders discard non-textual information before the decoder, so the collapse simply happens earlier. [page] — [Billa 2026](https://arxiv.org/html/2602.23136v1)
- Same paper, what makes it usable: a text-centric LoRA left speaker and lexical accuracy unchanged (55.0% vs 54.5%); a LoRA trained with an emotion objective (rank 16, alpha 32) raised held-out emotion generation accuracy from 17.3% to 61.8%. The author also cites own concurrent work that LoRA on the LLM backbone reduces text dominance by 23.9% while adapter-only training increases it by 26.5%. [page; the last figure is a self-citation not checked at its source] — [Billa 2026](https://arxiv.org/html/2602.23136v1)
- An ACL 2026 paper on cross-modal alignment argues visual tokens must remain compatible with the frozen LLM's anisotropic "text cone" and that projector updates pushing tokens into the modality gap make the frozen LLM fail on them; proposes an anchor-preserving projection. [snippet] — [ACL 2026 long paper 1699](https://aclanthology.org/2026.acl-long.1699.pdf)
- Text-pathway bottleneck elsewhere: a linear probe on frozen VLM features recovers hand joint angles far more accurately than the model's text output. [snippet] — [arXiv 2603.06459](https://arxiv.org/html/2603.06459)
- Feng et al. choose to pass externally computed motion to a video LLM as a text list in the prompt rather than as features. [page] — [Feng et al. 2026](https://arxiv.org/html/2603.13119)
- VET-Bench: writing the trajectory as text tokens before answering is what makes a small VLM trainable on tracking. [page] — [VET-Bench](https://arxiv.org/html/2603.08436)

### Inferences
- Three separable reasons a new channel is ignored, each testable: (a) geometry: the injected vectors lie off the text manifold in norm or direction, so existing attention keys do not match them; (b) no gradient pressure: if any easier feature (last frame, data source) already lowers the loss, the path through the new tokens gets no useful gradient; (c) capacity of what is trained: LoRA on attention matrices can re-route attention but must also teach downstream MLPs to decode the new code.
- Billa's result favours (b) over (c): the same LoRA rank did or did not make information usable depending on the objective. For the project this predicts that memory tokens will be used only if training contains examples where they are the sole way to get the answer right (twins with identical frames and source), preceded by a stage where the token channel is trained alone on an easy read-out task ("say the coordinates held in this token").
- The fact that the state is read well as a sentence shows the downstream computation exists; the missing piece is a translation from the injected vector into that same internal representation. A direct diagnostic follows in section 7 (patching text-derived activations).

### Gaps
- No controlled study was found comparing soft-token injection versus text for the same content across training durations, norm matching, or LoRA versus full fine-tuning. The claims in "Inferences" about alignment pre-training and norm matching are therefore hypotheses.
- Classic prompt/prefix-tuning papers were not re-read on source pages in this pass.

---

## 5. Entity and state tracking inside language models, and VLM analogues

### Takeaway
Text LLMs have an identifiable entity-tracking circuit that fine-tuning strengthens rather than replaces, and VLMs have an analogous binding circuit indexed by spatial position; both are read-out mechanisms over tokens that are present in context. No mechanistic study was found of state tracking over time in video LLMs.

### Cited Findings
- Kim and Schuster, "Entity Tracking in Language Models" (ACL 2023; arXiv May 2023): tests tracking the final state of entities through described state changes; only GPT-3.5-class models pre-trained heavily on code tracked non-trivially without fine-tuning; fine-tuned T5 could learn it with accuracy falling on harder splits. [snippet from abstract] — [Kim and Schuster 2023](https://arxiv.org/abs/2305.02363v1)
- Prakash et al., "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking" (ICLR 2024; arXiv Feb 2024): primarily the same circuit performs entity tracking in LLaMA-7B and its fine-tuned variants; the fine-tuned models' gain is attributed mainly to better handling of positional information by that circuit. [snippet from abstract] — [Prakash et al. 2024](https://arxiv.org/abs/2402.14811)
- Feng and Steinhardt (binding IDs, arXiv Oct 2023): LMs attach abstract binding-ID vectors to entities and attributes and answer by matching IDs; Dai et al. (EMNLP 2024) refine this with an ordering-ID signal. [snippet] — [Feng and Steinhardt](https://arxiv.org/html/2310.17191v2), [Dai et al. 2024](https://aclanthology.org/2024.emnlp-main.967.pdf)
- A 2026 preprint on dynamic entity tracking describes a "retrieval-conditioned rebinding" attention-head circuit for swaps and reports that its representational signature differs across model families. [snippet; peer-review status unknown] — [arXiv 2606.08644](https://arxiv.org/html/2606.08644)
- VLM binding: Saravanan et al. (May 2025) find a binding-ID-like mechanism in VLMs with object information carried by image tokens at the object's location; Assouel et al. identify position-ID head groups in Qwen2-VL / Qwen2.5-VL (see section 3). [snippet for the first, page for the second] — [Saravanan et al. 2025](https://arxiv.org/html/2505.22200v1), [Assouel et al. 2025](https://arxiv.org/html/2506.15871v2)
- "Seeing to Generalize: How Visual Data Corrects Binding Shortcuts" (ICML 2026): image-based training moves models away from a positional binding shortcut learned from text-only training (shown with interchange interventions and attention knockouts). [snippet] — [ICML 2026 listing](https://icml.cc/virtual/2026/79318)
- Li, Nye and Andreas (implicit state representations): no source page was retrieved in this pass.

### Inferences
- Prakash et al. imply that LoRA fine-tuning is most likely to succeed when it sharpens a mechanism the base model already has. The base LLM already tracks states described in text, which is consistent with the text-state condition working; there is no evidence that Qwen2.5-VL has a pre-existing mechanism that updates an object's state from visual motion, so LoRA has nothing to sharpen and falls back on the last-frame read-out.
- Since VLM binding is indexed by where an object's patches are, a testable prediction is that the position-ID heads (roughly the middle third of layers) point at the ball's last visible patches for both twins and carry no twin-specific signal.

### Gaps
- No circuit-level study of temporal state updating (object moves, state changes across frames) in a video LLM was found.

---

## 6. Shortcut learning in counterfactual / paired video training

### Takeaway
It is well documented that fine-tuned video-QA models exploit shortcuts and collapse on counterfactual twins, and that benchmark builders must filter the majority of candidate items to remove them; however, no paper was found that reports the project's specific failure (the model learning which data source a twin came from).

### Cited Findings
- CLAVI: fine-tuned models reach >90% on shortcut instances and below 4% on counterfactual instances for three of four models; balanced accuracy of 72-80% hides it. [page] — [CLAVI](https://arxiv.org/html/2306.08889v2)
- TOC-Bench: temporal-necessity filtering removed 60.7% of candidate items. [page] — [TOC-Bench](https://arxiv.org/abs/2605.09904)
- VET-Bench: filtering the Perception Test cups-game clips for appearance and final-frame cues drops Gemini-3-Pro from 0.80 to 0.31; overlaid arrows in another benchmark let Gemini-2.5-Pro score 56% while it stays at chance without them. [page] — [VET-Bench](https://arxiv.org/html/2603.08436)
- VSI-Bench: fine-tuning raises a blind (no-video) model from 25.9% to 44.7%; iterative pruning of text-answerable samples (about 70% retained) gives a debiased split. [snippet] — [Train on the Test Set](https://arxiv.org/html/2511.04655v1)
- Remedies reported in the literature (descriptions only, not verified in detail): hard negatives that keep objects fixed and change only the action (Feint6K line of work); synthetic counterfactual preference pairs (CounterVid, about 26k pairs); counterfactual videos built by horizontal flips and temporal reversal with a consistency reward (CRPO). [snippet] — [Feint6K-related](https://arxiv.org/html/2407.13094v1), [CounterVid](https://www.arxiv.org/pdf/2601.04778), [CRPO](https://arxiv.org/pdf/2605.21988)
- Arrow-of-time prior as a shortcut: models give the forward-content answer for 59-75% of reversed-clip errors. [page] — [Rahman 2026](https://arxiv.org/html/2607.12304v3)

### Inferences
- The "train a blind/ablated model on your own data" test from the VSI-Bench shortcut paper transfers directly: train a classifier (or the VLM itself) on a single frame, or on the last frame only, to predict the twin label or the data source. Any accuracy above chance quantifies the leak that the main model can exploit.
- A twin design that removes source cues by construction is to generate both members of every pair with the same pipeline and to make all non-twin training videos with that pipeline too, so that "source" has zero mutual information with the answer.

### Gaps
- No published report of a model learning data-source identity from mixed twin/non-twin video training was found; this claim about the project's failure has no direct literature counterpart yet.

---

## 7. Which hypotheses best explain the project's failures, and what cheap diagnostics would separate them?

### Takeaway
The literature supports four concrete, separable hypotheses: (H1) motion/trajectory is attenuated in the vision tower or merger; (H2) it survives but the LLM never builds an out-of-view state because nothing in the last frame's tokens can be addressed; (H3) the rotary temporal index is not the channel Qwen uses for time; (H4) new token channels fail because of objective and read-out mismatch, not capacity. Each has a probe, knockout or patching test that costs hours, not days.

### Cited Findings (evidence each hypothesis rests on)
- H1: camera-motion probe accuracy in the Qwen2.5-VL vision encoder peaks at block 7 and declines afterwards. [page] — [Feng et al. 2026](https://arxiv.org/html/2603.13119)
- H2: in Qwen2.5-VL temporal cues are aggregated into last-frame tokens (query restricted to last frame costs under 2%), with inter-frame construction near layer 10 and frame-to-query transfer near layer 15; VLM binding uses position IDs located in each object's own image patches. [page] — [Shi et al. 2025](https://arxiv.org/html/2508.11576v2), [Assouel et al. 2025](https://arxiv.org/html/2506.15871v2)
- H2 (training side): cross-frame attention only becomes causally important after video instruction tuning; failures share the routing of successes, pointing to early representation building. [page] — [Map the Flow](https://arxiv.org/html/2510.13251v2)
- H3: reversing position IDs changes Qwen2.5-VL-3B answer probability by under 1.4% while reversing frames changes it by up to 8.93%; Qwen3-VL follows the visual sequence under RoPE flips whereas Molmo2 follows RoPE. [page] — [Shi et al. 2025](https://arxiv.org/html/2508.11576v2), [Rahman 2026](https://arxiv.org/html/2607.12304v3)
- H4: information decodable at the final LLM layer is unused unless an objective requires it (emotion LoRA: 17.3% to 61.8%); direct-answer training of Qwen2.5-VL-3B on tracking stays at chance for 60 epochs, textual trajectory output fixes it in a similar-size model. [page] — [Billa 2026](https://arxiv.org/html/2602.23136v1), [VET-Bench](https://arxiv.org/html/2603.08436)
- Method precedents for the diagnostics: probes at encoder/projector/LLM with perceptual-versus-cognitive classification and probe-direction steering; attention knockout by edge type and layer window; activation patching of clean activations at a given layer; frame-order probe by layer. [page] — [Theodoridis et al. 2026](https://arxiv.org/html/2603.06054), [Gou et al. 2025](https://arxiv.org/html/2508.15360v1), [Rahman 2026](https://arxiv.org/html/2607.12304v3)

### Inferences (proposed diagnostics, ordered by cost; these are suggestions, not published results)
1. **Three-stage linear probe on twins (separates H1 from H2).** Train ridge/logistic probes for ball velocity direction and final out-of-view quadrant on (a) vision-tower block outputs, including early blocks around index 7, (b) merger/projector output, (c) LLM residual stream per layer, at earlier-frame token positions, last-frame token positions and the final text token. Use the Theodoridis classification: if (a) or (b) is at chance, the failure is perceptual and no decoder-side fix can work; if decodable at earlier-frame positions but not at last-frame or final-token positions, the failure is in cross-frame transfer; if decodable at the final token while the answer is wrong, it is a read-out failure. Region-pool over the ball's patches rather than average-pooling, since spatial concepts were invisible to average-pooled probes.
2. **Twin activation patching (localises where twin identity is lost).** Twins share a last frame, so run both and patch residual activations from twin A into twin B at (layer, token group). Find the earliest layer/position where patching earlier-frame tokens flips the answer; if no patch below the final token flips it, the trajectory signal never reaches a causal position. Rahman's result that order information is fixed by about layers 0-3 in Qwen3-VL suggests starting with early layers.
3. **Attention knockouts with the published layer windows.** (i) Block last-frame-to-earlier-frame attention in layers 1-16; (ii) block query-to-earlier-frame attention; (iii) block query-to-last-frame attention. If (i) and (ii) leave the fine-tuned model's answers unchanged, it is causally single-frame, as the image-only LLaVA-NeXT was. If (iii) alone destroys accuracy, the last frame is the sole read-out route.
4. **RoPE sanity check (tests H3 directly).** Reverse or shuffle only the temporal rotary index with pixels fixed, and separately reverse frames with indices fixed, on the project's own data. If answers change only in the second case, the rotary-shift attempt was acting on a channel the model does not use.
5. **Text-to-token patching for the memory channel (tests H4).** Run the model with the state written as a sentence, record residual activations at the state tokens in the layers where the answer becomes decodable, and patch them into the memory-token positions of a run without the sentence. If the answer becomes correct, the downstream computation is intact and the injected embeddings are simply not mapped into that representation; then supervise the memory tokens to match those activations (distillation) instead of relying on the answer loss. Also compare norm and cosine statistics of memory tokens against text-token embeddings, and measure attention mass received by memory tokens versus the sentence.
6. **Leak audit for twin training.** Train a small classifier on a single frame, and on the last frame only, to predict data source and twin label; then prune or regenerate until it is at chance (the "train on the test set" procedure).
7. **Intermediate-text test (cheap behavioural check before any architecture change).** Fine-tune with the target "trajectory as text (time, x, y) then answer", as in VET-Bench, on twins. If this works while direct answering does not, the bottleneck is the lack of an intermediate computation step and readable state, and the in-model fix should aim to internalise that step (e.g. distil the text trajectory into hidden states) rather than add a new channel from scratch.

### Gaps
- None of these diagnostics has been published for out-of-view object localisation in Qwen2.5-VL-3B; expected outcomes above are predictions.
- Layer indices quoted from other models (LLaVA-NeXT-7B, Qwen2-VL, Qwen2.5-VL-7B) will not map one-to-one onto the 3B model's 36-layer decoder (layer count not verified in this pass); sweep rather than assume.
