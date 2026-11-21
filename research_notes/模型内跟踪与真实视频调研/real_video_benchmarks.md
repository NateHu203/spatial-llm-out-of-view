# Real-video datasets and benchmarks for out-of-view / occluded object reasoning (excluding Beyond3D)

Notes compiled 2026-11-19. Every fact below was read on the linked source page during this session unless tagged **[unverified]** (from memory or a search snippet only) or **[uncertain]**. "Not stated" means the source page I read did not say; it does not mean the thing is absent. Beyond3D (arXiv 2609.34630, "Long Time No See", 2026-09-28) is deliberately not covered; it is the only benchmark my searches found that is explicitly about out-of-sight *moved* objects in real egocentric video — [arXiv 2609.34630](https://arxiv.org/abs/2609.34630).

## Key Question 1: Which benchmarks test VLMs/MLLMs on spatial reasoning over real video, especially with out-of-view objects, egocentric spatial memory or moving objects?

### Takeaway
Many real-video spatial benchmarks exist, but almost all either use static scenes (VSI-Bench, VSI-Super, OST-Bench, OpenEQA, SPAR-Bench) or test moving objects while they are visible (VLM4D, DSI-Bench, STI-Bench, STKit-Bench, EOC-Bench); apart from Beyond3D, I found no real-video VLM benchmark whose questions ask where a moving object is after it has left view. The closest in spirit are TOC-Bench (disappearance/reappearance, shortcut-filtered), DSI-Bench (flip/reverse symmetric groups), OST-Bench (static objects no longer visible) and the Perception Test cups game (hidden object under occluder).

### Cited Findings

**Static-scene, real video (camera moves, objects do not)**
- VSI-Bench (2024-12-18): "over 5,000" QA pairs on real indoor scan videos; the HF card gives 5,130 annotations, 288 videos, sources = validation sets of ScanNet, ScanNet++, ARKitScenes, license apache-2.0, 5.73 GB, evaluation through lmms-eval (`vsibench`, `vsibench_debiased`); multiple choice scored by accuracy, numerical by Mean Relative Accuracy. A debiased variant keeps 2,362 questions and the card says to report both separately — [arXiv 2412.14171](https://arxiv.org/abs/2412.14171); [HF card](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)
- VSI-Super / Cambrian-S (2025-11-06): two tasks. VSR = long-horizon recall of the order/locations of unusual objects that were inserted into four frames of real room-tour videos with an image-editing model; VSC = counting objects across concatenated VSI-Bench room-tour clips. Lengths 10–240 min (VSR) and 10–120 min (VSC). Gemini-2.5-Flash gets 41.5 (VSR) and 10.9 (VSC) at 60 min and runs out of context at 120 min. Data at hf.co/collections/nyu-visionx/vsi-super; dataset license not stated in the paper — [arXiv 2511.04670](https://arxiv.org/abs/2511.04670)
- OST-Bench (2025-07-10): online setting, about 1.4k scenes and 10k QA from ScanNet, Matterport3D, ARKitScenes; the paper states scenes are static and "the agent is the only dynamic element". Categories: agent state, agent visible info, agent-object spatial relationship; some questions target objects "invisible in the current round". Overall: Qwen2.5-VL-7B 41.2, 32B 46.9, 72B 45.6, human 83.5, chance 36.9. Object annotations come from EmbodiedScan/MMScan (9-DoF boxes). Code/data: github.com/InternRobotics/OST-Bench — [arXiv 2507.07984](https://arxiv.org/html/2507.07984v2)
- OpenEQA (CVPR 2024): over 1,600 human-written open-vocabulary questions from over 180 real-world environments, scored with LLM-Match; episodic-memory and active settings; MIT license; QA in `data/open-eqa-v0.json`, episode histories downloaded separately — [project page](https://open-eqa.github.io/); [GitHub](https://github.com/facebookresearch/open-eqa)
- SPAR-Bench (2025-03-29): 7,207 manually verified QA over 20 spatial tasks, single-view and multi-view inputs, derived from SPAR-7M (sample rows show ScanNet), 1.84 GB; license not listed on the card — [arXiv 2503.22976](https://arxiv.org/abs/2503.22976); [HF card](https://huggingface.co/datasets/jasonzhango/SPAR-Bench)
- MMSI-Bench (2025-05-29): 1,000 hand-written multiple-choice questions over multi-image inputs (not video); best open model ~30%, humans 97% — [arXiv 2505.23764](https://arxiv.org/abs/2505.23764)
- ODI-Bench (2025-10-13): 2,000 omnidirectional *images*, 4,000+ QA; not video — [arXiv 2510.11549](https://arxiv.org/abs/2510.11549)
- OpenView / OpenView-Bench (2025-12-21): "out-of-view VQA" — reasoning about content beyond the frame of a perspective crop taken from a panorama; 1,155 verified five-option questions; static content, built from frames, not object motion. Overall joint accuracy: Qwen2.5-VL-3B 6.75%, 7B 42.25%, 32B 56.80%, 72B 56.27%. Only annotations/projection metadata are released; raw panoramas must be fetched from the original providers — [arXiv 2512.18563](https://arxiv.org/html/2512.18563v2)

**Real video with moving objects (mostly while visible)**
- VLM4D (2025-08-04): 1,000 videos / 1,816 QA; 600 real videos (1,371 QA) from Ego4D, DAVIS, YouTube-VOS and 400 synthetic (Cosmos). Multiple choice on translational motion, rotational motion, counting, false positives. Overall: Qwen2.5-VL-7B 43.8, Qwen2.5-VL-72B 53.0, Qwen2-VL-7B 36.3, VideoLLaMA3-2B 43.7, human 98.8, random 24.1. Release of tracks/poses not stated — [arXiv 2508.02095](https://arxiv.org/html/2508.02095v2)
- DSI-Bench (2025-10-21): 943 mostly real videos (CameraBench, Kinetics-700, LLaVA-178K, online; SynFMC is synthetic), 1,700+ four-option questions on observer and object motion. Each video exists in four variants (original, horizontal flip, time-reversed, reversed+flipped) with answers adjusted by rule; scoring is sample-wise and group-wise. Qwen2.5-VL-32B: 36.73% sample-wise / 16.40% group-wise; 72B: 39.61% / 15.43%; random 25.00% / 0.05%. The paper said data and code would be released after review (availability not checked) — [arXiv 2510.18873](https://arxiv.org/html/2510.18873v1)
- STI-Bench (2025-03-31): 300+ real videos, 2,064 five-option QA from Waymo, ScanNet and Omni6DPose; static tasks (dimension, spatial relation, 3D grounding) and dynamic tasks (displacement and path length, speed and acceleration, ego orientation, trajectory description, pose). Ground truth comes from per-frame camera intrinsics/extrinsics and 3D boxes. Qwen2.5-VL-72B 40.7, VideoLLaMA3-7B 35.2, MiniCPM-V-2.6 26.9, random 20 — [arXiv 2503.23765](https://arxiv.org/html/2503.23765v6)
- ST-VLM / STKit-Bench (2025-03-25): 1,400 QA (200 per task × 7 tasks: traveled distance, speed, movement direction, direction timestamp, and three two-object comparisons) on real video from NuPlan (74.8%), nuScenes (12.5%), Argoverse2 (5.6%), Ego-Exo4D (7.1%); ground truth from LiDAR/SLAM 3D annotations; free-form answers parsed by GPT-4. Average: Qwen2.5-VL-7B 22.5, LLaVA-OneVision-7B 27.4, InternVL2.5-8B 26.0, GPT-4o 26.8, fine-tuned ST-VLM-7B 59.8 — [arXiv 2503.19355](https://arxiv.org/html/2503.19355v2)
- EOC-Bench (2025-06-05): 656 real egocentric videos (EPIC-KITCHENS, Ego4D, Charades-ego, MECCANO, 110 self-recorded), 3,277 QA in Past (1,422; includes "Object Location Retrospection"), Present (1,348) and Future (507; includes "Trajectory and Motion Prediction"). Formats: true/false, single-choice, multiple-choice, open-ended; objects referred to by point/box/mask prompts. Overall: Qwen2.5-VL-3B 38.17, 7B 43.13, 72B 49.87, VideoLLaMA3-2B 38.41, random 24.87, human 94.63. Data: huggingface.co/datasets/CircleRadon/EOC-Bench — [arXiv 2506.05287](https://arxiv.org/html/2506.05287v1)
- MMSI-Video-Bench (2025-12-11): 1,106 human-written questions on 1,278 real clips from 25 datasets (incl. ScanNet, ScanNet++, ARKitScenes, Waymo, nuScenes, OVIS, LaSOT, Ego4D, EPIC-KITCHENS, DROID) plus 140 in-house videos; categories include instance motion, camera motion, prediction and "Memory Update"; 4–6 options. QwenVL2.5-7B 29.7 (uniform-50 frames), 72B 32.7, random 24.1, human 96.4. Data: huggingface.co/datasets/rbler/MMSI-Video-Bench — [arXiv 2512.10863](https://arxiv.org/html/2512.10863v1)
- Ego3D-Bench (2025-09-08): 8,600+ QA from nuScenes, Waymo, Argoverse 1 validation sets; input is multi-view camera images (not described as video); "motion reasoning" is hypothetical-movement yes/no. Average: Qwen2.5-VL-3B 39.1 (44.4 with their cognitive-map add-on), 7B 41.1 (54.3), chance 37.5, human 85.3. Data: huggingface.co/datasets/vbdai/Ego3D-Bench — [arXiv 2509.06266](https://arxiv.org/html/2509.06266v2)
- SpaceVista (2025-10-10): SpaceVista-1M (about 1M QA, 38K video scenes, 19 task types, 5 spatial scales) plus a manually built benchmark; I verified only the abstract — [arXiv 2510.09606](https://arxiv.org/abs/2510.09606)
- EgoSchema (2023-08-17): 5,000+ five-option questions on three-minute Ego4D clips, 250+ hours; tests long-form temporal understanding, not object position — [arXiv 2308.09126](https://arxiv.org/abs/2308.09126)

**Object permanence / disappearance / hidden objects**
- TOC-Bench (2026-05-11): 2,323 human-verified QA on 1,951 real videos (Charades, Perception Test, OVIS, MOSE; mean 28.8 s). Each queried object is linked to a per-frame trajectory and event timeline. Ten dimensions include reappearance identity, reappearance/disappearance mechanism and relative spatial change. A three-layer filter (text-only, single-frame, frame-shuffled; filter model Qwen3-VL-8B) removed 60.7% of candidates. Overall: Qwen2.5-VL-7B 27.0, Qwen2.5-VL-72B 34.0, InternVL3-8B 30.1, random 30.4, human 89.3. Code: github.com/cjzcjz666/toc_bench; whether tracks are released is not stated — [arXiv 2605.09904](https://arxiv.org/html/2605.09904v2)
- Perception Test (2023-05-23): real videos with skill areas Memory, Abstraction, Physics, Semantics and reasoning types descriptive, explanatory, predictive, counterfactual; 38,060 multiple-choice questions, 189,940 object tracks over 11,609 videos, 8,647 point tracks; metadata flags `is_cup_game`, `is_occluder`, `is_masked`, `is_camera_moving`. Annotations CC-BY 4.0; videos 26.5 GB (train), 70.2 GB (valid), 41.8 GB (test); a 214.9 MB sample set exists — [arXiv 2305.13786](https://arxiv.org/abs/2305.13786); [GitHub](https://github.com/google-deepmind/perception_test)
- VET-Bench (2026-03-09): shell-game tracking of visually identical objects; the abstract calls it a *synthetic* testbed; state-of-the-art VLMs are at or near chance — [arXiv 2603.08436](https://arxiv.org/abs/2603.08436). A search summary of the same paper says that a filtered 107-item subset of the Perception Test cups game (appearance shortcuts removed) drops frontier models to about 30.77%, chance being 1/3 **[unverified: search snippet, not read in the paper]** — [arXiv 2603.08436](https://arxiv.org/pdf/2603.08436)

**Found but synthetic, or not a VLM test (listed so they are not mistaken for candidates)**
- Disjoint-3DQA (2025-05-30, EMNLP 2025): questions about object pairs never co-visible; built on Aria *Synthetic* Environments, static objects, 5,399 open-ended QA; text trajectories or BEV images gave small gains (Qwen2.5-72B 64.31 → 66.21 / 66.64) while full 3D metadata took GPT-4o to 83.2; data not yet released at paper time — [arXiv 2505.24257](https://arxiv.org/html/2505.24257v1)
- SpaMEM (2026-04-24): spawn/place/remove belief updating in 1,000 procedurally generated houses (synthetic) — [arXiv 2604.22409](https://arxiv.org/abs/2604.22409)
- GST-Bench (2026-08-06): synthetically generated video — [arXiv 2608.05747](https://arxiv.org/abs/2608.05747)
- REM (2025-11-30): multi-frame embodied trajectories probing object permanence and cognitive maps; real vs synthetic not confirmed **[uncertain]** — [arXiv 2512.00736](https://arxiv.org/abs/2512.00736)
- MemoBench (2026-06-25): 360 clips (196 Unreal Engine, 164 real indoor recordings) built around "target visible → camera pans away → camera returns"; real clips have calibrated intrinsics and MapAnything-estimated poses, plus annotated disappear/reappear frame indices. It evaluates video *generation* models; Gemini-3.1-Pro is only the judge — [arXiv 2606.27537](https://arxiv.org/html/2606.27537v6)
- PersistBench (2026-09-17): 2,000 paired sequences cut from real 360° YouTube videos (360-1M); the 360° footage is "omniscient" ground truth for objects that leave a perspective crop; static and dynamic subsets. Subjects are 4D reconstruction / camera-controlled video models, not VLMs — [arXiv 2609.20819](https://arxiv.org/html/2609.20819v2)
- WROP (2026-09-23): Blender-generated object-permanence tasks for video generation models — [arXiv 2609.28654](https://arxiv.org/abs/2609.28654)
- IntPhys 2 (2025-06-11): permanence, immutability, continuity, solidity in virtual environments; most models at chance (50%) — [arXiv 2506.09849](https://arxiv.org/abs/2506.09849)
- A 2026-10-21 paper on Video-LLM counting of screen crossings uses synthetic testbeds and argues the count signal is linearly decodable but not read out — [arXiv 2611.09949](https://arxiv.org/abs/2611.09949)
- Survey: "Spatial Reasoning in Multimodal Large Language Models: A Survey of Tasks, Benchmarks and Methods" (2025-11-14) — [arXiv 2511.15722](https://arxiv.org/abs/2511.15722)

### Inferences
- The project's exact question type (relative position of a moving object after it left view, with known ego-motion) has no ready-made real-video benchmark other than Beyond3D; existing benchmarks cover the two halves separately (static out-of-view memory: OST-Bench, VSI-Bench; visible moving objects: VLM4D, DSI-Bench, STI-Bench).
- Published small-model scores sit close to chance on the dynamic benchmarks (TOC-Bench 7B below random; Ego3D-Bench 3B 39.1 vs chance 37.5; OST-Bench 7B 41.2 vs 36.9), so a 3B baseline will likely be near floor and the informative comparison is baseline vs the project's position-sentence injection.
- Disjoint-3DQA and Ego3D-Bench are precedents for the project's method (feeding computed trajectories / cognitive maps as text); both report that textual 3D cues help, and Disjoint-3DQA shows a much larger gain from oracle 3D than from trajectories alone.

### Gaps
- Qwen2.5-VL-3B scores were not found for VLM4D, DSI-Bench, STI-Bench, OST-Bench, MMSI-Video-Bench, TOC-Bench or STKit-Bench (only 7B or larger are reported in the pages read).
- Sizes of VSI-Super and whether Cambrian-S numbers on it are in the paper body: not in the portion read.
- Dataset licenses are not stated in the papers for VLM4D, DSI-Bench, STI-Bench, TOC-Bench, EOC-Bench, MMSI-Video-Bench, Ego3D-Bench; the repositories were not opened.
- SpaceVista-Bench composition and scores were not verified. MMSI-Video-Bench per-source counts are not given.
- I did not confirm whether DSI-Bench data has actually been released.

## Key Question 2: Which tracking / intuitive-physics video datasets have real footage with ground-truth positions or tracks through occlusion or leaving the frame?

### Takeaway
Real datasets with metric 3D object positions *and* camera pose exist (Aria Digital Twin, HOT3D, TAPVid-3D, nuScenes, Waymo), and they are the only ones that let the project measure position error on real video; 2D-only datasets (TAO-Amodal, EgoTracks, LaSOT, EgoPoints, Perception Test, OVIS) mark or box objects through occlusion but mostly only flag, not locate, out-of-frame periods.

### Cited Findings

**Real, with 3D object positions and camera pose**
- Aria Digital Twin (2023-06-10): 200 real egocentric sequences (about 400 min) in two indoor scenes, 398 objects (324 stationary, 74 dynamic); ground truth includes continuous 6DoF device pose, object 6DoF pose, 2D boxes, instance segmentation, depth. Access requires an email and agreeing to a dataset license agreement (terms not read) — [arXiv 2306.06362](https://arxiv.org/abs/2306.06362); [project page](https://www.projectaria.com/datasets/adt/)
- HOT3D (2024-06-13): 833+ minutes, 3.7M+ images, 19 subjects, 33 rigid objects, recorded with Aria and Quest 3; mocap ground truth for 3D poses of objects, hands and cameras; object and hand visibility masks provided; test sequences have no ground truth; dataset under a separate "HOT3D license agreement"; download links expire after 14 days — [arXiv 2406.09598](https://arxiv.org/abs/2406.09598); [GitHub](https://github.com/facebookresearch/hot3d)
- TAPVid-3D (2024-07-08): 4,000+ real videos (about 4,500 npz files) from Aria Digital Twin, Waymo Open (DriveTrack) and Panoptic Studio; each file has `tracks_xyz` (3D point trajectories in metres), `visibility`, `queries_xyt`, `intrinsics`, and `extrinsics_w2c` for the moving-camera sources (Waymo, ADT; Panoptic Studio camera is static). Annotations CC-BY 4.0, but users must accept each source dataset's licence and regenerate from source; a `minival` split exists — [arXiv 2407.05921](https://arxiv.org/abs/2407.05921); [GitHub](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d)
- nuScenes (2019-03-26): 1000 scenes of 20 s, 6 cameras with 360° coverage, fully annotated with 3D boxes for 23 classes and 8 attributes; the devkit README notes boxes are annotated even when objects are temporarily occluded and lists a `mini` split; download needs an account and agreeing to the nuScenes Terms of Use — [arXiv 1903.11027](https://arxiv.org/abs/1903.11027); [devkit](https://github.com/nutonomy/nuscenes-devkit)
- Waymo Open Dataset (2019-12-10): 1150 scenes of 20 s with synchronized, calibrated LiDAR and camera — [arXiv 1912.04838](https://arxiv.org/abs/1912.04838)
- Ego-Exo4D (2023-11-30): 1,286 hours of synchronized ego and exo video with 3D point clouds, camera poses, IMU, gaze; no per-object 3D tracks mentioned in the abstract — [arXiv 2311.18259](https://arxiv.org/abs/2311.18259)

**Real, 2D tracks through occlusion / out of frame**
- TAO-Amodal (2023-12-19): amodal and modal boxes for visible, partially or fully occluded objects, "including those that are partially out of the camera frame"; 833 categories; built on TAO (2,907 videos averaging about half a minute). HF card: MIT licence, gated, 376 GB, fields `amodal_bbox`, `visibility` (0–1), `amodal_is_uncertain`; frames from ArgoVerse, BDD, Charades, HACS, LaSOT, YFCC100M — [arXiv 2312.12433](https://arxiv.org/abs/2312.12433); [HF card](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal); [TAO arXiv 2005.10356](https://arxiv.org/abs/2005.10356)
- EgoTracks (2023-01-09): long-term egocentric single-object tracking from Ego4D, designed around objects frequently exiting the frame and needing re-detection; 5.9k videos, 22.42k tracks; 2D boxes only (no 3D or pose fields); access via the Ego4D CLI (the site says access has moved to "Lab22") — [arXiv 2301.03213](https://arxiv.org/abs/2301.03213); [docs](https://ego4d-data.org/docs/data/egotracks/)
- Ego4D episodic memory (2021-10-13): visual queries answer "where is this object" with a 2D box and optionally a 3D displacement vector from the current camera to the object's 3D box; about 74K queries over 800 hours in total across query types — [docs](https://ego4d-data.org/docs/benchmarks/episodic-memory/); [arXiv 2110.07058](https://arxiv.org/abs/2110.07058)
- EgoPoints (2024-12-05): 517 EPIC-KITCHENS-100 sequences, 4,703 point tracks; 875 tracks have an out-of-view point at an evaluation frame and 593 need re-identification; out-of-view points are flagged, not located — [arXiv 2412.04592](https://arxiv.org/html/2412.04592v1)
- LaSOT (2018-09-20): 1,400 sequences, 3.5M+ frames, per-frame boxes, average length over 2,500 frames, targets "may disappear and re-appear again in the view" — [arXiv 1809.07845](https://arxiv.org/abs/1809.07845)
- EPIC-KITCHENS VISOR (2022-09-26): 272K manual masks of 257 classes, 9.9M interpolated masks, 36 hours of 179 untrimmed videos — [arXiv 2209.13064](https://arxiv.org/abs/2209.13064)
- OVIS (2021-02-02): 296k instance masks, 25 categories, heavy occlusion — [arXiv 2102.01558](https://arxiv.org/abs/2102.01558)
- TAP-Vid (2022-11-07): real videos with human-annotated 2D point tracks plus synthetic videos with perfect tracks — [arXiv 2211.03726](https://arxiv.org/abs/2211.03726)
- TAPVid-360 (2025-11-26): task = predict the 3D *direction* to queried points even when far outside a narrow field of view; built from real 360° YouTube video (360-1M) by rendering perspective crops; TAPVid360-10k has 4,772 clips, 10,000 dynamic objects (person, dog, car, bike, bird...), 36.28M in-frame and 45.64M out-of-frame points; ground truth is pseudo-GT from SAM2 + CoTracker3 on the 360° video; metrics reported separately for in-frame and out-of-frame; no VLMs evaluated — [arXiv 2511.21946](https://arxiv.org/html/2511.21946v2)

**Intuitive physics**
- InfLevel (TMLR 2022): violation-of-expectation benchmark with paired plausible/implausible videos on continuity, solidity, gravity; InfLevel-Lab is real footage filmed in an infant-cognition lab, InfLevel-Sim is AI2-THOR; Apache-2.0; evaluation only (no training or probing on it) — [GitHub](https://github.com/allenai/inflevel)
- Physics-IQ (2025-01-14) compares generated video with real videos of physical experiments; it targets video generation models; a 2026-06-17 audit ("Physics-IQ Verified") exists — [arXiv 2501.09038](https://arxiv.org/abs/2501.09038); [arXiv 2606.18943](https://arxiv.org/abs/2606.18943)
- Physion (2021-06-15) is simulation — [arXiv 2106.08261](https://arxiv.org/abs/2106.08261). PointOdyssey (2023-07-27) is synthetic: 104 videos averaging 2,000 frames — [arXiv 2307.15055](https://arxiv.org/abs/2307.15055)

### Inferences
- For position-error evaluation on real video with a moving camera of known pose, the practical shortlist is ADT (indoor, few dynamic objects, exact 6DoF), TAPVid-3D's ADT/DriveTrack npz files (already packaged as 3D tracks + extrinsics + visibility) and nuScenes (outdoor, many independently moving objects, boxes kept through occlusion).
- nuScenes' six surround cameras mean an object that leaves the front camera is often still annotated and visible in a side camera, which gives real ground truth for "where is it now" after it left the queried view; the same holds for 360°-derived data (TAPVid-360, PersistBench).
- TAO-Amodal's size (376 GB, gated) makes it a poor fit for a few-week project despite its on-topic annotations.

### Gaps
- nuScenes keyframe annotation rate, mini-split size, per-box visibility levels and the exact licence terms were not readable (nuscenes.org rendered empty); from memory these are 2 Hz, 10 scenes, 4 visibility bins and a non-commercial licence **[unverified]**.
- Waymo Open licence terms and whether TAPVid-3D keeps 3D positions for points that leave the frame (vs. only marking them not visible) were not confirmed.
- ADT and HOT3D licence text (commercial use, redistribution of derived clips) not read; ADT frame rate, resolution and download size not stated on the page.
- ScanNet++ and ARKitScenes were not separately researched; they appear above only as static-scene sources of VSI-Bench/OST-Bench.
- InfLevel-Lab video count and download size not stated on the repository page; I found no published VLM results on it.
- Ego4D VQ3D sizes and ground-truth details were not on the page read.

## Key Question 3: Per-item attributes (real/synthetic, scene and camera motion, pose, positions, format, size, access, small-model scores)

### Takeaway
The table condenses the verified attributes; the column "GT positions" is what decides whether position error (not only answer accuracy) can be checked. Blank or "n.s." means not stated on the pages read.

### Cited Findings

Sources for every row are the links given under Key Questions 1 and 2 for the same item.

| Item (date) | Real? | Objects move? | Camera | Camera pose given | Per-frame object positions | QA format | Size | Access / licence | Small-model scores |
|---|---|---|---|---|---|---|---|---|---|
| VSI-Bench (2024-12) | real | no | moving | in source scans (not in QA files) | 3D boxes in source datasets | MC + numeric | 5,130 QA, 288 videos, 5.73 GB | HF, apache-2.0, lmms-eval | Qwen2.5-VL-3B ≈ 29.0 macro avg per Spatial-MLLM **[search snippet]**; other papers report about 20–34 depending on protocol **[search snippet]** — [arXiv 2505.23747](https://arxiv.org/pdf/2505.23747) |
| VSI-Super (2025-11) | real, edited | no (inserted objects) | moving | n.s. | n.s. | recall order / count | 10–240 min videos | HF collection, licence n.s. | Gemini-2.5-Flash 41.5 / 10.9 at 60 min |
| OST-Bench (2025-07) | real | no | moving | in sources | 9-DoF boxes via EmbodiedScan | judgment, counting, estimation, temporal | 1.4k scenes, 10k QA | GitHub | Qwen2.5-VL-7B 41.2; chance 36.9 |
| OpenEQA (2024) | real environments | n.s. | moving | n.s. | no | open-ended, LLM-scored | 1,600+ Q, 180+ envs | MIT | n.s. |
| SPAR-Bench (2025-03) | real scans | no (n.s.) | multi-view | n.s. | n.s. | 20 tasks | 7,207 QA, 1.84 GB | HF, licence n.s. | n.s. |
| VLM4D (2025-08) | 600 real + 400 synthetic | yes | both ego and exo | n.s. | n.s. | MC | 1,816 QA | project page | Qwen2.5-VL-7B 43.8; random 24.1 |
| DSI-Bench (2025-10) | mostly real | yes | fixed and moving | no (human motion labels) | no | 4-option MC, 4 flip/reverse variants | 943 videos, 1,700+ QA | release promised | Qwen2.5-VL-32B 36.73 / 16.40 group |
| STI-Bench (2025-03) | real | yes (Waymo) and static | moving | yes in sources | 3D boxes in sources | 5-option MC | 300+ videos, 2,064 QA | project page | Qwen2.5-VL-72B 40.7; MiniCPM-V-2.6 26.9 |
| STKit-Bench (2025-03) | real | yes | moving | yes in sources | LiDAR/SLAM 3D | free-form, GPT-parsed | 1,400 QA | project page | Qwen2.5-VL-7B 22.5 |
| EOC-Bench (2025-06) | real egocentric | yes (handled) | moving | no | no (visual prompts only) | T/F, MC, open | 656 videos, 3,277 QA | HF | Qwen2.5-VL-3B 38.17; random 24.87 |
| MMSI-Video-Bench (2025-12) | real | mixed | mixed | n.s. | no | 4–6 option MC | 1,106 Q, 1,278 clips | HF | QwenVL2.5-7B 29.7; random 24.1 |
| Ego3D-Bench (2025-09) | real driving | hypothetical only | multi-view images | n.s. | n.s. | MC + metres | 8,600+ QA | HF | Qwen2.5-VL-3B 39.1; chance 37.5 |
| TOC-Bench (2026-05) | real | yes | mixed | no | per-frame trajectories used; release n.s. | 4-way MC, pairs, count, order | 1,951 videos, 2,323 QA | GitHub | Qwen2.5-VL-7B 27.0; random 30.4 |
| Perception Test (2023-05) | real | yes | flag per video | no | 2D object + point tracks | MC + grounded | 38,060 MC Q; valid 70.2 GB | CC-BY 4.0 | n.s. in pages read |
| OpenView-Bench (2025-12) | real panoramas | no | n/a | projection metadata | no | 5-option MC + rationale | 1,155 Q | GitHub; raw panoramas not included | Qwen2.5-VL-3B 6.75 joint |
| Aria Digital Twin (2023-06) | real | 74 dynamic objects | moving (head) | 6DoF | 6DoF + 2D boxes + masks + depth | none | 200 seq, ~400 min | email + licence agreement | – |
| HOT3D (2024-06) | real | 33 hand-held objects | moving (head) | yes | 3D pose + visibility | none | 833+ min | licence agreement; test GT withheld | – |
| TAPVid-3D (2024-07) | real | yes | moving (ADT, Waymo), static (PStudio) | `extrinsics_w2c` | 3D point tracks + visibility | none | ~4,500 videos | CC-BY 4.0 annotations + source licences | – |
| nuScenes (2019-03) | real | yes | moving vehicle, 6 cameras | yes (ego pose) | 3D boxes, kept through occlusion | none | 1000 × 20 s | account + Terms of Use | – |
| Waymo Open (2019-12) | real | yes | moving vehicle | calibrated | 3D annotations | none | 1150 × 20 s | n.s. | – |
| TAO-Amodal (2023-12) | real | yes | mixed | no | 2D amodal boxes + visibility | none | 376 GB | MIT, gated | – |
| EgoTracks (2023-01) | real | yes | moving (head) | no | 2D boxes when visible | none | 5.9k videos, 22.42k tracks | Ego4D licence | – |
| EgoPoints (2024-12) | real | yes | moving (head) | no | 2D points; OOV flagged | none | 517 seq, 4,703 tracks | project page | – |
| TAPVid-360 (2025-11) | real 360° | yes | synthetic crops of real 360° | by construction | direction vectors incl. out-of-frame (pseudo-GT) | none | 4,772 clips | project page, licence n.s. | – |
| InfLevel-Lab (2022) | real | yes | n.s. | no | no | paired plausible/implausible | n.s. | Apache-2.0, eval only | – |

### Inferences
- Only Qwen2.5-VL-3B numbers verified on a paper page: EOC-Bench 38.17, Ego3D-Bench 39.1, OpenView-Bench 6.75. The VSI-Bench 3B figure is from a search summary and varies by protocol, so the team should re-run it rather than cite it.
- None of the QA benchmarks ships per-frame object positions with the questions; position-error checks require going back to the source dataset (ScanNet/Waymo/nuScenes/ADT).

### Gaps
- Frame counts per question, typical clip length and resolution were not collected for most benchmarks, so GPU-time estimates are not sourced.
- "n.s." cells are genuine unknowns from the pages read, not confirmed absences.

## Key Question 4: Which allow "twin"-style counterfactual pairs, or questions that cannot be answered from the last frame alone?

### Takeaway
No real-video benchmark found uses exact last-frame-identical twins; the nearest published designs are DSI-Bench's four-variant flip/reverse groups with group-wise scoring, TOC-Bench's single-frame and shuffled-frame filters, InfLevel's plausible/implausible pairs, and PersistBench's input/reference pairs from 360° video. Exact twins on real video would have to be constructed by the team from datasets with ground-truth positions.

### Cited Findings
- DSI-Bench: every video is evaluated in four variants (original, horizontal flip, time-reversed, reversed + flipped) with rule-adjusted answers; group-wise accuracy for Qwen2.5-VL-72B is 15.43% against 39.61% sample-wise, random group-wise 0.05% — [arXiv 2510.18873](https://arxiv.org/html/2510.18873v1)
- TOC-Bench: candidates are removed if answerable text-only, from a single frame, or from shuffled frames; no last-frame-specific check is described — [arXiv 2605.09904](https://arxiv.org/html/2605.09904v2)
- Vinoground (2024-10-03): 1000 short natural video–caption pairs described as temporal counterfactuals — [arXiv 2410.02763](https://arxiv.org/abs/2410.02763)
- InfLevel: each implausible event is paired with a plausible one; the model should score the plausible video higher — [GitHub](https://github.com/allenai/inflevel)
- PersistBench: each pair has an input sequence where the object leaves view and a reference sequence (another crop of the same 360° video) where it stays visible; both are defined by camera poses — [arXiv 2609.20819](https://arxiv.org/html/2609.20819v2)
- Perception Test: test videos were cut where the ending would reveal the answer (e.g. where the hidden object ends up in the cups game), and cup-game videos carry occluder flags — [GitHub](https://github.com/google-deepmind/perception_test)
- OST-Bench: some questions require that the target object is invisible in the current round — [arXiv 2507.07984](https://arxiv.org/html/2507.07984v2)
- TAPVid-360 and PersistBench both rely on the fact that arbitrary perspective camera paths can be rendered from one real 360° video while the true object direction stays known — [arXiv 2511.21946](https://arxiv.org/html/2511.21946v2); [arXiv 2609.20819](https://arxiv.org/html/2609.20819v2)

### Inferences
(All points below are my proposals, not published procedures.)
- Flip/reverse twins (DSI-Bench style) can be applied to any real clip at zero annotation cost: a time-reversed clip swaps "moving left" with "moving right", and group-wise scoring penalises a model that ignores motion. Reversal does not keep the last frame identical, so it is weaker than the project's twins.
- Near-twins from fixed-camera footage: when the camera is static and the object has fully left the frame, the last frame shows only background, so two clips of the same scene in which the object exited left vs right have near-identical last frames. Candidates: Panoptic Studio part of TAPVid-3D (static camera), fixed-camera Perception Test videos (`is_camera_moving` flag), or self-recorded clips.
- Near-twins from 360° video: render two perspective crops that end at the same view direction over a static background but are taken at two different times, with the object out of the crop at different true directions. This follows the TAPVid-360 / PersistBench construction; ground truth is pseudo-GT from trackers.
- Truncation twins from 3D-annotated data (nuScenes, ADT, TAPVid-3D): cut the clip after the object leaves the queried camera, then ask the two-choice question about the time of the last frame or +2 s; the answer is computed from 3D boxes/poses, and a "last frame only" run gives the shortcut baseline. Last frames are not identical across items, so the control is a single-frame baseline rather than a guaranteed 0.
- InfLevel-Lab and the Perception Test cups game test permanence under occlusion with a mostly fixed camera; they do not exercise ego-motion compensation, which is the core of the project's system.

### Gaps
- I found no published real-video dataset with pairs whose final frames are pixel-identical while earlier motion differs.
- Whether DSI-Bench's variants and TOC-Bench's tracks are downloadable today was not checked.
- The share of Perception Test videos with a fixed camera, and the number of cup-game videos, are not stated on the repository page.

## Key Question 5: Most practical 2–3 choices for this project

### Takeaway
(Recommendation = inference from the findings above.) Cheapest first step: run Qwen2.5-VL-3B with and without the position sentence on an existing real-video QA benchmark that already has a published 3B or 7B baseline and small download (VSI-Bench for static out-of-view memory; EOC-Bench or TOC-Bench for moving objects). Strongest evidence: build a small two-choice set from a real dataset with ego pose and 3D object tracks (nuScenes mini or TAPVid-3D/ADT), where the project's own pipeline (per-frame positions → extrapolation → sentence) can be run end to end and position error measured.

### Cited Findings
- VSI-Bench is 5.73 GB, apache-2.0, integrated in lmms-eval, and has a debiased subset; scenes are static — [HF card](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)
- EOC-Bench has a verified Qwen2.5-VL-3B score (38.17 vs random 24.87) on real egocentric video with moved objects and "Object Location Retrospection" and "Trajectory and Motion Prediction" categories — [arXiv 2506.05287](https://arxiv.org/html/2506.05287v1)
- TOC-Bench is filtered so single-frame and shuffled-frame inputs fail, and small models score at or below its 30.4 random level — [arXiv 2605.09904](https://arxiv.org/html/2605.09904v2)
- TAPVid-3D ships per-video 3D tracks, visibility, intrinsics and world-to-camera extrinsics and has a `minival` split; it needs the source-data licences — [GitHub](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d)
- nuScenes gives 20 s scenes with 3D boxes for 23 classes from 6 surround cameras, boxes kept through temporary occlusion, and a mini split — [arXiv 1903.11027](https://arxiv.org/abs/1903.11027); [devkit](https://github.com/nutonomy/nuscenes-devkit)
- ADT gives exact 6DoF device and object poses, but only 74 dynamic objects in two scenes — [project page](https://www.projectaria.com/datasets/adt/)
- Ego3D-Bench reports that adding a text cognitive map built from estimated 3D coordinates raises Qwen2.5-VL-3B from 39.1 to 44.4 and 7B from 41.1 to 54.3 — [arXiv 2509.06266](https://arxiv.org/html/2509.06266v2)
- Disjoint-3DQA reports small gains from trajectory/BEV cues and a large gain from oracle 3D metadata (GPT-4o 65.60 → 83.2) — [arXiv 2505.24257](https://arxiv.org/html/2505.24257v1)

### Inferences
- Option A (days, cheap): VSI-Bench relative-direction / relative-distance subsets or OST-Bench agent-object questions. Shows the pipeline transfers to real footage and static out-of-view memory; does not test moving objects or extrapolation. Comparable published numbers exist (3B on VSI-Bench, 7B on OST-Bench).
- Option B (1–2 weeks, strongest): self-built two-choice set on nuScenes mini or TAPVid-3D (ADT + DriveTrack minival). Questions copy the simulated ones ("is A left or right of B, at the last frame / 2 s later", "which is closer to the camera"), answers computed from 3D ground truth and ego pose, items restricted to objects that are out of the queried camera at question time, plus last-frame-only and shuffled-frame controls. It allows three conditions: oracle positions, the project's estimated positions, and no sentence — and a position-error number. Cost: writing the question generator and a 2D→3D position estimator for real images (the simulated pipeline's estimator will not transfer unchanged).
- Option C (cheap add-on for an external, citable number on moving objects): EOC-Bench (has a 3B baseline) or TOC-Bench (shortcut-filtered) or DSI-Bench group-wise accuracy if its data is released. These report answer accuracy only.
- Lower priority: TAO-Amodal (376 GB), HOT3D (hand-held objects, test GT withheld), InfLevel/Perception Test cups (fixed camera, no ego-motion), PersistBench/MemoBench/TAPVid-360 (built for generative or tracking models; reusable as raw material but need a new QA layer).
- On one 24 GB GPU, inference with a 3B model is not the bottleneck for any of these; download size and data-access paperwork (ADT, HOT3D, nuScenes, Waymo, Ego4D sign-ups) are.

### Gaps
- No timing or memory measurements were found for Qwen2.5-VL-3B on these benchmarks; the feasibility statement is an estimate.
- nuScenes mini details and licence, and ADT licence terms on redistributing derived clips in a course report, need to be checked by the team before committing.
- Whether Beyond3D overlaps with Option B (it may already provide exactly this on egocentric data) is for the other researcher's notes.
