# Beyond3D (BEYOND3D) dataset / benchmark

Research date: 2026-11-19. All facts below were read on the cited page on that date. Pages were read through an automated fetch-and-summarise tool, so exact numbers should be re-checked against the PDF before they go into a report.

## 1. What is it exactly (identity, paper, repos, license, download, size)?

### Takeaway
"Beyond3D" is a real, exactly-named benchmark: BEYOND3D, introduced in "Long Time No See: Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos" (arXiv 2609.34630, 28 Sep 2026, ETH Zurich / Microsoft Spatial AI Lab / Bocconi). It is a multiple-choice video QA benchmark about objects that have left the camera's view, which is exactly the course project's topic, so the identification is close to certain. No other dataset with this name was found.

### Cited Findings
- Paper title: "Long Time No See: Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos"; arXiv id 2609.34630 [cs.CV]; only version v1, submitted Mon 28 Sep 2026; no venue listed in the comments field — [arXiv abstract](https://arxiv.org/abs/2609.34630)
- Authors: Fangzhou Ma, Ivo Alexander Ban, Eren Homburg, Gabriele Goletto, Rémi Pautrat, Mahdi Rad, Chiara Plizzari, Marc Pollefeys — [arXiv abstract](https://arxiv.org/abs/2609.34630)
- Affiliations: ETH Zurich (Ma, Ban, Homburg, Pollefeys); Microsoft Spatial AI Lab (Goletto, Pautrat, Rad, Pollefeys); Bocconi University (Plizzari) — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Abstract-level summary: a VQA benchmark for "out-of-sight spatiotemporal reasoning" in dynamic egocentric video; 9,000 questions, eight types, 135 videos, nine participants, built from HD-EPIC annotations; nine VLMs evaluated; best 42.2% vs 29.7% chance and 31.9% best text-only — [arXiv abstract](https://arxiv.org/abs/2609.34630)
- Project website: https://beyond3d-bench.github.io/website/ (the page is JavaScript-rendered; only its title "Beyond3D — Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning" could be read by the fetch tool) — [project site](https://beyond3d-bench.github.io/website/)
- Landing repository (README header, links to paper, website, dataset; 10 commits, 0 stars at time of reading; no license stated on it) — [GitHub Beyond3D-bench/Beyond3D-bench](https://github.com/Beyond3D-bench/Beyond3D-bench)
- Evaluation code repository, based on lmms-eval; original contributions under MIT, inherited lmms-eval code under MIT or Apache-2.0 — [GitHub Beyond3D-bench/vlm-evaluation](https://github.com/Beyond3D-bench/vlm-evaluation)
- Question/answer data on Hugging Face: `Ffffangzhu/BEYOND3D`, license CC BY-NC 4.0 (non-commercial), derived from HD-EPIC; no gating notice shown — [HF dataset card](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D/blob/main/README.md)
- HF repository files: `vqa_baseline.jsonl` (4.42 MB), `vqa_temporal_cues.jsonl` (6.25 MB), `data/` folder (parquet for the viewer), `prepare_viewer_data.py`, `README.md`; total repository size 11.5 MB; a commit message reads "Update temporal cues and remove distributed videos" — [HF file tree](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D/tree/main)
- Videos are NOT in the HF repository; `video_id` / `video_path` are reference identifiers only, and the source assets must be obtained separately following the vlm-evaluation repository — [HF dataset card](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D/blob/main/README.md)
- Download procedure per the evaluation README: (1) manually download HD-EPIC "intermediate data" participant ZIPs from a linked SharePoint (sign-in may be required) and extract to `$OOS_STORAGE_ROOT/data/HD-EPIC/Intermediate_data/` with folders `P01`–`P09`; (2) `bash setup.sh --data-only`, which downloads the two JSONL files from Hugging Face, downloads the HD-EPIC videos, and preprocesses them — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Disk needed for dataset preparation: about 120 GiB (about 115 GiB original HD-EPIC videos, about 2.2 GiB intermediate data) — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Underlying source dataset: HD-EPIC ("HD-EPIC: A Highly-Detailed Egocentric Video Dataset", Perrett et al., CVPR 2025, arXiv 2502.04144): 41 hours of video in 9 kitchens, digital twins of 413 kitchen fixtures, 20K object movements, 37K object masks lifted to 3D — [HD-EPIC arXiv](https://arxiv.org/abs/2502.04144)
- HD-EPIC download: University of Bristol data repository, complete download 2.3 TiB (sub-folders include Digital-Twin, SLAM-and-Gaze, Videos); the project page lists an alternative with Videos (mp4) 115.5 GB — [Bristol data repository](https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47); [HD-EPIC project page](https://hd-epic.github.io)
- HD-EPIC license is reported inconsistently: the Bristol record lists CC BY 4.0, the project page says CC BY-NC 4.0 (both seen only in a search-result summary, not re-read on the pages) — [Bristol data repository](https://data.bris.ac.uk/data/dataset/3cqb5b81wk2dc2379fx1mrxh47); [HD-EPIC project page](https://hd-epic.github.io)

### Inferences
- The name matches exactly (the paper, GitHub organisation and HF dataset all use "Beyond3D"/"BEYOND3D"), the date fits a Fall 2026 course (released 28 Sep 2026), and the topic (objects that have left the field of view) is the project's own research question. This is almost certainly what the professor meant.
- Earlier searches for "Beyond 3D" among spatial-reasoning benchmarks (VSI-Bench-style, MMSI-Video-Bench, DSI-Bench, Video-MSR, OpenBench, SIRI-Bench) returned nothing with that name, so there is no competing candidate. — search results only, see [MMSI-Video-Bench](https://arxiv.org/html/2512.10863v1), [DSI-Bench](https://arxiv.org/html/2510.18873v1)
- The earlier HF release apparently shipped preprocessed videos (commit "Release BEYOND3D benchmark videos and VQA") and later removed them, probably for HD-EPIC redistribution reasons. A search snippet describing "156 preprocessed MP4 videos" and "about 2.6 GB" most likely describes that earlier state; it does not match the current repository and is unverified.

### Gaps
- Venue: none stated; the paper is an arXiv preprint as of 2026-11-19. Whether it is under review somewhere is unknown.
- The project website body (leaderboard, examples) could not be read because it is rendered client-side.
- Exact size on disk of the preprocessed 448x448, 1 fps benchmark videos is not stated on any page read (only the 120 GiB preparation figure).
- Whether the SharePoint link for HD-EPIC intermediate data is open without an account was not tested.

## 2. What does the data contain (videos, sources, scene type, motion, length, resolution, fps)?

### Takeaway
All videos are real, unscripted, egocentric (head-worn Project Aria glasses) recordings of people cooking in 9 home kitchens, from HD-EPIC. The camera moves constantly; objects move only because the camera wearer picks them up and puts them down, and are assumed stationary in between. There are no self-moving objects.

### Cited Findings
- Scale: 135 videos, 9 participants, 9 kitchens, 581 target-object instances, 561 reference-object instances — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Source is only HD-EPIC (indoor kitchens, unscripted cooking and everyday activities, with object-movement annotations and 3D kitchen reconstructions) — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Objects are "active objects that the camera wearer manipulates and relocates"; between movements "we assume each object remains at its last annotated location until the next movement begins" — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Camera: Project Aria glasses, fisheye (the paper uses the Aria FISHEYE624 camera model); the image is a circular fisheye view — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Query times range up to 600 s into a video, stratified as early 0–149 s, middle 150–299 s, late 300–600 s; the HF viewer shows `query_time_sec` from 8 to 599 — [arXiv HTML](https://arxiv.org/html/2609.34630v1); [HF dataset](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D)
- Model input as used in the benchmark: the video prefix from start to the query time, sampled at 1 fps, resized to 448x448, pixels outside the fisheye circle masked black, a timestamp drawn in the corner of each frame — [arXiv HTML](https://arxiv.org/html/2609.34630v1); [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Dataset vocabulary seen in the HF viewer: 515 distinct `target_object` values, 267 `reference_object` values, 23 `fixture` values — [HF dataset](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D)
- About 70% of query anchors involve one or two moves of the target object before the query, with a tail up to 11 moves — [arXiv HTML](https://arxiv.org/html/2609.34630v1)

### Inferences
- A single question's input is up to about 600 frames at 1 fps (a 10-minute prefix). This is far longer than the project's 8-frame clips.
- The setting is "moved by a hand, then static", not "moving on its own". The project's velocity-extrapolation calculator ("where will the ball be 2 seconds after the last frame") has nothing to extrapolate here; the correct answer is always the last place the object was put down. What does transfer is: remembering the last sighting, and updating the object's position relative to a camera that keeps moving.
- Scenes are cluttered kitchens with hundreds of object categories, not three coloured balls; the position-estimating network trained on simulated balls would not apply without retraining.

### Gaps
- Number of videos is inconsistent across sources: the paper says 135; the fetch tool reported 148 distinct `video_id` values in the HF viewer (may be an artefact of how the viewer summarises a column); a search snippet said 156. Count unique `video_id` in `vqa_baseline.jsonl` to settle this.
- Original HD-EPIC video resolution and frame rate, and per-video duration statistics, were not found on the pages read.
- Total hours of video used by BEYOND3D (as opposed to HD-EPIC's 41 hours) is not stated.

## 3. What annotations exist (question types, answer format, counts, geometry)?

### Takeaway
9,000 multiple-choice questions (2–6 options) in eight types; the released files contain only questions, options, answers and a few metadata fields. Per-frame object positions, camera poses, kitchen meshes and visibility tracks are used to build the benchmark but come from HD-EPIC; whether the derived 1 fps visibility tracks are released was not confirmed.

### Cited Findings
- Structure: 1,000 out-of-sight query anchors (900 out of view, 100 occluded), each expanded into 8 questions (8,000), plus 1,000 visible control anchors used only for the Visibility Check question; total 9,000 — [arXiv HTML](https://arxiv.org/html/2609.34630v1); [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- The eight question types, answer format and chance level — [arXiv HTML](https://arxiv.org/html/2609.34630v1):
  - Visibility Check (visual grounding): yes/no, 50.0%
  - Last Visible Time (temporal grounding): 5 timestamps, 20.0%
  - Last Placement Time (temporal grounding): 5 timestamps, 20.0%
  - Nearest Fixture (scene localization): 5 fixture types or 3–6 counter areas, mean chance 22.7%
  - Object–Camera Direction (3D): 4 quadrants (front/back x left/right), 25.0%
  - Object–Camera Distance (3D): 3 bins (<1 m, 1–1.5 m, >=1.5 m), 33.3%
  - Object–Object Direction (3D): 3 clock ranges, 33.3%
  - Object–Object Distance (3D): 3 bins as above, 33.3%
- Correct-answer distributions are balanced for the temporal and 3D questions; Nearest Fixture keeps natural frequencies — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Questions are phrased as "At [TIME], assuming it remains at its last known position, ..." (reconstructed from the description of the visibility-cue variant) — [arXiv HTML, supplementary](https://arxiv.org/html/2609.34630v1)
- For object–object questions the reference object is marked with an 8x8 red marker, in the query frame only — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Released fields per row: `example_id`, `trajectory_id`, `video_id`, `variant`, `query_time_sec`, `target_object`, `reference_object`, `fixture`, `step` (8 values), `question_type` (8 values), `branch_group` (2 values), `question`, `choices`, `answer_index` (zero-based), `answer` — [HF dataset](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D); [HF dataset card](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D/blob/main/README.md)
- Example row: id `P03-20240217-100448__oos_staged_h10p0_20__4`, question asks which fixture type is closest to the fork's last known position, choices dishwasher / drawer / bin / counter / floor, answer dishwasher — [HF dataset](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D)
- Two configs, each 9,000 rows, single `test` split: `baseline`, and `temporal_cues` (same questions with a prepended note of the time intervals when the camera wearer moved the object) — [HF dataset card](https://huggingface.co/datasets/Ffffangzhu/BEYOND3D/blob/main/README.md)
- Geometry used to build the benchmark (from HD-EPIC): for each movement endpoint a 2D bounding box, mask, 3D centre and supporting/containing fixture; a reconstructed 3D digital twin per kitchen; camera poses; the Aria fisheye camera model — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Derived per-object visibility tracks at 1 fps with states visible, out_of_view, occluded, visually_unconfirmed, in_motion, fixture_ambiguous — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Visibility is computed in three stages: (1) field of view — nine footprint points are projected, the object is out_of_view if fewer than half fall inside the usable fisheye circle; (2) geometric occlusion — rays are cast against the static kitchen mesh with a 10 cm margin, occluded if at least half are blocked; (3) OWLv2 detection confirms visibility over intervals — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Human audit of the visibility tracks: three annotators, 30 videos, all nine participants, 4,102 scored marks over 344 frames; accuracy 83.5%, precision 81.2%, recall 62.7%, F1 70.8%, Krippendorff's alpha 0.76; all 1,000 out-of-sight anchors were also manually inspected — [arXiv HTML, supplementary](https://arxiv.org/html/2609.34630v1)

### Inferences
- There is no training split; BEYOND3D is test-only. Any training of a position head on real data would have to use HD-EPIC's own annotations (3D object centres at movement endpoints, camera poses), ideally from videos not in the 135.
- Positions are annotated only at pick-up/put-down endpoints, not densely per frame; "per-frame position" is the endpoint position held constant, combined with the per-frame camera pose.
- Because HD-EPIC provides camera poses, an oracle version of the project's "written track record + calculator" (last put-down 3D position transformed into the current camera frame) can be computed exactly, giving an upper bound for the four 3D question types.

### Gaps
- Whether the 1 fps visibility tracks and the 3D target positions per anchor are released (in the JSONL, the GitHub repo, or not at all) was not confirmed; the fields listed on the HF card do not include them.
- Exact definitions of the direction quadrants, the three clock ranges, and the reference frame for object–object direction were in a part of the paper not successfully extracted.
- Depth maps: not mentioned on any page read.

## 4. Does it contain questions about out-of-view / occluded objects, memory, or motion prediction?

### Takeaway
Yes on out-of-view, occlusion and memory: every one of the 8,000 main questions is about an object that was moved and is no longer visible at query time. No on motion prediction: nothing asks about a future position or a moving object's trajectory.

### Cited Findings
- The benchmark tests whether VLMs can track relocated objects and reason about their last known spatial state after they leave view — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- 900 anchors have the target out of the field of view and 100 have it occluded by scene geometry — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Out-of-sight horizon (time since the object was last visible) is binned short 2–10 s, medium 11–30 s, long >30 s, balanced across the query-time strata — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Memory-of-earlier-frames questions: Last Visible Time and Last Placement Time ask for a timestamp; Nearest Fixture asks where it was left — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Spatial-update questions: Object–Camera Direction and Distance ask where the unseen object is relative to the current viewpoint (example in the supplementary: "in which direction is the cloth from your viewpoint?" at query time 00:01:00.0) — [arXiv HTML, supplementary](https://arxiv.org/html/2609.34630v1)
- Objects are assumed to remain at the last annotated location until the next movement — [arXiv HTML](https://arxiv.org/html/2609.34630v1)

### Inferences
- Closest matches to the project's simulated questions: "Is the red ball left or right of the blue ball" maps to Object–Object Direction (3 clock ranges rather than left/right); "Which ball is closer to the camera" maps roughly to Object–Camera Distance (3 metric bins rather than a comparison). The "2 seconds after the last frame" future-extrapolation element has no counterpart.
- The out-of-sight horizons here (2 s to more than 30 s, within prefixes of up to 10 minutes) are longer than the project's 2 s.

### Gaps
- No accuracy is reported separately for out_of_view versus occluded anchors, and none by number of moves — [arXiv HTML](https://arxiv.org/html/2609.34630v1)

## 5. Evaluation protocol, metrics, and published baseline scores

### Takeaway
Metric is macro-averaged multiple-choice accuracy over the eight question types; chance is 29.7%. The best of nine models (Qwen-3.6-27B) reaches 42.2%; most 3D question types sit within a few points of chance for every model. Qwen2.5-VL, any roughly 3B model, and a human baseline are all absent from the paper.

### Cited Findings
- Metrics: per-question-type accuracy and macro-average accuracy with each type weighted equally; random guessing 29.7% — [arXiv HTML](https://arxiv.org/html/2609.34630v1); [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Three input settings: Text only; Last Frame Only + Text; Video + Text — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Prompt: system prompt "You are a helpful assistant trained to answer spatial and visual questions based on egocentric videos."; plus "Use the video to answer the question. The video is sampled at 1 frame per second."; then "Question: [QUESTION]", lettered options, and "Select the best option and output only its letter." — [arXiv HTML, supplementary](https://arxiv.org/html/2609.34630v1)
- Each question is evaluated independently with its video prefix as context — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Models evaluated (nine): Qwen-3.6 35B-A3B, Qwen-3.6 27B, Qwen-3.5 9B, Qwen-3-VL 8B, InternVL-3.5 8B (general-purpose); VLM-3R 7B, Spatial-MLLM 6B, Cambrian-P 7B, SenseNova-SI 8B (spatially specialised) — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Video + Text results, accuracy %, columns = Macro / Visibility Check / Last Visible Time / Last Placement Time / Nearest Fixture / Obj-Cam Direction / Obj-Cam Distance / Obj-Obj Direction / Obj-Obj Distance — [arXiv HTML, Table 2](https://arxiv.org/html/2609.34630v1):
  - Random: 29.7 / 50.0 / 20.0 / 20.0 / 22.7 / 25.0 / 33.3 / 33.3 / 33.3
  - Qwen-3.6 35B-A3B: 39.6 / 66.3 / 23.3 / 29.4 / 51.3 / 33.7 / 35.9 / 41.1 / 35.6
  - Qwen-3.6 27B: 42.2 / 68.4 / 30.4 / 39.4 / 50.0 / 34.6 / 34.9 / 43.6 / 36.2
  - Qwen-3.5 9B: 38.6 / 59.3 / 28.2 / 30.5 / 47.9 / 33.4 / 39.2 / 34.7 / 35.8
  - Qwen-3-VL 8B: 36.0 / 59.8 / 21.3 / 25.7 / 48.2 / 32.7 / 33.6 / 31.5 / 35.2
  - InternVL-3.5 8B: 35.8 / 60.4 / 21.8 / 20.7 / 38.9 / 31.4 / 42.6 / 35.8 / 34.9
  - VLM-3R 7B: 37.2 / 58.8 / 23.9 / 23.2 / 49.8 / 35.6 / 34.4 / 35.6 / 36.5
  - Spatial-MLLM 6B: 31.6 / 53.3 / 22.8 / 19.2 / 29.5 / 24.6 / 34.8 / 34.0 / 34.8
  - Cambrian-P 7B: 33.5 / 52.2 / 20.0 / 17.5 / 50.5 / 27.3 / 33.4 / 33.9 / 33.2
  - SenseNova-SI 8B: 35.4 / 57.9 / 20.6 / 22.9 / 44.4 / 29.8 / 32.9 / 34.9 / 39.8
- Text-only macro averages range from 29.0 (SenseNova-SI) to 31.9 (Cambrian-P); text-only Nearest Fixture reaches 41.3 (Cambrian-P) and 40.2 (Qwen-3.6 27B) against 22.7 chance — [arXiv HTML, Table 2](https://arxiv.org/html/2609.34630v1)
- Last-frame-only macro averages range from 31.8 to 35.2 (best Qwen-3.6 27B, 35.2); Visibility Check from a single frame reaches 76.5 (Qwen-3.6 27B), higher than the same model with full video (68.4) — [arXiv HTML, Table 2](https://arxiv.org/html/2609.34630v1)
- By out-of-sight horizon (macro, Video + Text): short 40.4%, medium 34.8%, long 31.9%; by query time: early 34.9%, middle 38.4%, late 36.6% (the text does not say which models these average over) — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- Ablations on Qwen-3.6-27B: the temporal cue (telling the model when the object was moved) improves all downstream question types across horizons (numbers only in a figure); adding a statement that the target is not visible raises Nearest Fixture by about 8% and the 3D tasks by 0–2.4% — [arXiv HTML](https://arxiv.org/html/2609.34630v1)
- The paper notes 3D-specialised models may be handicapped by tighter context budgets, and that evaluation uses a single kitchen-based source — [arXiv HTML](https://arxiv.org/html/2609.34630v1)

### Inferences
- The benchmark is very hard and has little headroom between chance and state of the art: with 1,000 questions per type, the 95% binomial half-width on one type is about +/-3 points, so many per-type differences in the table (for example 33.4 vs 34.9 on a 33.3% chance task) are not distinguishable from chance. A 3B model from an older generation should be expected to score near chance on the 3D types; differences from the project's method will need the full 1,000 questions per type and paired comparison to be detectable.
- The Nearest Fixture question has a strong language prior (about 40% from text alone), so a text-only control is needed for any claim on that type.
- Even when told when the object was moved, 3D-relation accuracy improves only modestly, which suggests the bottleneck is spatial updating relative to the moving camera, not only finding the last sighting. This is the same split the project studies (track record vs calculator).

### Gaps
- Qwen2.5-VL (any size): not evaluated in the paper and not among the repository's model presets. No published BEYOND3D number for it was found.
- Models around 3B: none evaluated; the smallest is Spatial-MLLM at 6B.
- Human-level performance on the questions: not reported in any text that could be read (the human study in the paper validates the visibility labels, not question answering).
- No third-party papers citing or evaluating on BEYOND3D were found; the benchmark is under two months old.
- Exact numbers for the temporal-cue ablation and per-model horizon curves are only in figures.

## 6. What would be needed to run a Qwen2.5-VL-3B pipeline on it?

### Takeaway
There is an official lmms-eval-based harness with a documented JSONL format, a validation script, and a `OOS_LIMIT` option for small runs, but no Qwen2.5-VL preset, so an adapter must be added. The main practical costs are about 120 GiB of disk for HD-EPIC videos and long inputs (up to about 600 frames at 448x448 per question, 9,000 questions).

### Cited Findings
- Requirements: Linux x86-64, NVIDIA GPU with a CUDA 12.8-compatible driver, Python 3.10+, `uv`, a C compiler and `make` (for bundled FFmpeg); GPU memory requirements are not stated — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Presets available: `qwen3_5_9b`, `qwen3_6_27b`, `qwen3_6_35b_a3b`, `qwen3_vl` (Qwen3-VL-8B-Instruct), `sensenova_qwen`, `internvl`, `cambrian_p`, `spatial_mllm`, `vlm3r`; defined in `launchers/models/`, pinned in `models/manifest.yaml`; Qwen2.5-VL is not listed — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Run command: `OOS_MODEL=qwen3_5_9b OOS_OFFLINE=1 bash launchers/run_oos_eval.sh`; temporal-cue variant via `OOS_DATASET_FILE=vqa_temporal_cues.jsonl`; subset run via `OOS_LIMIT=2`; Slurm launcher `launchers/slurm_oos_eval.sh`; outputs under `outputs/oos_videoqa/<model>/` — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Frame convention: prefix up to query time at 1 fps, 448x448, fisheye mask, burned-in timestamp; models with shorter context get uniformly subsampled frames with the query-time frame always kept; the paper used 150 frames for InternVL-3.5, Cambrian-P and VLM-3R and 64 frames for Spatial-MLLM — [arXiv HTML](https://arxiv.org/html/2609.34630v1); [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Custom data format (`docs/dataset.md`): one JSON object per line with an id, `query_time_sec`, `video_path`, `question`, `choices` + `correct_idx` (zero-based), optional `question_class` used for macro-accuracy grouping; set `OOS_DATASET_JSONL` and `OOS_VIDEO_BASE_DIR`; validate with `tools/validate_dataset.py` (`--schema-only` works before videos exist) — [vlm-evaluation docs/dataset.md](https://github.com/Beyond3D-bench/vlm-evaluation/blob/main/docs/dataset.md)
- The BEYOND3D task and the model adapters live in the repository's `lmms_eval/` directory — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)
- Data preparation needs the HD-EPIC intermediate data (manual SharePoint download, folders `P01`–`P09` each with per-video folders and `device_calibration.json`) before `setup.sh --data-only` — [vlm-evaluation README](https://github.com/Beyond3D-bench/vlm-evaluation)

### Inferences
- Minimum path to a comparable baseline number: add a Qwen2.5-VL-3B adapter/preset to the harness (upstream lmms-eval has a Qwen2.5-VL model class that can likely be reused; not verified in this fork), keep the official prompt and preprocessing, choose and report a frame cap (64 or 150 would match caps already used in the paper), and report macro accuracy plus the text-only and last-frame controls.
- The project's own 8-frame, 224x224 setting is far from the official protocol. Running at 8 frames is possible via subsampling but will mostly discard the moment the object was last seen; results at that setting should be reported as a separate, non-comparable condition.
- Cheaper first step: the JSONL is 4.4 MB and can be inspected today without any video; a subset (for example one participant, or the four 3D types) can be run with `OOS_LIMIT` or a filtered JSONL through the custom-dataset path.
- Because HD-EPIC supplies camera poses and 3D put-down positions, the project can separately test its "calculator" with oracle inputs (ground-truth last position + camera pose) and its "track record" with oracle last-seen times (the `temporal_cues` config is effectively this), isolating which component fails on real video.
- Compute rough guide (own estimate, not from a source): at a 150-frame cap and 448x448, each question is of the order of tens of thousands of visual tokens for a Qwen-VL-style model; 9,000 questions on a 3B model is a multi-GPU-hour job, not a multi-day one, but memory at long context is the constraint to check first.

### Gaps
- No documentation on adding a new model to the harness was found; the adapter interface must be read from the source.
- GPU memory and wall-clock time for the paper's runs are not stated.
- Whether the harness scores by exact letter match or with a more lenient parser is not documented in the README; check the task file in `lmms_eval/`.
- Whether HD-EPIC's licence terms (CC BY 4.0 vs CC BY-NC 4.0 reported inconsistently) impose any constraint beyond non-commercial academic use was not verified; course use is non-commercial either way.
