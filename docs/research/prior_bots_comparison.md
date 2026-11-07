# Prior bots comparison: KataCR [1], Claude Royale [2] and clashbot (ours)

Written 2026-11-07. Read-only research: nothing was installed or run, no bot launched, no other file edited. Scratch copies of the sources are under the session scratchpad only.

**How the sources were checked.** [1]: the arXiv abstract page, the full arXiv HTML (v1) and the PDF were fetched and the text extracted locally; every number below was read from that text, not from a summary. The local clone `~/cr_scratch/katacr` (commit `36ceb9f`, 2024-06-06, MIT) was read for README, `UPDATE_LOGS.md` and the interaction code. [2]: the GitHub API file tree (commit `70ecc8c`, 2026-01-25) was listed and README, `CLAUDE.md`, `eval/*`, `memory/*`, `.claude/agents/*` and the `scripts/*` were fetched raw. Our numbers come from `docs/research/course_mapping_audit.md` (cited as CMA) and `docs/research/packaging_and_benchmarks_audit.md` (cited as PBA).

**Cell tags used in every table.**

| tag | meaning |
|---|---|
| R | reported by the authors (paper, README, repo docs) |
| R-code | in the authors' repo code or notes but not in the paper/README |
| M | measured by us (file named in the cell) |
| I | my inference from code or numbers; flagged, not stated by the authors |
| NR | not reported |
| TBD | to be filled from the user's live Training Camp games |

---

## 0. Headline results

| | KataCR [1] | Claude Royale [2] | clashbot (ours) |
|---|---|---|---|
| Best reported win rate vs built-in AI | **5%** (1 of 20) for the headline model StARformer-3L, step 50; best row **10%** (StARformer-3L, step 30). Range over nine models 0-10%. Paper Table 3, Sec. 5.3 [R] | NR (never played the built-in AI; the harness only taps "Battle" on the ladder) [I from `scripts/tap.sh`, `CLAUDE.md`] | TBD (n games); 1 documented win in 3 games in run 20261107-013441 [M, CMA C6] |
| Qualitative claim | Abstract: "successfully defeating built-in AI opponents"; Conclusion: "our agents still cannot consistently defeat the built-in AI" [R]. README links a video of "12 Consecutive Wins" (EN) / "12个获胜对局" = "12 winning matches" (CN); the number of games the 12 were drawn from is not given [R] | "1000+ trophies" (README); `memory/STATUS.md` says 1723 trophies, Arena 6, peak about 1813 as of 2026-01-25 [R]. Vs human ladder players, not a built-in AI | first win 2026-11-07 (user report) |
| Decision latency | 120 ms decision + 240 ms perception fusion [R] | "about 7 seconds" screenshot to action per agent (README); 3 agents stagger plays to one card per 2-3 s [R] | p50 338.9 / p95 571.8 ms capture-to-click, run 20261107-111554 [M, CMA C7] |
| Detector quality | mAP 67.4 / AP50 84.3 for YOLOv8-l x2 (our weights are very likely this pair, see 1.2) on a 6,939-image real validation set [R, Table 2] | no detector; Claude vision reads screenshots. Elixir read exactly in 25/25 test screenshots after prompt fixes [R] | uses KataCR weights unchanged; no own detector accuracy; live-state parity cosine 0.6714 on one game [M, CMA C1] |
| Offline decision metric | none against expert moves; reward and win rate only (Table 3) | 25 labeled screenshots, Haiku overall 65% (card choice 11/25 = 44%) [R, `eval/MODEL_COMPARISON.md`] | 4-way ranker test 0.4974 vs random 0.25 (n = 766) [M, CMA C4] |

---

## 1. KataCR [1]: Wu et al., arXiv:2504.04783 (ICIRA 2024)

arXiv v1 submitted 7 Apr 2025, cs.LG, the only version. Published as Springer LNCS vol. 15206 (ICIRA 2024), pp. 255-269, doi 10.1007/978-981-96-0792-1_20 (checked on arXiv and Crossref). Authors: Tianyang Wu, Lipeng Wan, Yuhang Wang, Qiang Wan, Xuguang Lan (Xi'an Jiaotong University). Code https://github.com/wty-yy/katacr; datasets https://github.com/wty-yy/Clash-Royale-Detection-Dataset and https://github.com/wty-yy/Clash-Royale-Replay-Dataset. The README calls it "my undergraduate thesis project".

### 1.1 Problem setting and system

| item | value | source | tag |
|---|---|---|---|
| Game and mode | Clash Royale, classic 1v1, "Season 59 in May 2024". Regular time 3 min, overtime 2 min, deck of 8, hand of 4 | Sec. 2, 2.2 | R |
| Claim of novelty | "non-embedded" agent: acts only from pixels, as a human does; offline RL because online interaction with a phone is too slow | Abstract, Sec. 1 | R |
| Pipeline | phone video stream -> object detection + OCR + classifiers -> feature fusion into state -> offline-RL policy -> ADB tap | Fig. 1, README, `policy/env/*` | R / R-code |
| Perceived content | arena (king tower, minor towers, troops), hand cards, game time, elixir | Sec. 2.1 | R |
| Unit state | position on an 18 x 32 grid, class (1-150), faction (own/enemy), health bar image | Sec. 2.3 | R |
| Capture path | `cv2.VideoCapture(2)` on a phone video stream; README: the stream "must rely on V4L2 in the Linux kernel"; the decision part "can only run on Linux" | `sar_daemon.py`, README | R-code |
| Actuation | `adb shell input tap x y` | `policy/env/utils.py` | R-code |
| Opponent in eval | "built-in AI with 8000 rating" (expert data); the eval script navigates to the "Training camp button" and starts "trainer matches" | Sec. 5.3; `sar_daemon.py::_start_new_episode`, README | R / R-code |

### 1.2 Perception

| item | value | source | tag |
|---|---|---|---|
| Detector | refactored YOLOv8 (plus a self-written JAX YOLOv5, abandoned); "YOLOv8-l x n" = n large models, each covering a size band of classes | Sec. 3, 5.2 | R |
| Classes | 154 slice categories, 150 to be detected | Sec. 5.1 | R |
| Training data | synthetic only: 4,654 slices cut out with SAM, composited into generated frames; each training set 20,000 images, up to 80 epochs | Sec. 3, 5.1, 5.2 | R |
| Validation | 6,939 manually labeled real-stream images, 116,878 boxes (about 17 per image); models never trained on it | Sec. 5.1 | R |
| Other perception parts | OCR (PaddleOCR, for clock and on-screen text), ResNet card classifier (README: "Classifies only 2.6 Cycle Hog Rider cards"), ResNet elixir classifier. The paper body gives no OCR or classifier accuracy | README, `policy/perceptron` | R-code |
| Published detector weights | "YOLOv8 x2": detector1 and detector2, v0.7.13, updated 2024.05.01 (Google Drive) | README | R |
| Our weights = theirs | `katacr_d1.pt` / `katacr_d2.pt` are downloaded from those two Drive ids (`tools/worker_setup.sh:33-34`, PBA) and are 87.8 MB each, which matches a YOLOv8-l in fp16; so they are very likely the paper's YOLOv8-l x2 row. `UPDATE_LOGS.md` v0.7.12 lists "P,R,AP50,mAP: 0.89,0.797,0.843,0.676", close to the x2 row | CMA C3, `UPDATE_LOGS.md` | I |

**Table 2 of [1] (Sec. 5.2), copied.** FPS on an RTX 4090 at batch size 1; FPS(T) = detection plus ByteTrack on an RTX 4060 Laptop. The paper does not define "mAP"; `UPDATE_LOGS.md` reports "AP@50, AP@75, mAP", so it is presumably COCO mAP@[.5:.95] (I).

| Model | AP50 | P50 | R50 | mAP | mAP(S) | FPS | FPS(T) | Aug. |
|---|---|---|---|---|---|---|---|---|
| YOLOv5-l | 66.2 | 84.4 | 63.8 | 53.2 | NA | 59 | NA | no |
| YOLOv8-x | 83.1 | 93.9 | 68.3 | 67.7 | 39.8 | 68 | 31 | no |
| YOLOv8-x | 85.3 | 90.7 | 80.4 | 66.8 | 35.9 | 68 | 31 | yes |
| YOLOv8-l x2 | 84.3 | 89.5 | 79.8 | 67.4 | 43.9 | 34 | 18 | yes |
| YOLOv8-l x3 | 85.2 | 89.7 | 80.9 | 68.8 | 48.3 | 23 | 10 | yes |

### 1.3 Decision model

| item | value | source | tag |
|---|---|---|---|
| Method | offline RL "more akin to imitation learning than value estimation"; Decision Transformer (DT) formulation, model built on StARformer | Sec. 4.2 | R |
| Variants | StARformer-3L (input length 3L), StARformer-2L (2L), DT-4L (4L, temporal attention only); step sizes L = 30, 50, 100 | Sec. 4.2, Table 3 | R |
| State | grid tensor 18 x 32 x 15 (category, sub-faction, health, extras) plus global vector (hand cards, elixir) | Sec. 4.1 | R |
| Action space | continuous deployment coordinates `a_pos` (2-D) and discrete hand-card index `a_select`; plus a predicted delay to the next action (delayed action prediction, T_delay = 20) | Sec. 4.1, 4.3 | R |
| Reward (used for return-to-go and for reporting) | tower health + tower destruction (main tower = 3x) + main-tower activation (+/-0.1) + elixir-overflow penalty (0.05 per second) | Sec. 4.1, Eq. 1-2 | R |
| Expert data | 105 episodes, one human player vs the 8000-rated built-in AI, fixed decks; 113,981 frames; action frames 4.01%; resampling ratio 24.92 : 1.04; mean action delay 21.26 frames; T_delay = 20 | Sec. 5.3 | R |
| Episode count discrepancy | `UPDATE_LOGS.md` v0.8: "增大离线数据集到113981，总共124个episodes" = 124 episodes for the same 113,981 frames, vs 105 in the paper | `UPDATE_LOGS.md` | R-code (conflict) |
| Augmentation | random permutation of the hand cards | Sec. 5.3 | R |
| Decks | paper: "fixed card decks". Replay folders are named `golem_ai` (the training run names contain it) and `fast_hog_2.6`; the card classifier covers one deck only. Deck contents are inferred from folder and file names | Sec. 5.3, Replay Dataset repo, README, `train_policy.sh` | R / I |
| Training compute | single GPU (`--batch-size 32`, nominal 128, 20-30 epochs); GPU model and hours NR | README | NR (partial) |

**Table 3 of [1] (Sec. 5.3), copied.** "Highest average reward obtained from 20 rounds of interaction with the environment for the first 10 epochs of each model". Values are mean +/- std over 20 rounds; Game Duration units are not stated in the paper (I: seconds, since a match is at most 300 s). Win Rate is a multiple of 5%, consistent with n = 20.

| Model | Step L | Total reward | Game duration | Num. actions | Win rate |
|---|---|---|---|---|---|
| DT-4L | 50 | -5.7 +/- 2.5 | 148.9 +/- 33.6 | 128.7 +/- 37.7 | 5% |
| StARformer-2L | 30 | -6.0 +/- 2.3 | 135.0 +/- 35.1 | 141.8 +/- 57.9 | 5% |
| StARformer-2L | 50 | -6.2 +/- 2.2 | 131.9 +/- 44.3 | 195.3 +/- 69.8 | 5% |
| StARformer-2L | 100 | -4.9 +/- 2.8 | 150.2 +/- 35.6 | 187.6 +/- 48.2 | 0% |
| StARformer-3L | 30 | -5.1 +/- 3.7 | 147.2 +/- 37.4 | 190.8 +/- 52.7 | 10% |
| **StARformer-3L** | **50** | **-4.7 +/- 3.1** | **158.9 +/- 27.7** | **207.8 +/- 48.2** | **5%** |
| StARformer-3L | 100 | -6.1 +/- 2.2 | 125.9 +/- 37.8 | 144.6 +/- 42.9 | 5% |
| StARformer-3L (Full Card) | 50 | -5.6 +/- 2.1 | 150.2 +/- 38.6 | 195.3 +/- 69.8 | 0% |
| StARformer-2L (Discrete Action) | 50 | -7.5 +/- 0.8 | 123.1 +/- 39.2 | 21.9 +/- 9.4 | 0% |

Text claims around Table 3 (Sec. 5.3): "a 37% improvement in performance when transitioning from discrete prediction to continuous prediction" and "a 24% increase" from StARformer 2L to 3L. The paper does not show the arithmetic; both reproduce as relative reductions of the (negative) total reward: (7.5 - 4.7) / 7.5 = 37% (2L discrete vs 3L continuous, so depth changes too; the like-for-like 2L continuous row gives (7.5 - 6.2) / 7.5 = 17%) and (6.2 - 4.7) / 6.2 = 24% (2L vs 3L at step 50). These are reward gains on a shaped reward, not win-rate gains.

### 1.4 Latency, hardware, evaluation protocol

| item | value | source | tag |
|---|---|---|---|
| Decision time | "average decision-making time is 120ms" | Sec. 5.3, README | R |
| Perception fusion time | 240 ms | Sec. 5.3, README | R |
| End-to-end | not stated; about 360 ms if the two stages are serial | - | I |
| Decision frequency | not stated. The daemon emits one state-action-reward tuple every 2nd perceived frame (`interval=2`), and the eval executes a predicted action only if its predicted delay is at most 8 frames (`max_delay=8`) | `sar_daemon.py`, `eval.py` | R-code |
| Detector throughput | 34 FPS (RTX 4090) / 18 FPS with ByteTrack (RTX 4060 Laptop) for the x2 detector | Table 2 | R |
| Hardware | smartphone running HarmonyOS (1080 x 2400 screen per README); Ubuntu 24.04 LTS, Ryzen 9 7940H CPU, RTX 4060 Laptop GPU | Sec. 5.3, README | R |
| Eval opponent and size | built-in AI; 20 rounds per model (Table 3); the paper's validation environment is the phone and the README eval scripts drive it. The paper gives no game count for the match videos | Sec. 5.3, README | R |
| Model selection | the reported row is the best average reward over the first 10 epochs, chosen on the same 20-round interaction environment; no separate held-out evaluation | Sec. 5.3 | R (caveat is mine) |
| Match videos | Bilibili BV1xn4y1R7GQ, "12 Consecutive Wins" in the English README; these are the only evidence for wins beyond 0-10% | README | R |

### 1.5 Limits acknowledged by the authors (Sec. 6)

Fixed card sets only; agents "cannot consistently defeat the built-in AI, falling short of human-level performance"; datasets need substantial human effort; suggests online RL and faster perception fusion as future work.

---

## 2. Claude Royale [2]: E. Houseworth, github.com/houseworthe/claude-royale

Author name is taken from the GitHub profile (login `houseworthe`, name "Ethan Houseworth"). Repo created 2025-12-07, last commit 2026-01-25 (`70ecc8c`), 54 commits, MIT licence, 26 stars, 8 forks (GitHub API, 2026-11-07). README: "Built over a weekend. December 2025." It is a harness plus prompts, not a paper; there is no write-up beyond the README and the `eval/` and `memory/` notes.

### 2.1 System

| item | value | source | tag |
|---|---|---|---|
| Game and platform | Clash Royale in BlueStacks on macOS, window fixed at (0, 33), 1728 x 1084; Retina screenshots 3456 x 2234 | README, `memory/LEARNINGS.md` | R |
| Perception | `screencapture -R` of the BlueStacks window (`scripts/screenshot.sh`); the agent opens the PNG with the `Read` tool and a Claude vision model interprets it. No detector, OCR or tracker | `scripts/screenshot.sh`, `.claude/agents/player-classic.md` | R / R-code |
| Action | `scripts/play_card.sh <slot 1-4> <cell>`: `cliclick` click on the card slot, 0.15 s sleep, click on a cell of an 8 x 8 grid (columns 1-8, rows A-H) whose pixel coordinates are in `config/gameplay.json` | `scripts/play_card.sh` | R-code |
| Architecture | one "Commander" Claude Code instance (menus, chests, taps Battle, tracks trophies, updates memory files) plus 3 parallel "Player" sub-agents per match that all play the same match and stop at the result screen | README, `CLAUDE.md` | R |
| Model | Players: `model: haiku` in `.claude/agents/player-classic.md` and `player-eval.md`. Commander model: not stated (`.claude/settings.json` sets none). README mentions `SPAWN_INSTRUCTIONS_HAIKU.md`, which is not in the repo tree | agent frontmatter, API tree | R-code / NR |
| Knowledge | hand-written prompts: lane-defence rules, "never sit at 10 elixir", Hog Rider only at 2E/5E; `memory/{STATUS,LEARNINGS,GOALS,DECK}.md` carry across sessions. No retrieval, no pro data | `CLAUDE.md`, `memory/*` | R |
| Training | none (prompt engineering only); the README says Claude wrote its own strategy prompt | README | R |
| Opponent | ladder matchmaking against human players (names listed in `memory/STATUS.md`) | `memory/STATUS.md` | R |
| Auto-opener | first card at 5 s and second at 8 s after tapping Battle, scripted, because the model was too slow | README, `CLAUDE.md` | R |
| Latency | README: "roughly 7 seconds" round trip screenshot -> vision -> decision -> tool call, per agent; 3 agents give a card every 2-3 s. Eval harness: Haiku about 15-20 s per screenshot (17 s), Sonnet 4.5 about 20-25 s (23 s), Opus 4.5 about 35-40 s (38 s). The two sets of numbers disagree; my guess (I) is that the eval prompt also asks for perception and reasoning fields, which lengthens the output | README, `eval/MODEL_COMPARISON.md` | R |
| Cost per game | not stated anywhere (grep for cost, token, API key, subscription finds only a warning that subagent transcripts "consume massive tokens") | repo | NR |
| Hardware | a Mac (BlueStacks, Retina display); model not stated; inference in the cloud | README | R / NR |
| Streaming | 12+ hours live on Twitch; chat read between matches; 2v2 mode with a human partner | README | R |

### 2.2 Results reported

| result | value | source | tag |
|---|---|---|---|
| Trophies | README: "1000+ trophies (Spell Valley - Arena 5)". `memory/STATUS.md` (updated 2026-01-25): 1723 trophies, King Tower level 21, Arena 6; peak about 1813; "ceiling around ~1770, skill gap emerging". The README is stale | README, STATUS | R |
| Games in a row | "played 14 games in a row autonomously, with minor human intervention" | README | R |
| Per-session records in the notes | Session 38 (2026-01-10): 4W-9L-1D at 1000 trophies; Session 36: 0W-4L; Session 35: 0W-1L; Session 11: 2W-7L (-41 trophies); account-level "151 total battles won" at session 39. No overall win rate, no total game count | STATUS, LEARNINGS | R |
| Win/loss reading | the result screen says "WINNER!" on every result, including losses; wins are judged by trophy delta; Haiku agents misread "-14" as "+14" in session 11 | README, LEARNINGS | R |
| Offline screenshot eval | 25 labeled screenshots from real matches, labels written by the author with Claude structuring them; rubric with partial credit (perception 40 pts, decision 60 pts); code grader `scripts/grade-evals.sh`; pass threshold 90/100 for pass@k | `eval/EVAL_DESIGN.md`, `eval/grading.json` | R |
| Model comparison (2026-01-18) | original baseline: Haiku 63%, Sonnet 4.5 54%, Opus 4.5 56% overall; card choice 10/25, 6/25, 8/25; placement 11/25, 8/25, 8/25. After prompt changes Haiku 65%; elixir read 25/25; card choice 11/25 (44%); placement 11/25 (44%); destroyed-tower detection 3/7 (43%). The summary table says Haiku 65%, Sonnet 54%, Opus 56% | `eval/MODEL_COMPARISON.md` | R |
| Wrong-card and hallucination counts | wrong card (not hallucinated): Haiku 15, Sonnet 18, Opus 16; Sonnet and Opus each recommended a card not in hand at least once | `eval/MODEL_COMPARISON.md` | R |
| Raw outputs | `eval/results/` is documented but not in the repo (tree has only `labels/`, `screenshots/`, `grading.json`, docs), so the table cannot be re-derived | API tree | I |

### 2.3 Limitations stated or visible

| limitation | source |
|---|---|
| Hard-coded to the author's BlueStacks window position and resolution; "Can I use this? Not yet" | README |
| About 7 s latency; early losses from slow openings; agents "freeze at 10 elixir" in critical moments (STATUS Known Issues) | README, STATUS |
| Sub-agents once tapped "Play Again" by mistake and started unsupervised matches; fixed by restricting tools | README, LEARNINGS |
| No aggregate win rate, no variance, opponent skill uncontrolled (ladder), deck changed several times (Giant beatdown, Fire Spirit, Hog cycle) | STATUS, DECK |
| Eval set is 25 screenshots with one annotator; "bigger models perform worse" is from n = 25 on one rubric | `eval/MODEL_COMPARISON.md` |
| Needs BlueStacks, Claude Code CLI, Node, `cliclick`, `jq` | README |

---

## 3. Feasibility on our Mac by Monday 2026-11-09 (under about 3 hours of setup)

State of this machine (read-only check, 2026-11-07): Apple silicon (arm64); present: `claude` CLI, `jq`, Node 24, Homebrew, conda Python; **absent: `cliclick`, `adb`, `scrcpy`, any Android emulator (no BlueStacks in /Applications)**; `ANTHROPIC_API_KEY` is not set in this shell; KataCR detector weights and the clone are present.

| question | KataCR [1] | Claude Royale [2] |
|---|---|---|
| Verdict for under 3 h | **No.** | **Not as shipped. A re-implemented "Claude-Royale-style" baseline on our own stack: plausible in 2-3 h (my estimate), but it is a baseline, not their code.** |
| What is missing | policy checkpoints (StARformer_3L step 50 etc.) are Google Drive folder links in the README, availability not checked by us; JAX/Flax/Orbax policy stack, PaddleOCR, two ResNet classifiers (trained for one fixed deck) | BlueStacks + Clash Royale install and a separate account (tutorial); `cliclick`; coordinates for the grid and buttons (`scripts/calibrate.sh`, interactive) |
| Hard blockers | (1) frames come from `cv2.VideoCapture(2)` V4L2 phone stream, README: Linux only; (2) taps use `adb shell input tap`, an iPhone has no ADB; (3) screen split constants are for 1080 x 2400 (ratio 2.22); (4) `requirements.txt` pins CUDA builds (`jaxlib 0.4.26+cuda12`, `paddlepaddle_gpu==2.6.1.post120`), CPU wheels exist for macOS arm64 (jaxlib 0.4.26; paddlepaddle 2.6.2, not 2.6.1; checked on PyPI) but the stack is untested by us; (5) card classifier only knows the 2.6 Hog Rider deck and the policies were trained on `golem_ai` replays | (1) the harness plays on the ladder via "Battle", not Training Camp, so results would be vs humans; (2) hard-coded to BlueStacks window geometry; to point it at iPhone Mirroring, `screenshot.sh`, `play_card.sh` and `tap.sh` must be rewritten and the grid recalibrated; (3) the Commander/Player design needs Claude Code subagents and an unattended ladder account |
| Weights / keys | public weights on Drive (unverified); no key | no weights; needs Claude access. The repo says "Claude Code CLI" (installed here, login state unchecked); a direct API route needs an API key, not set here |
| Effort estimate (mine) | 1-2 days for a faithful port plus retraining the classifiers for our deck; high risk of a silent train/serve mismatch (our own live-state parity is only 0.67 cosine, CMA C1) | 1-2 h to swap capture/click onto our `MirroringSource` / `MirroringExecutor` (guarded by `--live` and `CLASHBOT_ALLOW_LIVE=1`, Training Camp only, D-031/D-050), plus 1 h for an 8 x 8 grid mapping and a Haiku prompt; the 7 s loop means only about 10-12 plays per 3-minute match (their own estimate for Haiku in `MODEL_COMPARISON.md`) |
| Expected value of running it | low: their own Table 3 says 0-10% win rate on 20 rounds for the best checkpoints; quoting it is as informative | medium: it is the only prior LLM-agent point of comparison and would give a Training Camp number for an LLM-only agent; but latency makes it likely to lose to a built-in AI |
| Recommendation | do not run; cite the reported numbers (Table 3) and compare perception/latency only | do not run the repo; if the user wants a third arm, ask first, then build the baseline behind a flag and run the same 10 Training Camp games. Do not point the original scripts at the real ladder account |
| Safe cheap addition | we already run their detector pair; report our measured 100.5 ms mean per call on MPS (CMA C7) next to their 34 / 18 FPS (RTX 4090 / 4060 Laptop with tracker) | none needed |

---

## 4. Proposed fair comparison

Columns are the three systems. Each cell ends with a tag. "Same" and "different" notes for protocol are in section 5.

| dimension | KataCR [1] | Claude Royale [2] | clashbot (ours) |
|---|---|---|---|
| Task and access | non-embedded, pixels only, real phone, Training Camp built-in AI (8000 rating) [R, R-code] | non-embedded, pixels only, BlueStacks window on macOS, ladder vs humans [R] | non-embedded, pixels only, iPhone Mirroring window, Training Camp only (guarded) [M, CMA] |
| Perception: detector | YOLOv8-l x2 (or x3) on 150 classes, synthetic training data [R, Table 2] | none; Claude vision (Haiku for players) reads the screenshot [R-code] | KataCR YOLOv8 d1 + d2 weights unchanged, MPS [M] |
| Perception: other | PaddleOCR, ResNet card and elixir classifiers (single deck), ByteTrack in FPS(T) [R, R-code] | none; prompt hints for the elixir bar and tower states [R] | OCR (rapidocr), Kalman tracker, hand tracker, template card matcher [M, PBA/CMA] |
| Perception accuracy | mAP 67.4, AP50 84.3, mAP(S) 43.9 (x2) on 6,939 real images [R] | elixir 25/25, hand and threat reads partial; tower detection 3/7 on 25 screenshots [R] | no own detector metric; live-state parity: phase 0.62, towers 0.77, elixir 0.55, hand 0.57, units Jaccard 0.34, median cosine 0.6714 (one game) [M]; tracker ETA error 1.49 s -> 0.30 s (n = 384) [M] |
| Perception throughput | 34 FPS (4090) / 18 FPS (4060 Laptop + ByteTrack), 23 / 10 FPS for x3 [R] | NR (an LLM call per screenshot) | detector 100.5 ms mean per call; whole perception p50 102.4 / p95 230.0 ms; source 7.97 fps of 10 target [M] |
| Decision maker | StARformer-3L / DT-4L offline-RL transformer, imitates one human [R] | Claude Haiku (players) per screenshot, plus a Claude Code Commander [R-code] | rules + LightGBM "act now?" gate, tactical engine, then fine-tuned 600M LLM (d1) picks 1 of 4 candidates [M] |
| Action space | continuous (x, y) on 18 x 32 grid + card index + delay [R] | card slot 1-4 + cell of an 8 x 8 grid, free choice [R-code] | choose 1 of at most 4 candidate (card, placement) plays, up to 2 from the engine, the rest from kNN/BM25 retrieval, KG and fallback [M] |
| Knowledge sources, retrieval | none at inference; knowledge lives in policy weights [R] | hand-written prompts and persistent notes (`memory/*`); no retrieval [R] | 10,923 pro states: exact kNN over 644-d vectors + BM25, RRF k = 60; knowledge graph (KG v1/v2, up to 3 `KG:` lines); card-role table of 122 cards; tactics notes [M] |
| Training data | 105 episodes (124 in `UPDATE_LOGS`) of human vs AI, 113,981 frames, fixed decks; detector: 4,654 slices -> 20,000 generated images [R] | none (prompting); 25 labeled screenshots for eval only [R] | 197 pro games from 33 videos by one creator, about 12 match-hours; ranker set 3,641 / 761 / 766; detector not trained by us [M] |
| Training compute | NR beyond single GPU | none | d1 fine-tunes on M5 MPS: 13,241 s then 4,554 s (shipped model), peak 22.75 GB [M] |
| Action latency | 120 ms decision + 240 ms fusion (about 360 ms if serial, I) [R] | about 7 s per agent; 3 agents -> card every 2-3 s; eval harness 15-20 s per screenshot (Haiku) [R] | p50 338.9 / p95 571.8 ms capture to click (execute alone 183.3 / 188.1); all 931 decisions p50 347 / p95 613 ms [M] |
| Decision frequency | every 2nd frame, delay <= 8 frames [R-code] | one play per agent per about 7 s; scripted opener at 5 s and 8 s [R] | source 7.97 fps of 10 target (run 20261107-111554); 125 decisions in 7,169 frames; 746 of 931 logged decisions (all runs) sent a tap [M] |
| Actuation | ADB tap [R-code] | `cliclick` two-click (card then cell) [R-code] | Quartz clicks to iPhone Mirroring; 95/100 taps landed (run 20261106-233907), 86% in the earlier MAC-AF run [M] |
| Hardware | HarmonyOS phone + Ubuntu laptop, Ryzen 9 7940H, RTX 4060 Laptop [R] | a Mac + BlueStacks; cloud LLM; exact Mac NR [R, NR] | Apple M5, 32 GB, macOS 26.5, MPS; d1 run through `transformers` [M] |
| Cost per game | NR | NR | no paid API in the live path (all models local) [M, design fact] |
| Evaluation protocol | real phone, built-in AI, 20 rounds per checkpoint, checkpoint chosen as best of the first 10 epochs on the same environment [R] | (a) live ladder vs humans, trophy trajectory, per-session W-L notes; (b) 25 labeled screenshots, rubric, 3 Claude models [R] | (a) offline: by-video split 70/15/15, 4-way ranker on val 761 / test 766; (b) live Training Camp runs with per-decision logs (13 runs) [M]; (c) planned: about 10 Training Camp games |
| Offline decision metric | none vs expert actions; reward and win only [R] | card choice 11/25 = 44% (Haiku, labels by the author) [R] | 4-way accuracy val 0.5401 / test 0.4974 vs random 0.25, placement prior 0.4928 / 0.3838, LightGBM LambdaRank 0.5484 / 0.4913 [M] |
| **Win rate vs built-in AI / Training Camp (n games)** | **5% (1/20) headline model; 0-10% over 9 models; 95% CI for 1/20: 0.1-24.9%** [R, CI computed by me] | **NR; opponents are humans** | **TBD (n = TBD); 1 win in the first 3 games of run 20261107-013441 [M]** |
| Win rate vs humans | NR | no aggregate; Session 38: 4W-9L-1D; 1723 trophies [R] | not attempted (Training Camp only) [M] |
| Other metrics | total reward -4.7 +/- 3.1 (best), game duration 158.9, actions per game 207.8 +/- 48.2 [R] | trophies, W-L per session, inter-action gap, cards per minute (`analyze-latency.sh`) [R] | executed-play sources (engine-only 47, engine+kNN+KG 25, kNN+KG 16, fallback 7 in run 20261107-111554); ability taps fired 6/6 in run 20261106-233907 [M] |
| Code and weights public | yes, MIT; Drive weights; datasets on GitHub [R] | yes, MIT; no weights needed; not runnable without recalibration [R] | repo local; weights not yet hosted (PBA blocker 2) [M] |
| Limits stated | fixed decks, 0-10% win, heavy human effort for data [R] | latency, hard-coded window, no aggregate stats [R] | live win rate with variance not measured; perception parity weak; legality check passes only 0.54-0.60 of the pros' own moves [M, CMA] |

---

## 5. Where the protocols differ (read before comparing numbers)

| issue | consequence for the comparison |
|---|---|
| Opponent: Training Camp AI (KataCR, ours) vs human ladder (Claude Royale) | Win rates are not comparable between [2] and the other two. Trophies mix matchmaking and skill. Only KataCR and ours may share a "vs built-in AI" column |
| Sample size: 20 per checkpoint (KataCR) vs about 10 (ours) | Exact 95% intervals (Clopper-Pearson): 1/20 = 0.1-24.9%, 2/20 = 1.2-31.7%, 0/20 = 0-16.8%; ours 1/10 = 0.3-44.5%, 3/10 = 6.7-65.2%, 5/10 = 18.7-81.3%. One-sided Fisher exact test of our k/10 vs their 1/20: k = 3 p = 0.095, k = 4 p = 0.031, k = 5 p = 0.009. Do not claim superiority with fewer than 4 wins of 10 |
| Selection on the test environment (KataCR) | Their row is best-of-10-epochs by reward on the same 20-round evaluation, so 5% is an optimistic estimate of their typical checkpoint (the other eight rows are 0-10% too) |
| Same environment is not the same opponent | "Built-in AI with 8000 rating" is not defined in the paper; whether it equals today's Training Camp opponent is unknown (game version: Season 59, May 2024, vs Nov 2026) |
| Deck | KataCR trains and tests on fixed decks (Golem / 2.6 Hog), Claude Royale changed decks across sessions (Giant, Fire Spirit, Hog cycle), ours uses the user's deck. Deck strength is confounded with agent skill in every win rate |
| Offline metrics measure different things | KataCR: environment reward (its own shaped reward, Eq. 2). Claude Royale: agreement with one author's labels on 25 screenshots with partial credit. Ours: top-1 accuracy at picking the pro's play among 4 constructed options (random 0.25). Do not put 0.4974, 44% and -4.7 in one chart axis |
| Latency definitions | KataCR: decision (120 ms) and fusion (240 ms) reported separately, actuation excluded. Claude Royale: screenshot to tool call, per agent, with 3 agents staggered. Ours: capture to click, including 184 ms of tap sequence. A fair statement is "about 0.36 s (KataCR, if serial), about 0.34 s p50 (ours), about 7 s per agent (Claude Royale)" with these caveats |
| Throughput measured on different silicon | RTX 4090 / 4060 Laptop vs Apple M5 MPS; ours is a different crop pipeline and includes a tracker and hand reader, and the weights are the same family |
| Information available to the agent | all three are non-embedded (pixels). Ours additionally uses a pro-play database, a knowledge graph and an engine; KataCR and Claude Royale do not. State this as a design difference, not as a like-for-like model comparison |
| Ground truth for "good play" | KataCR learns from one human's moves vs AI; ours from 197 pro games (one creator, 82.2% pro win rate in the data); Claude Royale's labels are the author's own |

---

## 6. Wording that stays defensible

| claim | evidence | safe to write |
|---|---|---|
| Latency class | ours 338.9 / 571.8 ms (M), KataCR about 360 ms (I from R), Claude Royale about 7 s (R) | "comparable to KataCR's reported latency and about 20x faster than the LLM-per-screenshot harness" (7 s / 0.34 s is about 21x; per-agent, not effective) |
| Perception reuse | our weights are KataCR's (I, strongly supported by size and Drive ids) | "we reuse KataCR's published YOLOv8 detector weights and do not claim a new detector" |
| Offline accuracy | 0.4974 vs random 0.25 vs placement prior 0.3838 (test) | "a 4-way pick-the-pro-move accuracy, not a win rate" |
| Win rate | TBD | report wins / n with an exact interval; compare with KataCR's 1/20 only on the "built-in AI" row and state that decks and opponent versions differ |
| Claude Royale | trophy ladder vs humans | "a different protocol; cited for the LLM-agent design and its 7 s latency, not as a benchmark" |

---

## 7. BibTeX

The proposal cites the arXiv version, so the first entry is that preprint; the second is the published version (same paper). Use one of the two, not both.

```bibtex
@misc{wu2025nonembedded,
  author        = {Wu, Tianyang and Wan, Lipeng and Wang, Yuhang and Wan, Qiang and Lan, Xuguang},
  title         = {Playing Non-Embedded Card-Based Games with Reinforcement Learning},
  year          = {2025},
  eprint        = {2504.04783},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2504.04783},
  note          = {Submitted 7 April 2025; code: https://github.com/wty-yy/katacr}
}

@inproceedings{wu2025nonembedded_icira,
  author    = {Wu, Tianyang and Wan, Lipeng and Wang, Yuhang and Wan, Qiang and Lan, Xuguang},
  title     = {Playing Non-Embedded Card-Based Games with Reinforcement Learning},
  booktitle = {Intelligent Robotics and Applications (ICIRA 2024)},
  series    = {Lecture Notes in Computer Science},
  volume    = {15206},
  pages     = {255--269},
  publisher = {Springer Nature Singapore},
  address   = {Singapore},
  year      = {2025},
  doi       = {10.1007/978-981-96-0792-1_20},
  note      = {arXiv:2504.04783}
}

@misc{houseworth2025claudeRoyale,
  author       = {Houseworth, Ethan},
  title        = {{Claude Royale}: {AI} Agents Playing {Clash Royale} Autonomously},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/houseworthe/claude-royale},
  note         = {Created 2025-12-07; last commit 2026-01-25 (70ecc8c); MIT licence; accessed 2026-11-07}
}
```

Verification notes: [1] title, five authors, date and DOI are from the arXiv `citation_*` meta tags; volume 15206 from the arXiv journal-reference line; pages 255-269 and publisher from Crossref. Editors of the proceedings were not returned and are omitted. [2] has no paper or formal title; the title string is the README H1 plus the GitHub description; the author's name comes from the GitHub profile (the proposal wrote "E. Houseworth"). The year 2025 matches the creation date; the repo kept changing until 2026-01-25.

---

## 8. Sources

| source | link |
|---|---|
| [1] arXiv abstract | https://arxiv.org/abs/2504.04783 |
| [1] arXiv HTML (read in full) | https://arxiv.org/html/2504.04783 |
| [1] PDF | https://arxiv.org/pdf/2504.04783 |
| [1] DOI (Springer) | https://doi.org/10.1007/978-981-96-0792-1_20 |
| [1] code | https://github.com/wty-yy/katacr (local clone `~/cr_scratch/katacr`, README_en.md, UPDATE_LOGS.md, `katacr/policy/env/sar_daemon.py`, `katacr/policy/env/utils.py`) |
| [1] datasets | https://github.com/wty-yy/Clash-Royale-Detection-Dataset, https://github.com/wty-yy/Clash-Royale-Replay-Dataset |
| [1] match video | https://www.bilibili.com/video/BV1xn4y1R7GQ |
| [2] repo | https://github.com/houseworthe/claude-royale |
| [2] README | https://github.com/houseworthe/claude-royale/blob/main/README.md |
| [2] model comparison | https://github.com/houseworthe/claude-royale/blob/main/eval/MODEL_COMPARISON.md |
| [2] eval design and rubric | https://github.com/houseworthe/claude-royale/blob/main/eval/EVAL_DESIGN.md, https://github.com/houseworthe/claude-royale/blob/main/eval/grading.json |
| [2] status and learnings | https://github.com/houseworthe/claude-royale/blob/main/memory/STATUS.md, https://github.com/houseworthe/claude-royale/blob/main/memory/LEARNINGS.md |
| [2] agents and scripts | https://github.com/houseworthe/claude-royale/blob/main/.claude/agents/player-classic.md, https://github.com/houseworthe/claude-royale/blob/main/scripts/play_card.sh, https://github.com/houseworthe/claude-royale/blob/main/scripts/screenshot.sh |
| Ours | `docs/research/course_mapping_audit.md` (sections C1, C3, C4, C6, C7), `docs/research/packaging_and_benchmarks_audit.md` (sections 1.0, 1.1) |
