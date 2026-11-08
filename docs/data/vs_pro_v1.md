# The bot against the pros on their own videos (open loop)

Generated 2026-11-08T02:31:35+00:00 by `tools/eval/eval_vs_pro.py --score`. Code: HEAD at the runs `17578314d2`, `1bb98a4b7d`, `54243d78bc`, `701a1913e0`, `afbe4a2a24`, `fff4b64549` (only evaluation, dashboard and test files differ between them), decision code frozen at `66917e7` (changed decision files: clashbot/demo.py). 24 games, 90 min of pro video, 1561 pro plays, 2042 bot taps. Confidence intervals: 95%, bootstrap over games (2000 resamples).

## Headline

| metric | bot | 95% CI | n | baseline | baseline value |
|---|---|---|---|---|---|
| pro-play recall (a bot tap within 2 s of the pro play) | 77.1% | 71.8% to 83.0% | 1561 | same taps shifted 20-60 s (null) | 68.9% (63.6% to 74.5%) |
| bot-tap precision (a pro play within 2 s of the tap) | 72.6% | 69.3% to 76.2% | 2042 | same taps shifted 20-60 s (null) | 67.9% (65.3% to 70.8%) |
| taps per minute | 22.8 | 20.18 to 25.86 | 89 min | the pro's plays per minute | 17.4 (16.50 to 18.38) |
| same card, on matched pairs | 17.1% | 14.6% to 20.2% | 725 | random card of the bot's perceived hand | 12.8% (11.6% to 14.1%) |
| same card, on the pairs where a KG threat exists | 19.8% | 13.3% to 26.9% | 126 | KG v1 `counter_scores` first among the bot's perceived hand | 15.6% (8.8% to 23.1%) |
| ceiling: the pro's card is in the hand the bot perceived | 40.1% | 37.3% to 43.2% | 1130 | in the hand the dataset read | 53.0% (49.0% to 57.3%) |
| same lane, given the same card | 42.7% | 31.6% to 53.5% | 103 | placement prior's lane for the pro's card | 44.6% |
| same zone, given the same card | 52.4% | 45.2% to 60.0% | 103 | placement prior's zone for the pro's card | 56.4% |
| same key (card, lane, zone), given the same card | 26.2% | 16.8% to 36.0% | 103 | placement prior's key for the pro's card | 29.7% |
| same key, on matched pairs | 4.5% | 2.7% to 6.7% | 594 | the prior's most common key overall | 1.7% |

Baselines are evaluated on the same pro plays as the bot (the matched pairs). The hand is the one the bot perceived at that moment (its latest hold or decision row, at most 3 s old); the rows marked dataset use the hand the dataset's offline pipeline read, and the threats always come from the dataset's state. The ceiling row shows how often the pro's card was in the perceived hand at all: the bot cannot agree on a card it does not see. The placement-prior rows give the predictor the pro's card, which the bot does not get.

## Timing

| metric | value | 95% CI |
|---|---|---|
| recall +-1 s | 58.9% | 53.5% to 65.2% |
| recall +-2 s | 77.1% | 71.8% to 83.0% |
| recall +-3 s | 84.5% | 79.9% to 89.2% |
| precision +-1 s | 47.6% | 45.0% to 50.6% |
| precision +-2 s | 72.6% | 69.3% to 76.2% |
| precision +-3 s | 85.7% | 82.7% to 89.0% |
| recall +-2 s, pro stamp shifted -0.5 s | 77.1% | 71.9% to 82.8% |
| precision +-2 s, pro stamp shifted -0.5 s | 72.8% | 69.7% to 76.2% |
| bot acts earlier than the pro | 52.9% | 50.2% to 55.9% |
| median abs time difference of matched pairs | 0.47 s | 0.43 to 0.50 s |
| median signed difference (bot minus pro; negative = the bot is earlier) | -0.07 s | -0.11 to 0.00 s |
| bot taps per minute | 22.77 | 20.18 to 25.86 |
| pro plays per minute | 17.41 | 16.50 to 18.38 |
| bot elixir per minute | 58.72 | 51.28 to 67.67 |
| pro elixir per minute | 33.58 | 29.62 to 37.63 |

## Choice on matched pairs

| metric | bot | 95% CI | n |
|---|---|---|---|
| same card (matched pairs) | 17.1% | 14.6% to 20.2% | 725 |
| same lane (matched pairs) | 43.6% | 40.0% to 47.6% | 594 |
| same zone (matched pairs) | 35.5% | 32.1% to 39.4% | 594 |
| same key (matched pairs) | 4.5% | 2.7% to 6.7% | 594 |
| same lane given same card | 42.7% | 31.6% to 53.5% | 103 |
| same zone given same card | 52.4% | 45.2% to 60.0% | 103 |
| same key given same card | 26.2% | 16.8% to 36.0% | 103 |
| within 2 tiles, same card | 8.7% | 2.9% to 16.0% | 103 |
| mean tile distance, same card | 7.80 tiles | 6.73 to 8.80 | 103 |

## Baselines at the pro plays

| baseline | on matched pairs | on pro plays the bot covered (+-2 s) | on all pro plays |
|---|---|---|---|
| random card of the bot's perceived hand (expected same card) | 12.8% (n=725) | 12.4% (n=883) | 12.5% (n=1130) |
| KG v1 counter_scores first in the bot's perceived hand (if a threat exists) | 15.6% (n=126) | 14.4% (n=137) | 14.2% (n=167) |
| random card of the dataset's hand (the pipeline's offline read) | 18.9% (n=570) | 18.7% (n=662) | 18.2% (n=864) |
| KG v1 counter_scores first in the dataset's hand (if a threat exists) | 21.6% (n=111) | 21.5% (n=121) | 22.9% (n=148) |
| placement prior's most common key (same key) | 1.7% (n=594) | 1.5% (n=716) | 1.2% (n=917) |
| placement prior, pro's card: same lane | 46.4% (n=577) | 45.8% (n=696) | 44.9% (n=895) |
| placement prior, pro's card: same zone | 57.7% (n=577) | 57.5% (n=696) | 56.8% (n=895) |
| placement prior, pro's card: same key | 29.8% (n=577) | 28.6% (n=696) | 27.5% (n=895) |

## By the bot's mode

| mode | bot taps | precision +-2 s | matched pairs | same card | n |
|---|---|---|---|---|---|
| counter | 328 | 55.8% | 119 | 23.2% | 95 |
| defend | 1308 | 76.1% | 660 | 16.3% | 484 |
| leak | 28 | 75.0% | 14 | 8.3% | 12 |
| learned | 315 | 73.3% | 164 | 13.8% | 116 |
| punish | 62 | 80.6% | 27 | 33.3% | 18 |
| spell | 1 | 100.0% | 1 | n/a | 0 |

## By the pro play's confidence (dataset detection)

| confidence | pro plays | recall +-2 s | matched pairs | same card | n |
|---|---|---|---|---|---|
| high | 655 | 77.9% | 417 | 13.7% | 256 |
| low | 101 | 76.2% | 58 | n/a | 0 |
| medium | 805 | 76.6% | 510 | 19.0% | 469 |

## By game

| game | result | min | pro plays | bot taps | recall +-2 s | precision +-2 s | same card |
|---|---|---|---|---|---|---|---|
| 5nOaTBlYLlg-m0.0 | win | 4.96 | 95 | 130 | 74.7% | 83.8% | 7.8% |
| 5nOaTBlYLlg-m0.1 | win | 9.0 | 136 | 170 | 70.6% | 70.0% | 17.1% |
| 5nOaTBlYLlg-m0.2 | win | 3.79 | 67 | 31 | 38.8% | 64.5% | 15.4% |
| 5nOaTBlYLlg-m0.3 | loss | 4.42 | 88 | 122 | 84.1% | 80.3% | 12.8% |
| 5nOaTBlYLlg-m0.4 | win | 5.41 | 86 | 105 | 70.9% | 67.6% | 21.1% |
| 5nOaTBlYLlg-m0.5 | win | 1.43 | 20 | 31 | 85.0% | 67.7% | 25.0% |
| 6qYQNJ1Uaeg-m0.0 | win | 2.85 | 46 | 72 | 84.8% | 69.4% | 12.5% |
| 6qYQNJ1Uaeg-m0.1 | win | 2.95 | 42 | 71 | 85.7% | 69.0% | 16.7% |
| 6qYQNJ1Uaeg-m0.2 | win | 2.12 | 34 | 69 | 97.1% | 63.8% | 20.8% |
| 6qYQNJ1Uaeg-m0.3 | win | 3.52 | 57 | 62 | 78.9% | 75.8% | 17.1% |
| 6qYQNJ1Uaeg-m0.4 | win | 5.0 | 105 | 147 | 91.4% | 84.4% | 16.7% |
| 6qYQNJ1Uaeg-m0.5 | win | 2.0 | 31 | 74 | 93.5% | 62.2% | 19.0% |
| 6qYQNJ1Uaeg-m0.6 | unknown | 3.22 | 52 | 78 | 86.5% | 78.2% | 30.0% |
| SsOmbv1PfUg-m0.0 | win | 2.88 | 47 | 57 | 87.2% | 75.4% | 23.1% |
| SsOmbv1PfUg-m0.1 | win | 2.13 | 32 | 48 | 78.1% | 81.2% | 17.6% |
| SsOmbv1PfUg-m0.2 | win | 4.83 | 91 | 106 | 81.3% | 68.9% | 19.4% |
| SsOmbv1PfUg-m0.3 | win | 2.08 | 38 | 25 | 65.8% | 72.0% | 36.4% |
| SsOmbv1PfUg-m0.4 | loss | 4.95 | 96 | 77 | 69.8% | 76.6% | 11.1% |
| SsOmbv1PfUg-m0.5 | win | 4.91 | 99 | 74 | 54.5% | 79.7% | 8.1% |
| Zt6Onn89EVk-m0.0 | win | 4.53 | 89 | 113 | 71.9% | 68.1% | 17.9% |
| Zt6Onn89EVk-m0.1 | win | 3.0 | 42 | 57 | 71.4% | 71.9% | 12.5% |
| Zt6Onn89EVk-m0.2 | win | 3.55 | 57 | 145 | 100.0% | 62.1% | 8.6% |
| Zt6Onn89EVk-m1.0 | win | 1.27 | 29 | 56 | 96.6% | 80.4% | 30.0% |
| Zt6Onn89EVk-m1.1 | win | 4.89 | 82 | 122 | 86.6% | 64.8% | 25.6% |

## When the bot held while the pro played (357 pro plays with no bot tap within 2 s)

| what the bot was doing (mode: reason) | pro plays | share |
|---|---|---|
| none: no affordable card | 82 | 23.0% |
| save: the tower handles it | 81 | 22.7% |
| hold: act model: wait (p below threshold) | 67 | 18.8% |
| decision held: hold_policy | 54 | 15.1% |
| save: no adequate answer in hand | 29 | 8.1% |
| save: push already answered | 18 | 5.0% |
| save: waiting for a card's window | 10 | 2.8% |
| decision held: hold_fallback | 9 | 2.5% |
| save: saving for a card | 5 | 1.4% |
| not in a match frame (readiness menu) | 2 | 0.6% |

## Examples for the slides

**1. AGREE** game 5nOaTBlYLlg-m0.3, video t = 1251.2 s (bot tap at 1251.4 s, difference +0.1 s)
- pro: cannon|middle|mid at tile [9, 9] (detection confidence high)
- bot: cannon|middle|mid at tile [8.0, 10.0], mode punish: fast act p=0.54 h=2s
- d1 over the bot's candidates: cannon|middle|mid (1.00)
- screenshot: `runs/eval/vs_pro/5nOaTBlYLlg-m0.3/shots/d37504_t1251.4_a.jpg`

**2. AGREE** game Zt6Onn89EVk-m1.1, video t = 1027.8 s (bot tap at 1027.5 s, difference -0.4 s)
- pro: ice-spirit|left|bridge at tile [3, 13] (detection confidence medium)
- bot: ice-spirit|left|bridge at tile [3.5, 14.0], mode learned: fast act p=0.57 h=2s
- d1 over the bot's candidates: ice-spirit|left|bridge (0.62), ice-spirit|left|mid (0.38)
- screenshot: `runs/eval/vs_pro/Zt6Onn89EVk-m1.1/shots/d30793_t1027.5_a.jpg`

**3. AGREE** game 6qYQNJ1Uaeg-m0.1, video t = 222.2 s (bot tap at 221.8 s, difference -0.5 s)
- pro: bomb-tower|middle|mid at tile [8, 9] (detection confidence medium)
- bot: bomb-tower|middle|mid at tile [8.0, 10.0], mode defend: defend monk left: bomb-tower (high)
- d1 over the bot's candidates: bomb-tower|middle|mid (1.00)
- screenshot: `runs/eval/vs_pro/6qYQNJ1Uaeg-m0.1/shots/d6646_t221.8_a.jpg`

**4. DISAGREE** game 6qYQNJ1Uaeg-m0.4, video t = 738.6 s (bot tap at 737.1 s, difference -1.5 s)
- pro: bats at tile None (detection confidence high)
- bot: miner|right|enemy_half at tile [13, 20], mode learned: fast act p=0.53 h=2s
- d1 over the bot's candidates: miner|left|enemy_half (0.22), spear-goblins|right|mid (0.25), bats|right|back (0.02), miner|right|enemy_half (0.52)
- screenshot: `runs/eval/vs_pro/6qYQNJ1Uaeg-m0.4/shots/d22091_t737.1_a.jpg`

**5. DISAGREE** game 6qYQNJ1Uaeg-m0.3, video t = 713.9 s (bot tap at 713.9 s, difference -0.0 s)
- pro: berserker|middle|bridge at tile [7, 14] (detection confidence high)
- bot: spear-goblins|right|mid at tile [14.5, 3.0], mode defend: defend bats right: spear-goblins (emergency)
- d1 over the bot's candidates: spear-goblins|right|back (0.20), spear-goblins|right|mid (0.80)
- screenshot: `runs/eval/vs_pro/6qYQNJ1Uaeg-m0.3/shots/d21395_t713.9_a.jpg`

**6. DISAGREE** game 6qYQNJ1Uaeg-m0.6, video t = 1256.6 s (bot tap at 1256.9 s, difference +0.3 s)
- pro: spear-goblins|middle|bridge at tile [10, 14] (detection confidence high)
- bot: spear-goblins|right|bridge at tile [14.5, 11.5], mode defend: defend ice-wizard right: spear-goblins (medium)
- d1 over the bot's candidates: spear-goblins|right|bridge (0.57), spear-goblins|right|mid (0.43)
- screenshot: `runs/eval/vs_pro/6qYQNJ1Uaeg-m0.6/shots/d37668_t1256.9_a.jpg`

## Games

| game | video min | wall s | x real time | frames | decisions | calib | hero | evo |
|---|---|---|---|---|---|---|---|---|
| 5nOaTBlYLlg-m0.0 | 5.0 | 373.6 | 1.25 | 2975 | 137 | runs/videos/5nOaTBlYLlg/match_0/calib.json | - | archers, royal-hogs |
| 5nOaTBlYLlg-m0.1 | 9.0 | 621.1 | 1.15 | 5394 | 223 | runs/videos/5nOaTBlYLlg/match_0/calib.json | valkyrie | - |
| 5nOaTBlYLlg-m0.2 | 3.8 | 255.3 | 1.12 | 2274 | 37 | runs/videos/5nOaTBlYLlg/match_0/calib.json | - | - |
| 5nOaTBlYLlg-m0.3 | 4.4 | 305.5 | 1.15 | 2648 | 141 | runs/videos/5nOaTBlYLlg/match_0/calib.json | - | - |
| 5nOaTBlYLlg-m0.4 | 5.4 | 847.0 | 2.61 | 3242 | 118 | runs/videos/5nOaTBlYLlg/match_0/calib.json | dark-prince | - |
| 5nOaTBlYLlg-m0.5 | 1.4 | 101.9 | 1.19 | 856 | 33 | runs/videos/5nOaTBlYLlg/match_0/calib.json | valkyrie | - |
| 6qYQNJ1Uaeg-m0.0 | 2.8 | 196.6 | 1.15 | 1708 | 75 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| 6qYQNJ1Uaeg-m0.1 | 3.0 | 195.6 | 1.10 | 1770 | 78 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| 6qYQNJ1Uaeg-m0.2 | 2.1 | 145.8 | 1.15 | 1268 | 69 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| 6qYQNJ1Uaeg-m0.3 | 3.5 | 235.1 | 1.11 | 2108 | 64 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| 6qYQNJ1Uaeg-m0.4 | 5.0 | 353.4 | 1.18 | 2998 | 158 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| 6qYQNJ1Uaeg-m0.5 | 2.0 | 183.3 | 1.53 | 1199 | 79 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| 6qYQNJ1Uaeg-m0.6 | 3.2 | 199.3 | 1.03 | 1931 | 84 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.0 | 2.9 | 201.6 | 1.17 | 1728 | 64 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.1 | 2.1 | 133.9 | 1.05 | 1279 | 49 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.2 | 4.8 | 336.2 | 1.16 | 2899 | 116 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.3 | 2.1 | 132.7 | 1.06 | 1249 | 29 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.4 | 4.9 | 3700.5 | 12.47 | 2969 | 81 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.5 | 4.9 | 338.9 | 1.15 | 2949 | 87 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| Zt6Onn89EVk-m0.0 | 4.5 | 326.0 | 1.20 | 2718 | 117 | runs/videos/Zt6Onn89EVk/match_0/calib.json | valkyrie | - |
| Zt6Onn89EVk-m0.1 | 3.0 | 192.9 | 1.07 | 1799 | 66 | runs/videos/Zt6Onn89EVk/match_0/calib.json | valkyrie | - |
| Zt6Onn89EVk-m0.2 | 3.6 | 226.2 | 1.06 | 2130 | 148 | runs/videos/Zt6Onn89EVk/match_0/calib.json | valkyrie | goblin-barrel, skeleton-army |
| Zt6Onn89EVk-m1.0 | 1.3 | 99.1 | 1.30 | 763 | 56 | runs/videos/Zt6Onn89EVk/match_0/calib.json | valkyrie | goblin-barrel, skeleton-army |
| Zt6Onn89EVk-m1.1 | 4.9 | 320.9 | 1.09 | 2930 | 138 | runs/videos/Zt6Onn89EVk/match_0/calib.json | valkyrie | - |

Not run: CInrqMTlVkg-m0.0 (deck incomplete (7 cards): the stack needs the 8-card deck); CInrqMTlVkg-m0.1 (deck incomplete (7 cards): the stack needs the 8-card deck); CInrqMTlVkg-m0.2 (deck invalid (5 cards): the stack needs the 8-card deck); CInrqMTlVkg-m0.3 (deck incomplete (7 cards): the stack needs the 8-card deck).

## How to read this (caveats)

Open loop: the bot's taps do not change the video, so each tap is what the bot would do in the pro's situation, not a game it played; after the pro's own play the board already shows the pro's units, which the bot also sees. Tap density: the bot taps more often than the pro (see taps per minute), so timing recall is read against the null row (the same taps moved 20-60 s), not against zero. Perception ceiling: the pro's card is in the hand the bot perceived in only the share the ceiling row gives, so same-card agreement cannot be high; in open loop the bot's hand memory is never told about a play (taps are not sent), which may explain a stale hand (not verified here). Calibration: each game uses its own video's calibration (`runs/videos/<video>/match_0/calib.json`), because `runs/poc/mirror_calib.json` puts the arena up to 2-4.5 tiles off on these videos. Perception noise: the bot perceives the screen with the same detector it uses live, and the pro's plays come from the dataset's offline pipeline (card labels on 'medium' / 'low' confidence plays and 'unknown'-card plays are noisy; a card outside the pro's deck is kept for timing and dropped from the choice metrics). Time stamps: the dataset stamps a play when the card lands on the arena, about 0.5 s after the pro's tap, so the bot looks slightly early; the shifted rows show the effect. Hidden information: the pro knows his own next cards, the opponent's real elixir and his plan, and plays for the match result; the bot sees only the screen and acts only when its act model says so, so low recall is not an error by itself and a different card is not always a worse one. Retrieval was limited to the train and val splits so the bot never retrieves the very state the pro played; the KG v2 answers and the placement prior are mined from train games only.
