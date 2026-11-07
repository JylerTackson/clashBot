# The bot against the pros on their own videos (open loop)

Generated 2026-11-07T23:50:59+00:00 by `tools/eval/eval_vs_pro.py --score`. Code: HEAD at the runs `1bb98a4b7d`, `54243d78bc` (only evaluation, dashboard and test files differ between them), decision code frozen at `66917e7` (changed decision files: none). 8 games, 31 min of pro video, 516 pro plays, 717 bot taps. Confidence intervals: 95%, bootstrap over games (2000 resamples).

## Headline

| metric | bot | 95% CI | n | baseline | baseline value |
|---|---|---|---|---|---|
| pro-play recall (a bot tap within 2 s of the pro play) | 77.5% | 73.3% to 85.0% | 516 | same taps shifted 20-60 s (null) | 70.4% (66.0% to 76.3%) |
| bot-tap precision (a pro play within 2 s of the tap) | 74.1% | 70.1% to 79.4% | 717 | same taps shifted 20-60 s (null) | 66.6% (62.7% to 71.3%) |
| taps per minute | 23.5 | 20.72 to 27.48 | 30 min | the pro's plays per minute | 16.9 (15.44 to 18.66) |
| same card, on matched pairs | 16.5% | 12.3% to 21.2% | 242 | random card of the bot's perceived hand | 12.6% (10.4% to 15.5%) |
| same card, on the pairs where a KG threat exists | 17.1% | 7.1% to 29.6% | 41 | KG v1 `counter_scores` first among the bot's perceived hand | 20.7% (5.3% to 37.8%) |
| ceiling: the pro's card is in the hand the bot perceived | 40.4% | 36.3% to 46.1% | 354 | in the hand the dataset read | 52.5% (45.3% to 63.8%) |
| same lane, given the same card | 36.4% | 20.5% to 60.0% | 33 | placement prior's lane for the pro's card | 39.4% |
| same zone, given the same card | 51.5% | 38.9% to 64.7% | 33 | placement prior's zone for the pro's card | 51.5% |
| same key (card, lane, zone), given the same card | 18.2% | 6.4% to 36.0% | 33 | placement prior's key for the pro's card | 24.2% |
| same key, on matched pairs | 3.2% | 1.0% to 5.8% | 190 | the prior's most common key overall | 0.5% |

Baselines are evaluated on the same pro plays as the bot (the matched pairs). The hand is the one the bot perceived at that moment (its latest hold or decision row, at most 3 s old); the rows marked dataset use the hand the dataset's offline pipeline read, and the threats always come from the dataset's state. The ceiling row shows how often the pro's card was in the perceived hand at all: the bot cannot agree on a card it does not see. The placement-prior rows give the predictor the pro's card, which the bot does not get.

## Timing

| metric | value | 95% CI |
|---|---|---|
| recall +-1 s | 62.0% | 57.8% to 69.2% |
| recall +-2 s | 77.5% | 73.3% to 85.0% |
| recall +-3 s | 86.4% | 82.7% to 92.7% |
| precision +-1 s | 48.8% | 45.1% to 53.6% |
| precision +-2 s | 74.1% | 70.1% to 79.4% |
| precision +-3 s | 86.1% | 82.6% to 90.6% |
| recall +-2 s, pro stamp shifted -0.5 s | 77.7% | 72.4% to 86.0% |
| precision +-2 s, pro stamp shifted -0.5 s | 73.4% | 69.3% to 78.9% |
| bot acts earlier than the pro | 54.5% | 49.6% to 60.0% |
| median abs time difference of matched pairs | 0.50 s | 0.43 to 0.50 s |
| median signed difference (bot minus pro; negative = the bot is earlier) | -0.10 s | -0.20 to 0.00 s |
| bot taps per minute | 23.45 | 20.72 to 27.48 |
| pro plays per minute | 16.88 | 15.44 to 18.66 |
| bot elixir per minute | 58.94 | 52.59 to 68.25 |
| pro elixir per minute | 32.51 | 25.38 to 41.89 |

## Choice on matched pairs

| metric | bot | 95% CI | n |
|---|---|---|---|
| same card (matched pairs) | 16.5% | 12.3% to 21.2% | 242 |
| same lane (matched pairs) | 42.6% | 34.6% to 53.5% | 190 |
| same zone (matched pairs) | 34.7% | 28.4% to 40.8% | 190 |
| same key (matched pairs) | 3.2% | 1.0% to 5.8% | 190 |
| same lane given same card | 36.4% | 20.5% to 60.0% | 33 |
| same zone given same card | 51.5% | 38.9% to 64.7% | 33 |
| same key given same card | 18.2% | 6.4% to 36.0% | 33 |
| within 2 tiles, same card | 9.1% | 2.1% to 22.7% | 33 |
| mean tile distance, same card | 7.98 tiles | 5.24 to 9.62 | 33 |

## Baselines at the pro plays

| baseline | on matched pairs | on pro plays the bot covered (+-2 s) | on all pro plays |
|---|---|---|---|
| random card of the bot's perceived hand (expected same card) | 12.6% (n=242) | 12.5% (n=281) | 12.4% (n=354) |
| KG v1 counter_scores first in the bot's perceived hand (if a threat exists) | 20.7% (n=41) | 19.8% (n=43) | 19.8% (n=48) |
| random card of the dataset's hand (the pipeline's offline read) | 20.4% (n=190) | 19.5% (n=215) | 18.1% (n=276) |
| KG v1 counter_scores first in the dataset's hand (if a threat exists) | 25.7% (n=35) | 24.3% (n=37) | 25.2% (n=41) |
| placement prior's most common key (same key) | 0.5% (n=190) | 0.5% (n=220) | 0.4% (n=278) |
| placement prior, pro's card: same lane | 48.1% (n=185) | 46.5% (n=215) | 46.9% (n=273) |
| placement prior, pro's card: same zone | 58.4% (n=185) | 57.7% (n=215) | 58.6% (n=273) |
| placement prior, pro's card: same key | 30.3% (n=185) | 28.4% (n=215) | 29.3% (n=273) |

## By the bot's mode

| mode | bot taps | precision +-2 s | matched pairs | same card | n |
|---|---|---|---|---|---|
| counter | 134 | 60.4% | 52 | 25.0% | 40 |
| defend | 449 | 77.3% | 217 | 14.1% | 156 |
| leak | 13 | 53.8% | 4 | 33.3% | 3 |
| learned | 105 | 79.0% | 59 | 13.9% | 36 |
| punish | 16 | 81.2% | 9 | 28.6% | 7 |

## By the pro play's confidence (dataset detection)

| confidence | pro plays | recall +-2 s | matched pairs | same card | n |
|---|---|---|---|---|---|
| high | 229 | 77.7% | 149 | 11.6% | 95 |
| low | 32 | 84.4% | 23 | n/a | 0 |
| medium | 255 | 76.5% | 169 | 19.7% | 147 |

## By game

| game | result | min | pro plays | bot taps | recall +-2 s | precision +-2 s | same card |
|---|---|---|---|---|---|---|---|
| 5nOaTBlYLlg-m0.0 | win | 4.96 | 95 | 130 | 74.7% | 83.8% | 7.8% |
| 5nOaTBlYLlg-m0.1 | win | 9.0 | 136 | 170 | 70.6% | 70.0% | 17.1% |
| 6qYQNJ1Uaeg-m0.0 | win | 2.85 | 46 | 72 | 84.8% | 69.4% | 12.5% |
| 6qYQNJ1Uaeg-m0.1 | win | 2.95 | 42 | 71 | 85.7% | 69.0% | 16.7% |
| SsOmbv1PfUg-m0.0 | win | 2.88 | 47 | 57 | 87.2% | 75.4% | 23.1% |
| SsOmbv1PfUg-m0.1 | win | 2.13 | 32 | 48 | 78.1% | 81.2% | 17.6% |
| Zt6Onn89EVk-m0.0 | win | 4.53 | 89 | 113 | 71.9% | 68.1% | 17.9% |
| Zt6Onn89EVk-m1.0 | win | 1.27 | 29 | 56 | 96.6% | 80.4% | 30.0% |

## When the bot held while the pro played (116 pro plays with no bot tap within 2 s)

| what the bot was doing (mode: reason) | pro plays | share |
|---|---|---|
| save: the tower handles it | 26 | 22.4% |
| hold: act model: wait (p below threshold) | 22 | 19.0% |
| decision held: hold_policy | 19 | 16.4% |
| none: no affordable card | 17 | 14.7% |
| save: no adequate answer in hand | 10 | 8.6% |
| decision held: hold_fallback | 8 | 6.9% |
| save: waiting for a card's window | 8 | 6.9% |
| save: push already answered | 4 | 3.4% |
| not in a match frame (readiness menu) | 1 | 0.9% |
| save: saving for a card | 1 | 0.9% |

## Examples for the slides

**1. AGREE** game 6qYQNJ1Uaeg-m0.1, video t = 222.2 s (bot tap at 221.8 s, difference -0.5 s)
- pro: bomb-tower|middle|mid at tile [8, 9] (detection confidence medium)
- bot: bomb-tower|middle|mid at tile [8.0, 10.0], mode defend: defend monk left: bomb-tower (high)
- d1 over the bot's candidates: bomb-tower|middle|mid (1.00)
- screenshot: `runs/eval/vs_pro/6qYQNJ1Uaeg-m0.1/shots/d6646_t221.8_a.jpg`

**2. AGREE** game SsOmbv1PfUg-m0.1, video t = 272.0 s (bot tap at 271.5 s, difference -0.5 s)
- pro: skeletons|middle|bridge at tile [11, 14] (detection confidence medium)
- bot: skeletons|middle|mid at tile [11.5, 14.0], mode defend: defend electro-wizard middle: skeletons (medium)
- d1 over the bot's candidates: skeletons|middle|bridge (0.31), skeletons|middle|mid (0.69)
- screenshot: `runs/eval/vs_pro/SsOmbv1PfUg-m0.1/shots/d8145_t271.5_a.jpg`

**3. AGREE** game 6qYQNJ1Uaeg-m0.0, video t = 35.0 s (bot tap at 34.5 s, difference -0.5 s)
- pro: wall-breakers|right|bridge at tile [13, 14] (detection confidence medium)
- bot: wall-breakers|right|bridge at tile [14, 14], mode punish: fast act p=0.61 h=2s
- d1 over the bot's candidates: wall-breakers|right|bridge (1.00)
- screenshot: `runs/eval/vs_pro/6qYQNJ1Uaeg-m0.0/shots/d1035_t34.5_a.jpg`

**4. DISAGREE** game Zt6Onn89EVk-m0.0, video t = 56.1 s (bot tap at 55.5 s, difference -0.6 s)
- pro: valkyrie|middle|mid at tile [8, 10] (detection confidence medium)
- bot: magic-archer|middle|back at tile [8.0, 1.0], mode learned: fast act p=0.61 h=2s
- d1 over the bot's candidates: magic-archer|middle|mid (0.23), magic-archer|middle|back (0.43), valkyrie|left|mid (0.34)
- screenshot: `runs/eval/vs_pro/Zt6Onn89EVk-m0.0/shots/d1662_t55.5_a.jpg`

**5. DISAGREE** game 5nOaTBlYLlg-m0.0, video t = 51.4 s (bot tap at 51.4 s, difference -0.0 s)
- pro: electro-spirit|right|bridge at tile [12, 14] (detection confidence high)
- bot: bomb-tower|middle|mid at tile [10.0, 10.0], mode defend: defend mother-witch right: bomb-tower (high)
- d1 over the bot's candidates: bomb-tower|middle|mid (1.00)
- screenshot: `runs/eval/vs_pro/5nOaTBlYLlg-m0.0/shots/d1539_t51.4_a.jpg`

**6. DISAGREE** game SsOmbv1PfUg-m0.0, video t = 22.3 s (bot tap at 21.7 s, difference -0.6 s)
- pro: electro-spirit|middle|bridge at tile [7, 12] (detection confidence high)
- bot: electro-spirit|left|mid at tile [10.5, 5.0], mode defend: defend inferno-dragon left: electro-spirit (emergency)
- d1 over the bot's candidates: electro-spirit|middle|mid (0.35), electro-spirit|left|mid (0.65)
- screenshot: `runs/eval/vs_pro/SsOmbv1PfUg-m0.0/shots/d0651_t21.7_a.jpg`

## Games

| game | video min | wall s | x real time | frames | decisions | calib | hero | evo |
|---|---|---|---|---|---|---|---|---|
| 5nOaTBlYLlg-m0.0 | 5.0 | 373.6 | 1.25 | 2975 | 137 | runs/videos/5nOaTBlYLlg/match_0/calib.json | - | archers, royal-hogs |
| 5nOaTBlYLlg-m0.1 | 9.0 | 621.1 | 1.15 | 5394 | 223 | runs/videos/5nOaTBlYLlg/match_0/calib.json | valkyrie | - |
| 6qYQNJ1Uaeg-m0.0 | 2.8 | 196.6 | 1.15 | 1708 | 75 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| 6qYQNJ1Uaeg-m0.1 | 3.0 | 195.6 | 1.10 | 1770 | 78 | runs/videos/6qYQNJ1Uaeg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.0 | 2.9 | 201.6 | 1.17 | 1728 | 64 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| SsOmbv1PfUg-m0.1 | 2.1 | 133.9 | 1.05 | 1279 | 49 | runs/videos/SsOmbv1PfUg/match_0/calib.json | berserker | - |
| Zt6Onn89EVk-m0.0 | 4.5 | 326.0 | 1.20 | 2718 | 117 | runs/videos/Zt6Onn89EVk/match_0/calib.json | valkyrie | - |
| Zt6Onn89EVk-m1.0 | 1.3 | 99.1 | 1.30 | 763 | 56 | runs/videos/Zt6Onn89EVk/match_0/calib.json | valkyrie | goblin-barrel, skeleton-army |

Not run: CInrqMTlVkg-m0.0 (deck incomplete (7 cards): the stack needs the 8-card deck); CInrqMTlVkg-m0.1 (deck incomplete (7 cards): the stack needs the 8-card deck); CInrqMTlVkg-m0.2 (deck invalid (5 cards): the stack needs the 8-card deck); CInrqMTlVkg-m0.3 (deck incomplete (7 cards): the stack needs the 8-card deck); Zt6Onn89EVk-m0.1 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 5nOaTBlYLlg-m0.2 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 6qYQNJ1Uaeg-m0.2 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); SsOmbv1PfUg-m0.2 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); Zt6Onn89EVk-m0.2 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 5nOaTBlYLlg-m0.3 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 6qYQNJ1Uaeg-m0.3 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); SsOmbv1PfUg-m0.3 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 5nOaTBlYLlg-m0.4 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 6qYQNJ1Uaeg-m0.4 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); SsOmbv1PfUg-m0.4 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); Zt6Onn89EVk-m1.1 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 5nOaTBlYLlg-m0.5 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 6qYQNJ1Uaeg-m0.5 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); SsOmbv1PfUg-m0.5 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report); 6qYQNJ1Uaeg-m0.6 (not run yet: queued behind the shared heavy lock; `--split test` resumes, `--score` rewrites this report).

## How to read this (caveats)

Open loop: the bot's taps do not change the video, so each tap is what the bot would do in the pro's situation, not a game it played; after the pro's own play the board already shows the pro's units, which the bot also sees. Tap density: the bot taps more often than the pro (see taps per minute), so timing recall is read against the null row (the same taps moved 20-60 s), not against zero. Perception ceiling: the pro's card is in the hand the bot perceived in only the share the ceiling row gives, so same-card agreement cannot be high; in open loop the bot's hand memory is never told about a play (taps are not sent), which may explain a stale hand (not verified here). Calibration: each game uses its own video's calibration (`runs/videos/<video>/match_0/calib.json`), because `runs/poc/mirror_calib.json` puts the arena up to 2-4.5 tiles off on these videos. Perception noise: the bot perceives the screen with the same detector it uses live, and the pro's plays come from the dataset's offline pipeline (card labels on 'medium' / 'low' confidence plays and 'unknown'-card plays are noisy; a card outside the pro's deck is kept for timing and dropped from the choice metrics). Time stamps: the dataset stamps a play when the card lands on the arena, about 0.5 s after the pro's tap, so the bot looks slightly early; the shifted rows show the effect. Hidden information: the pro knows his own next cards, the opponent's real elixir and his plan, and plays for the match result; the bot sees only the screen and acts only when its act model says so, so low recall is not an error by itself and a different card is not always a worse one. Retrieval was limited to the train and val splits so the bot never retrieves the very state the pro played; the KG v2 answers and the placement prior are mined from train games only.
