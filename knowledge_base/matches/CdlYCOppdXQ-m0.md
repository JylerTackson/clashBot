---
video_id: CdlYCOppdXQ
video_title: Jynxzi Challenged Me in C.H.A.O.S Mode
video_url: https://www.youtube.com/watch?v=CdlYCOppdXQ
match_index: 0
video_time: [259.9, 833.8]
creator: ryleycr1
own_deck: [ronin, tombstone, barbarian-hut, suspicious-bush, p-e-k-k-a, bandit, mighty-miner, minions, skeletons]
own_deck_key: null
own_archetype: control
opponent_deck_seen: [goblin-barrel, the-log, arrows, rocket, mini-p-e-k-k-a, royal-delivery, miner, berserker, ronin, bandit]
opponent_archetype_guess: bait
result: unknown
quality: {"readable_seconds": 573.9, "match_frames": 1370, "own_elixir_drift": {"n": 69, "mean": 1.209, "abs_mean": 1.483, "max_abs": 5.026, "last": 0.953}, "events_total": 42, "events_unidentified": 4, "events_by_source": {"inferred": 4, "deploy_label": 28, "arena": 8, "hud": 2}, "hand_conf_mean": 0.345}
links:
  cards: [ronin, tombstone, barbarian-hut, suspicious-bush, p-e-k-k-a, bandit, mighty-miner, minions, skeletons, goblin-barrel, the-log, arrows, rocket, mini-p-e-k-k-a, royal-delivery, miner, berserker, golden-knight, monk, witch, guards, wall-breakers, magic-archer, knight, poison]
  decks: []
  archetypes: [control, bait]
---

## Summary

This is not one match. The pipeline packaged the whole readable window (video
259.9s to 833.8s) as `match_0`, but the transcript contains at least three
"good luck / one more" resets, so the 42 events below are spread across three
or four separate C.H.A.O.S. games against Jynxzi, and Ryley says out loud at
262s that he is changing his deck between them ("Okay, I'm gonna change out my
[deck]") while Jynxzi answers at 323s "I did, too. I changed my win condition
up." Because of that there is no single 8-card own deck to record: `own_deck`
above is the union of the nine cards I could actually support with a deploy
label, an ally arena track or a HUD play, and `own_deck_key` is null. The one
game with real coverage is the middle one (~561s to ~745s), where Ryley runs a
defensive spawner/anti-tank shell — [Tombstone](../cards/tombstone.md) and
[Barbarian Hut](../cards/barbarian-hut.md) behind
[Ronin](../cards/ronin.md), [P.E.K.K.A.](../cards/p-e-k-k-a.md) and
[Mighty Miner](../cards/mighty-miner.md), with
[Suspicious Bush](../cards/suspicious-bush.md) as cheap chip — against a
[Goblin Barrel](../cards/goblin-barrel.md) bait shell with
[Rocket](../cards/rocket.md), [The Log](../cards/the-log.md),
[Arrows](../cards/arrows.md), [Royal Delivery](../cards/royal-delivery.md),
[Mini P.E.K.K.A.](../cards/mini-p-e-k-k-a.md) and [Miner](../cards/miner.md).
Perception quality is poor throughout (calibration fell back to default
corners, hand confidence 0.345, elixir drift `abs_mean` 1.48 with a 5.0
maximum, the battle clock was never read once, and there are only 42 events in
574 s), so most of the value in this window is what Ryley *says* about
C.H.A.O.S. rather than what the pipeline saw. No crown or tower outcome was
read, so the result is unknown for every game here.

## Key moments

Note: the match clock was never read in this window, so every timestamp is
video seconds. Game boundaries are inferred from the transcript, not from a
clock reset (there was none to see).

### Pre-game / deck and modifier editing (~260s-320s)

- t=262-266s: hand reads flip to a different four cards every frame and elixir
  jumps 3 -> 0 with no readable hand change (an `UNIDENTIFIED` own event at
  t=266.4s, -3 elixir). Commentary places this on the deck/modifier screen:
  "Okay, I'm gonna change out my [deck]" (262s), "What does this do? What is
  this one do?" (263s), "This has to be good" (265s).
- t=291-303s: deploy labels read `ARROWS`, `THeJLog` and a Goblin Barrel track
  while the HUD is still in menu mode. Either the tail of a game that started
  before the readable window, or menu card art picked up as deploy labels —
  see Data gaps.
- t=301.8s: second own `UNIDENTIFIED` event, elixir 6.0 -> 1.0 (-5). This is
  the only -5 own drop in the file and it lands during the menu phase, so it is
  *not* evidence of a Tombstone Hero "Regal Revive".

### Game A (~322s-~540s)

- t=321.8s (HUD): [Minions](../cards/minions.md) played, slot 2 emptied,
  elixir -3, seconds after both players say "Good luck, bro" (320-322s).
- t=326.1s: opponent [Ronin](../cards/ronin.md) (5 elixir) and
  [The Log](../cards/the-log.md) (2 elixir) inside the same second. Jynxzi had
  already flagged the Ronin at 346s: "Oh, you have the boomerang log, bro. That
  log's kind of like I want to see what it does."
- t=355s (commentary only, no events): "It just like killed the whole
  tombstone" — a modified Log clearing a [Tombstone](../cards/tombstone.md).
- t=361-387s (commentary only): Ryley's Ronin parry lesson, quoted in full
  under Lessons.
- t=433-443s (commentary only): Jynxzi spots the Tombstone + Wall Breakers
  modifier combination — "Look at the wall breaker." / "The tombstone. The
  tombstone." / "WHY? THAT'S NOT FAIR. LIKE, WHAT? You just get infinite wall
  breakers." Nothing in the event stream covers this stretch; 330s-560s
  produced zero events.
- t=512-527s (commentary only): "Look at the bandits" (three times), then "Yo,
  the wall breakers" and Jynxzi accusing Ryley of an "elixir glitch" because
  "ALL I'M DOING IS DEFENDING. How do you just KEEP ATTACKING?"
- t=540-547s: "OKAY, one more. No, no, one more. I can win one more." Game A
  ends; no result was visible.

### Game B (~561s-~745s) — the only game with real event coverage

- t=562.4s: own `UNIDENTIFIED` event, elixir 3.0 -> 0.0 (-3), immediately after
  "OKAY, there we go" (561s) — an opening play whose card could not be read.
- t=567.4-576.4s: four [Ronin](../cards/ronin.md) deploy labels attributed to
  the opponent, all at the same tile [4, 19]. That tile is the file's junk
  coordinate (Berserker, Bandit and Arrows also land on it), so these are
  almost certainly one Ronin play re-read across frames, with an unreliable
  side attribution.
- t=576.4s: own [Suspicious Bush](../cards/suspicious-bush.md) at tile [5, 9]
  (deploy label, no HUD confirmation) — his half, left-of-centre, i.e. sent up
  the lane rather than used defensively, which matches the card's role as an
  offensive building-targeter.
- t=578-586s: an ally [Mighty Miner](../cards/mighty-miner.md) is tracked
  crossing from (17, 3) to (13, 8), i.e. walking up the field, while an enemy
  [Royal Delivery](../cards/royal-delivery.md) lands at (14, 7-8).
- t=592.1s (HUD): [Skeletons](../cards/skeletons.md), slot 2 emptied, elixir -1.
- t=595.4s: own [Barbarian Hut](../cards/barbarian-hut.md) at tile [5, 5] —
  deep in his own half, well behind the river, the standard defensive spawner
  placement.
- t=606-608s: an ally [P.E.K.K.A.](../cards/p-e-k-k-a.md) is tracked at (15, 5)
  then (16, 5), marked "retreating" (i.e. she has turned onto a defensive
  target rather than pushing).
- t=611.4s: own [Ronin](../cards/ronin.md) at tile [8, 4] — centre, deep in his
  own half, a defensive/cycle placement rather than a bridge push.
- t=620.4s: own [Tombstone](../cards/tombstone.md) at tile [5, 6], one tile in
  front of the Barbarian Hut he placed 25 s earlier at [5, 5] — the two
  spawners stacked in the same pocket, exactly the "Spawner strategy" the
  Tombstone card notes describe.
- t=622.4-631.4s: [Berserker](../cards/berserker.md) deploy labels (four reads
  at the junk tile [4, 19]).
- t=631.4s: own [Bandit](../cards/bandit.md) at tile [4, 9], on his own half
  near the river on the left — a punish/counterpush placement, and the only
  Bandit read attributed to Ryley.
- t=639.4s: opponent [Rocket](../cards/rocket.md) at tile [7, 14], on Ryley's
  half at the river. With a Barbarian Hut and a Tombstone stacked at [5, 5-6]
  this is the exact spell punish the Barbarian Hut card warns about.
- t=649.4s: enemy [Mini P.E.K.K.A.](../cards/mini-p-e-k-k-a.md) tracked at
  (6, 23), then walking down the left lane at (6, 24) -> (6, 22) around 650s.
- t=658-660s (commentary over the events): Jynxzi asks "Yo, Riley, why would you
  even put down the Pekka?" and Ryley answers "I HAVE THE RONIN, BABY."
- t=660.5s: own [Ronin](../cards/ronin.md) at tile [7, 11] — second Ronin of
  the game, this one much closer to the river, i.e. used to meet the push.
- t=664-669s: "It's okay. I don't really mind that too much. I just need to
  cycle through my deck. Even though this card is like actually insane."
- t=666.5s / t=606.4s: opponent [Goblin Barrel](../cards/goblin-barrel.md)
  deploy labels (lvl 11), plus repeated Goblin Barrel arena tracks from 581s to
  666s — the opponent's most-used card in this game.
- t=671-679s: "Chad, I'm telling you, if you're underlevelled, just use the
  Ronin, bro. Like, this guy is literally the underleveled Chad. The Ronin is
  the definition of an underleveled [win]."
- t=679.5-685.5s: [Bandit](../cards/bandit.md) deploy labels attributed to the
  opponent (three reads, junk tile [4, 19]).
- t=693-704s: "Whoa. Where are my guards? Where are my guards?" / Ryley: "Oh,
  you got the tombstone." / "you dropped it right before you clicked the
  modifiers, so you get it on the next thing" — the [Guards](../cards/guards.md)
  modifier did not apply because the card was already on the field.
- t=740-750s: game ends in an argument with no readable outcome — Jynxzi: "you
  Riley, you lost, bro. I'm actually sorry"; Ryley: "Jinxy, just prepare to get
  touched." No crown or tower state was read, so the result stays unknown.

### Games C and D (~763s-833.8s)

- t=762-769s: a new game starts on Jynxzi's "I have a mini mega[knight] hog,
  bro. Good luck", with Ryley replying "you're not going to get to my tower in
  time, bro, cuz I'm putting [modifiers] on your head." Zero events were
  produced for this whole game.
- t=778-792s: Jynxzi: "Bro, I DIDN'T EVEN KNOW YOU HAD THE SNEAKY PEKKA, MAN"
  and then "Riley, I'M NOT EVEN DOING DAMAGE TO YOU, BRO" — the only signal
  either way about how a game went, and it is commentary, not observation.
- t=804.6-812.4s: last game. Opponent [Miner](../cards/miner.md) at tile
  [3, 13] on Ryley's half and a Goblin Barrel track at (14, 17); own deploy
  labels read [Barbarian Hut](../cards/barbarian-hut.md) and then
  [P.E.K.K.A.](../cards/p-e-k-k-a.md), both at tile [1, 11], two seconds apart,
  with own elixir reading 3 -> 0 -> 2 -> 0. Six plus seven elixir in two
  seconds is not possible at that elixir, so at least one of these is a misread
  (see Data gaps).
- t=832.6s: opponent [Arrows](../cards/arrows.md) at the junk tile; the
  readable window ends at 833.8s with the game still running.

## How Ryley uses his cards

- **[Ronin](../cards/ronin.md)** (9 commentary mentions, the most-discussed card
  in the video; two own deploy labels). Played at [8, 4] (t=611.4s) and then at
  [7, 11] (t=660.5s) — both on his own half, the first deep and central, the
  second up near the river to intercept. He never bridge-pushed it in the
  covered game. His pitch is levels, not placement: "if you're underlevelled,
  just use the Ronin, bro... The Ronin is the definition of an underleveled
  [card]" (671-679s), and when asked why he bothered with the P.E.K.K.A.:
  "I HAVE THE RONIN, BABY" (660s). He also treats the Ronin's Parry as its
  headline mechanic — see Lessons.
- **[Tombstone](../cards/tombstone.md)** (5 mentions; one deploy label). Placed
  at [5, 6] at t=620.4s, one tile in front of the Barbarian Hut he had put at
  [5, 5], stacking two spawners in the same defensive pocket. His verdict at
  729-736s: "I really think that the Tombstone is really good."
- **[Barbarian Hut](../cards/barbarian-hut.md)** (two deploy labels: [5, 5] at
  t=595.4s, [1, 11] at t=808.6s). Used as the anchor of the spawner pocket, and
  the opponent's [Rocket](../cards/rocket.md) at [7, 14] 44 s later is the price
  of it. His own assessment is negative: "spawners are pretty good in this too,
  huh Ry?" / "Yeah, bar[barian] hut's not that good, though... I don't think
  that the bar[barian] hut is that good anymore. It used to be the best"
  (725-736s).
- **[P.E.K.K.A.](../cards/p-e-k-k-a.md)** (ally arena track at (15, 5)-(16, 5)
  at 606-608s marked "retreating"; a deploy label at [1, 11] at t=810.6s).
  Used as a defensive heavy, not a win condition — she is on his own half both
  times she is seen, and when Jynxzi questions the card ("why would you even put
  down the Pekka?", 658s) Ryley's answer is about the Ronin, not about pushing
  with her. Jynxzi later calls it "the sneaky Pekka" (778s), i.e. it had a
  C.H.A.O.S. modifier.
- **[Suspicious Bush](../cards/suspicious-bush.md)** (deploy label [5, 9],
  t=576.4s). Sent up his own half rather than kept for defence, which is the
  correct use of a 2-elixir invisible building-targeter; it was the first thing
  he put down in the covered game. No commentary on it.
- **[Mighty Miner](../cards/mighty-miner.md)** (ally arena track (17, 3) ->
  (13, 8), 578-586s, plus the most stable of the four hand slots for most of the
  game). Seen walking up the field while an enemy Royal Delivery lands nearby.
  Not discussed.
- **[Bandit](../cards/bandit.md)** (deploy label [4, 9], t=631.4s). One own
  placement, on his half near the left-side river — a punish/counterpush spot
  consistent with the card's offensive role. Jynxzi's "Look at the bandits"
  (512-518s), plural and laughing, points at a C.H.A.O.S. modifier rather than
  the base card.
- **[Minions](../cards/minions.md)** (HUD play, t=321.8s, -3 elixir). Played
  seconds after the game-A "good luck", the only Minions event in the window;
  placement unknown.
- **[Skeletons](../cards/skeletons.md)** (HUD play, t=592.1s, -1 elixir).
  Cheapest cycle card in the covered game; placement unknown. The ally skeleton
  tracked at (6, 14)-(5, 12) around 674-680s is more likely a Tombstone spawn
  than this card.

## Opponent

Jynxzi's cards, and how Ryley handled them:

- **[Goblin Barrel](../cards/goblin-barrel.md)** — his signature card here, read
  by deploy label twice (t=606.4s, t=666.5s at lvl 11) and tracked repeatedly
  between 581s and 807s. Ryley's two spawners at [5, 5-6] are a structural
  answer to it: the Tombstone card's own notes point out the opponent must spend
  a spell on it, and the Skeletons cover barrel drops. At 685-689s Jynxzi says
  "You got the goblin b[arrel] when I lost" and Ryley replies "It's not the
  goblin b[arrel] one, but it's pretty broken" — so the barrel modifier was the
  benchmark for a broken roll.
- **[The Log](../cards/the-log.md)** — played at [3, 0] (t=294.1s) and [4, 8]
  (t=326.1s), both on Ryley's half. Jynxzi called it "the boomerang log" (346s)
  and Ryley wanted to see what the modifier did; at 355s "It just like killed
  the whole tombstone", i.e. a modified Log wiping a Tombstone outright.
- **[Rocket](../cards/rocket.md)** — t=639.4s at [7, 14], 44 s after the
  Barbarian Hut and 19 s after the Tombstone. Exactly the punish the hut invites.
- **[Arrows](../cards/arrows.md)** — t=291.1s at [5, 0] and t=832.6s.
- **[Mini P.E.K.K.A.](../cards/mini-p-e-k-k-a.md)** — tracked at (6, 22)-(6, 24)
  around 649-652s, walking the left lane; the Tombstone and its Skeletons are
  the natural distraction for a single-target melee threat like this.
- **[Royal Delivery](../cards/royal-delivery.md)** — landed on Ryley's half at
  (14, 7-8) at t=675.5s, in the middle of the Mighty Miner's walk up the field.
- **[Miner](../cards/miner.md)** — t=804.6s at [3, 13], on Ryley's half, in the
  last game of the window.
- **[Berserker](../cards/berserker.md)**, **[Ronin](../cards/ronin.md)** and
  **[Bandit](../cards/bandit.md)** were also read as opponent deploy labels, but
  all of them at the file's junk tile [4, 19], so their side attribution is not
  trustworthy — the Ronin in particular is clearly Ryley's card from the
  commentary. See Data gaps.

## Lessons stated by Ryley

- t=361-387s — the Ronin Parry as a Golden Knight counter: "he parries the dash,
  too. He parries everything... he's crazy against the
  [Golden Knight](../cards/golden-knight.md) cuz if the parry blocks the dash,
  then the Golden Knight just stops moving and it doesn't dash the stuff behind
  it. It's like a Golden Knight counter." He then spells out the scenario:
  "imagine you have your [Witch](../cards/witch.md) right behind a Ronin. If the
  Ronin parries a Golden Knight dash, then it won't go to the Witch and the
  skellies. It will just stop right there."
- t=405-417s — scout the opponent's modifier immediately: "this may be kind of
  sweaty, but I click on the thing right away to see what you get, to at least
  not get caught off guard. I like to try to prepare for defense every time."
- t=590-596s — modifiers do not stack: "it cancels. So, if you get two modifiers
  for the same card, it will use the most recent one you chose."
- t=600-612s — keep re-rolling modifiers rather than hoarding: "Almost always
  [take a new one], unless you get an insanely crazy one... You know how I said
  the Ronin [Monk](../cards/monk.md) one is the best one? If you have a
  different one on the Ronin, it's good to replace. But most of the time, you
  just get new ones. Otherwise, you're just wasting one."
- t=639s — spend modifiers on your best cards: "Yes. Like save your big dogs.
  That's what I do."
- t=645-648s — the mode has a shallow skill ceiling: "It's very little. Like if
  you play like an hour or two of C.H.A.O.S., you're maxed out on all the
  strategies."
- t=700-720s — the modifier timer rule, after Jynxzi lost a
  [Guards](../cards/guards.md) modifier: "you dropped it right before you click
  the modifiers, so you get it on the next thing... If you look at the top
  right, it says when you're going to get your next modifier. So, right before
  that timer, you probably don't want to drop any cards unless you absolutely
  have to... unless you need to defend."
- t=664-669s — cycling is still the point: "I don't really mind that too much. I
  just need to cycle through my deck. Even though this card is like actually
  insane."
- t=671-679s — level-independence: "if you're underlevelled, just use the Ronin,
  bro. Like, this guy is literally the underleveled Chad."
- t=729-736s — spawners in C.H.A.O.S.: "bar[barian] hut's not that good, though.
  I really think that the Tombstone is really good. I don't think that the
  bar[barian] hut is that good anymore. It used to be the best."

## Data gaps

- **This is not one match.** The pipeline emitted a single `match_0` covering
  259.9s-833.8s, but the transcript contains at least three game starts
  ("Good luck, bro" at 320-322s; "OKAY, there we go" at 561s after "I ALMOST
  QUEUED A BATTLE"; "I have a mini mega[knight] hog, bro. Good luck" at 762s)
  and a fourth "one more" at 794-802s. The battle clock was never read (`clock
  None` on every one of the 1370 frames), so there is no clock reset to confirm
  the boundaries — the game split above is inferred from the transcript alone
  and should be treated as interpretation.
- **Deck changed between games**, stated on camera: Ryley at 262s "I'm gonna
  change out my [deck]", Jynxzi at 323-327s "I did, too. I changed my win
  condition up. I don't want to use the same deck again cuz that'd be kind of
  lame." `own_deck_key` is therefore null and `own_deck` is the union of what
  was observed across games, not a legal 8-card deck.
- **The pipeline's own-deck read was overridden.** Its per-game and video-level
  consensus was Barbarian Hut, Suspicious Bush, Tombstone, P.E.K.K.A., Minions,
  Skeletons, Mighty Miner and Ronin. I kept those (Ronin, Tombstone, Barbarian
  Hut, Suspicious Bush and P.E.K.K.A. on deploy labels; Mighty Miner on an ally
  arena track plus the most persistent hand slot; Minions and Skeletons on
  single HUD plays only) and **added Bandit**, which has an own-half deploy
  label at [4, 9] (t=631.4s) that the consensus dropped. Confidence tiers:
  deploy-label confirmed = Ronin, Tombstone, Barbarian Hut, Suspicious Bush,
  P.E.K.K.A., Bandit; arena-track + slot = Mighty Miner; HUD hand read only,
  weakest = Minions, Skeletons.
- **Hand reads are unusable.** Mean hand confidence is 0.345 and the four slots
  change to a different set of cards almost every 2 s frame (Heal Spirit, X-Bow,
  Mortar, Ice Wizard, Battle Ram, Flying Machine, Giant, Balloon, Executioner,
  Dark Prince and others appear that are certainly not in this deck). None of
  those cards are recorded anywhere in this file. The only partial exception is
  the 596s-676s stretch, where slot 3 repeatedly reads Barbarian Hut and slot 4
  repeatedly reads Mighty Miner; that stability is why Mighty Miner is listed at
  all. The two HUD-sourced plays (Minions t=321.8s, Skeletons t=592.1s) inherit
  this weakness even though the tool marked them high confidence.
- **Tile [4, 19] is a junk coordinate.** Eleven opponent deploy-label events
  land on exactly that tile — Ronin x4 (567-576s), Berserker x4 (622-631s),
  Bandit x3 (679-685s) and Arrows (832.6s). Repeated identical tiles at 3 s
  spacing are the same on-screen label re-read across frames, not separate
  plays, and the fixed coordinate means their side attribution ("opponent")
  cannot be trusted. Treat each cluster as one play of unknown ownership. The
  Ronin cluster in particular conflicts with Ryley's own Ronin deploy labels and
  his "I HAVE THE RONIN, BABY" (660s).
- **Calibration fell back to default corners**, so all tiles are approximate.
  Concretely inconsistent examples: Goblin Barrel deploy labels at [2, 25] and
  [2, 23] (a barrel is thrown at the *enemy* tower, so it cannot land on the
  thrower's own half), and Goblin Barrel arena tracks parked at (14, 17-19) for
  minutes at a time, which is a static misclassification rather than a barrel.
- **The t=808.6s / t=810.6s own pair is not physically possible.** Barbarian Hut
  (6 elixir) and P.E.K.K.A. (7 elixir) two seconds apart with own elixir reading
  3 -> 0 -> 2 -> 0 cannot both be real plays at normal elixir. Either a
  C.H.A.O.S. elixir modifier was in effect, or one of the labels is menu card
  art read as a deploy label (t=807s commentary is "Let me get this. Low key. I
  have an idea", which is modifier-picking language). Both cards are supported
  elsewhere, so this does not change the deck list, but do not use these two
  placements.
- **Four unidentified own events** (elixir moved, no card readable):
  t=266.4s (-3), t=301.8s (-5), t=562.4s (-3), t=812.4s (-3). The -5 at
  t=301.8s is the only own -5 in the file; it falls in the menu stretch, not in
  a game, so I did **not** read it as a Tombstone Hero "Regal Revive". Nothing
  else in the events or the commentary shows a Tomb Queen or the hero form, so
  [tombstone-hero](../heroes/tombstone-hero.md) is *not* cited as being in play
  in this match.
- **Own elixir is drifting badly**: n=69, mean +1.209, abs_mean 1.483, max_abs
  5.026. Every elixir number quoted above is an estimate, and C.H.A.O.S.
  modifiers can change card costs, which makes elixir-delta card inference even
  less safe than usual here.
- **Two long stretches have commentary but no events at all**: 330s-560s (the
  bulk of game A, including the "infinite wall breakers" Tombstone combo and the
  "look at the bandits" moment) and 686s-800s (all of game C). Anything in those
  ranges is transcript-only and is marked as such above.
- **Cards named in commentary that are not in the knowledge base** or that could
  not be tied to a play: "[Wall Breakers](../cards/wall-breakers.md)" (4
  mentions) appears only as a Tombstone modifier effect, never as a played card;
  the "mini mega[knight] hog" at 762s is a garbled auto-transcript phrase and is
  not recorded as any card; [Magic Archer](../cards/magic-archer.md) ("the magic
  archer expo", 400s), [Knight](../cards/knight.md) (448s) and
  [Poison](../cards/poison.md) (696s) are single passing mentions with no
  supporting event.
- **C.H.A.O.S. modifiers are not cards** and are not recorded as such anywhere
  above. The modifiers that clearly mattered — a Tombstone that spawns Wall
  Breakers, a Log that "kills the whole tombstone", a "sneaky" P.E.K.K.A., a
  Ronin/Monk modifier, multiplying Bandits — are described in prose only.
- **Result is unknown for every game.** No crown counts or tower hitpoints were
  read (the tower HP fields are either `None` or implausible values such as 106
  and 6104), and the closest thing to an outcome is Jynxzi's "Riley, I'M NOT
  EVEN DOING DAMAGE TO YOU, BRO" at 789-792s.
- **`own_archetype: control`** describes only the game-B shell (two spawners
  plus Ronin/P.E.K.K.A./Mighty Miner as heavy defence, Suspicious Bush for
  chip). It is not a claim about the whole window, since the deck changed
  between games. `opponent_archetype_guess: bait` rests on the Goblin Barrel
  being present in three of the four games plus Log/Arrows/Rocket, and is
  likewise low confidence.
