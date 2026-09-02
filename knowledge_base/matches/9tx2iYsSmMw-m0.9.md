---
video_id: 9tx2iYsSmMw
video_title: "Ryley's BEST Games of CRL 2026"
video_url: https://www.youtube.com/watch?v=9tx2iYsSmMw
match_index: 0.9
video_time: [2250.2, 2454.3]
creator: ryleycr1
own_deck: [royal-hogs, furnace, goblin-hut, dark-prince, barbarian-barrel, goblins, fireball, electro-spirit]
own_deck_key: barbarian-barrel-dark-prince-electro-spirit-fireball-furnace-goblin-hut-goblins-royal-hogs
own_archetype: bridge-spam
opponent_deck_seen: [lava-hound, rune-giant, valkyrie, skeleton-dragons, zap, fireball, inferno-dragon, tombstone, little-prince]
opponent_archetype_guess: beatdown
result: win
quality:
  readable_seconds: 204.0
  match_frames: 1193
  own_elixir_drift: {n: 300, mean: -0.209, abs_mean: 1.063, max_abs: 7.51, last: 0.238}
  events_total: 84
  events_unidentified: 13
  events_by_source: {deploy_label: 35, arena: 14, hud: 22, inferred: 13}
  hand_conf_mean: 0.449
links:
  cards: [royal-hogs, furnace, goblin-hut, dark-prince, barbarian-barrel, goblins, fireball, electro-spirit, fire-spirit, lava-hound, rune-giant, valkyrie, skeleton-dragons, zap, inferno-dragon, tombstone, little-prince]
  decks: []
  archetypes: [bridge-spam, control, beatdown]
---

## Summary

This is CRL broadcast footage of game 2 of a duel set, with Ryley talking over
it; the bottom ("own") side of the arena is Ryley. He says so himself at
t=2310: "here he had a lava hound rune giant deck with the Valkyrie and I had a
nice Goblin Hut Piggies deck", and every card he narrates playing (Royal Hogs,
Furnace, Goblin Hut, Dark Prince, Barbarian Barrel, Goblins, Fireball) shows up
as a bottom-side deploy label. The per-game automatic deck read (Furnace,
Fireball, Goblins, Electro Spirit, Dark Prince, Ice Spirit, Princess, Wizard)
is wrong in two of its slots: Princess, Wizard and Ice Spirit come only from
the broadcast-overlay "hand" reader, which is unreadable here (hand confidence
0.449, and the reader also invented Golem, Bowler, Mortar, Royal Giant and
Cannon Cart), while Royal Hogs and Goblin Hut — both confirmed by commentary
and by deploy labels / an ally `goblin-hut` track — were pushed onto the
"opponent" side by the pipeline. The clock is unreadable throughout
(broadcast overlay), so every reference below is a video second.

The match is Royal Hogs hut-spam against Lava Hound beatdown. Ryley never lets
the beatdown player build: he cycles Furnace and Goblin Hut for permanent chip,
throws Royal Hogs at the bridge five times in ~160 s (t=2257, 2292, 2346,
2391, 2415, including a split at t=2346), and adds Dark Prince, Barbarian
Barrel and Goblins at the bridge on top so the opponent can never afford a Lava
Hound. "I was being really aggressive in this game just over and over again. I
was just constantly spamming him because I don't want to let him get many lava
hound pushes" (t=2351). It works: the Lava Hound only gets going twice
(t=2264-2288 and t=2390), and the game ends with a Fireball + Heroic Dark
Prince dismount on the enemy tower — "the rune giant actually pushed the rhino
away and I was able to take the tower guys. So we were going to a game three"
(t=2445-2450). Ryley came into this game 0-1 down in the duel ("I was 0 to 1",
t=2247) and levelled it.

## Key moments

- t=2250-2258 (clock unreadable) — Clip opens with own towers full (3052/3052)
  and Ryley immediately going Furnace at [9, 2] (t=2256) and Royal Hogs into
  the left bridge at [2, 17] (t=2257). Commentary is still on the previous
  game: "I was coming off of a very disappointing loss... there was like three
  decently sized mistakes in the last 50 seconds of the game" (t=2253-2268).
- t=2264-2294 (clock unreadable) — The opponent's only early Lava Hound push
  (`lava-hound(e)` tracked from [17, 25] down to [16, 10], with
  `skeleton-dragon(e)` in the left lane); Ryley's right tower falls from 3052 to
  1720 across this stretch while he answers with a second Furnace at [9, 12]
  / [9, 10] and Goblins at [14, 12]. His voice-over here is entirely about
  mindset, not the play: "there's no point of sitting there sulking in the
  middle of a duel when... I'm still able to win the duel set" (t=2278-2284).
- t=2310-2316 (clock unreadable) — The deck call: "here he had a lava hound
  rune giant deck with the Valkyrie and I had a nice Goblin Hut Piggies deck.
  Once again, you guys are seeing a piggies. I think the only duel set where I
  didn't use piggies in today's video was against Ian."
- t=2317-2323 (clock unreadable) — Barbarian Barrel at [4, 17] (t=2317) then
  Dark Prince at [2, 17] (deploy label 'kPRiNce', t=2320/2323). "Here I went
  for the dark prince at the bridge. I wanted to be really aggressive in this
  matchup. And I even barb barrel'd just to have a more supported dark prince"
  (t=2323-2330).
- t=2329-2335 (clock unreadable) — Fireball at [4, 10] on his own half to clear
  the left-lane defence, then Furnace at [8, 2] (t=2334). He explains he
  skipped the Dark Prince ability for it: "I decided against it because I could
  just go for the furnace here. Get to my Evo furnace and he couldn't really
  lava hound because I had the evo piggies in cycle and he still had to deal
  with the furnace" (t=2333-2342).
- t=2342-2349 (clock unreadable) — Royal Hogs at [10, 18] (t=2346) and an
  offensive Fireball at [2, 27] (t=2349). "I was just kind of abusing the fact
  that he couldn't really spell my furnace away because he had to watch out for
  the evo piggies. Here I went for the split piggies" (t=2342-2348).
- t=2363-2377 (clock unreadable) — Electro Spirit at [14, 18], then Barbarian
  Barrel at [4, 17] (t=2370) with Goblins, then Furnace at [9, 10] (t=2377).
  "I'm going dark prince at the bridge on four elixir. I even ability here...
  but I instantly went barbarian barrel plus goblins at the bridge, so he
  wasn't able to go for a lava hound. I knew I'd force out the skeleton dragons
  there cuz obviously he didn't want to play anything else. So then I went for
  the evo furnace" (t=2367-2380). The forced Skeleton Dragons duly appear at
  t=2382 (`skeleton-dragon(e)` at [4, 18] and [1, 18]).
- t=2387-2400 (clock unreadable) — An ally `goblin-hut` track sits at [16, 9]
  from t=2392, and Dark Prince goes down defensively at [9, 11] (t=2393) with
  Electro Spirit at [10, 17] (t=2395) and another Furnace at [11, 10]
  (t=2400). "Right here, I went for the early hut. And my goal right now was
  just trying to make sure that his Valkyrie didn't walk up to my tower. I even
  pre-dark Prince predicting his Valkyrie... he had to go for the Evo Zap. And
  since he had to waste his evo zap, he was unable to take out my furnace or my
  goblin hut" (t=2387-2412). He calls it "just a very good dark prince play out
  of me" (t=2416).
- t=2415-2427 (clock unreadable) — Royal Hogs at [2, 18] into the opponent's
  Fireball, a Furnace pushed high at [8, 18] (t=2420), then Barbarian Barrel at
  [12, 17] plus Goblins at [15, 17] in the other lane (t=2424/2427). "He
  fireballed and I timed it with the piggies very nicely. I even went for a
  high furnace here because I knew that he would have to continue defending the
  piggies, so the fire spirit jumps over the bridge, which is what I was
  planning. I even went for the barb barrel plus goblins opposite lane kind of
  just knowing that he'd have to Valkyrie on the left side — and he actually
  Valkyrie'd, and guys look at these goblins on the right side. He didn't have
  anything" (t=2418-2436).
- t=2436-2450 (clock unreadable) — Electro Spirit at [8, 16], `dark-prince(a)`
  tracked over the bridge in the right lane, Fireball at [15, 27] (t=2441) on
  the Skeleton Dragons, and the enemy right tower reads 474. "I was just able
  to dark Prince at the bridge here. His tower was in a two fireball range so I
  knew that the odds of me winning this game were very high. I went for the
  ability and the fireball on the skelly drags and the rune giant actually
  pushed the rhino away and I was able to take the tower guys. So we were going
  to a game three" (t=2437-2450). Meanwhile his own left tower is being chewed
  by the enemy Valkyrie (`valkyrie(e)` at [3, 11], own left 2448 -> 1916).

## How Ryley uses his cards

- **Royal Hogs** — his win condition and his clock. Five bridge deploys inside
  the clip: [2, 17] (t=2257), [2, 18] (t=2292), [10, 18] (t=2346), [2, 17]
  (t=2391), [2, 18] (t=2415) — four in the left lane, one right-lane split.
  He uses them as denial as much as damage: "I was just constantly spamming him
  because I don't want to let him get many lava hound pushes" (t=2351), and
  keeping the Evo cycle available is what stops the opponent spending on the
  Hound: "he couldn't really lava hound because I had the evo piggies in cycle"
  (t=2337). At t=2415 he deliberately times them into the opponent's Fireball:
  "He fireballed and I timed it with the piggies very nicely" (t=2418).
- **Furnace** — cycled six times (t=2256 [9, 2], t=2263/2267 [9, 12], t=2270
  [9, 10], t=2315 [8, 2], t=2334 [8, 2], t=2377 [9, 10], t=2400 [11, 10],
  t=2420 [8, 18]), i.e. roughly one every 20 s, at every depth from behind the
  king to right on the river. Two distinct jobs: safe back placements to bank
  chip while the Hound is out, and the "high furnace" at t=2420 so the spawned
  Fire Spirits cross into the enemy side — "I even went for a high furnace here
  because I knew that he would have to continue defending the piggies, so the
  fire spirit jumps over the bridge, which is what I was planning" (t=2422).
  The spell-baiting logic is explicit: "I was just kind of abusing the fact that
  he couldn't really spell my furnace away because he had to watch out for the
  evo piggies" (t=2342).
- **Goblin Hut** — the second spawner, tracked as an ally building at [16, 9]
  from t=2392. "Right here, I went for the early hut" (t=2387), and he plays
  the pair as a rotation the opponent's single spell cannot keep up with:
  "he was unable to take out my furnace or my goblin hut cuz obviously I was
  going to cycle back to a new set of goblin hut and furnace. So I was able to
  defend this push very very easily because of that" (t=2404-2412).
- **Dark Prince** (Heroic / Rhino variant) — used at the bridge as a threat and
  on his own half as a prediction. Bridge: [2, 17] at t=2320/2323 ("Here I went
  for the dark prince at the bridge. I wanted to be really aggressive in this
  matchup", t=2323) and again over the river at t=2437 ("I was just able to
  dark Prince at the bridge here"). Own half: [9, 11] at t=2393 as a pre-place
  against a card he had not yet seen — "I even pre-dark Prince predicting his
  Valkyrie cuz I knew he had the Valkyrie and I wanted to instantly wipe it
  out" (t=2393-2398). He is deliberate about the 3-elixir Destructive Dismount:
  he declines it at t=2333 ("I kind of felt like I wanted to ability, but I
  decided against it because I could just go for the furnace here"), takes it
  at t=2367 with the caveat "I knew that this ability was aggressive", and
  takes it again to close the game at t=2443, where the opponent's Rune Giant
  knocks the Rhino away but the tower still falls.
- **Barbarian Barrel** — always paired, never solo. [4, 17] at t=2317 as cover
  for the bridge Dark Prince ("I even barb barrel'd just to have a more
  supported dark prince", t=2328), [4, 17] again at t=2370 with Goblins ("I
  instantly went barbarian barrel plus goblins at the bridge, so he wasn't able
  to go for a lava hound", t=2372), and [12, 17] at t=2424 as the opposite-lane
  version of the same pair.
- **Goblins** — cheap defence at [14, 12] (t=2278) and [15, 12] (t=2404)
  against the Hound support, and offence at [16, 11] (t=2386) and [15, 17]
  (t=2427). The t=2427 pair is a lane-manipulation play: "I even went for the
  barb barrel plus goblins opposite lane kind of just knowing that he'd have to
  Valkyrie on the left side, and he actually Valkyrie'd, and guys look at these
  goblins on the right side. He didn't have anything" (t=2427-2436).
- **Fireball** — three casts, all with a job: [4, 10] on his own half at t=2329
  clearing left-lane defenders behind the Dark Prince, [2, 27] at t=2349 on the
  enemy left as chip behind the split hogs, and [15, 27] at t=2441 on the
  Skeleton Dragons to finish. He is counting tower HP: "His tower was in a two
  fireball range so I knew that the odds of me winning this game were very
  high" (t=2439), and "I went for the ability and the fireball on the skelly
  drags" (t=2443).
- **Electro Spirit** — the filler that keeps the bridge plays coming: [6, 14]
  (t=2290) on defence, then [14, 18] (t=2363), [10, 17] (t=2395), [10, 16]
  (t=2412) and [8, 16] (t=2436) — a 1-elixir reset thrown in alongside the
  hogs and the Dark Prince rather than saved. No commentary on it.

## Opponent

Per Ryley the opponent is on "a lava hound rune giant deck with the Valkyrie"
(t=2310). What the footage supports:

- **Lava Hound** — two real pushes: t=2264-2288 in the right lane (tracked from
  [17, 25] down to [16, 10], with lava pups at t=2294) which takes Ryley's
  right tower from 3052 to ~1720, and a second at t=2390 ([16, 26] -> [16,
  21]) which Ryley defends with Dark Prince, Goblin Hut and Furnace. Ryley's
  whole plan is to price it out of the opponent's hands: "he couldn't really
  lava hound because I had the evo piggies in cycle" (t=2337), "I don't want to
  let him get many lava hound pushes, mainly lava pushes with the Valkyrie
  behind it" (t=2357).
- **Valkyrie** (Heroic variant) — respected all game before it appeared: "I
  know that the Valkyrie ability can be crazy" (t=2361) and "my goal right now
  was just trying to make sure that his Valkyrie didn't walk up to my tower"
  (t=2389). It finally lands late, tracked at [3, 11] from t=2444, and grinds
  his left tower 2448 -> 1916 while he is closing on the other side.
- **Skeleton Dragons** — Ryley uses them as a read: he says the bridge pressure
  at t=2370 would "force out the skeleton dragons there cuz obviously he didn't
  want to play anything else" (t=2376), and `skeleton-dragon(e)` tracks appear
  at t=2382 and again at t=2416-2424; his last Fireball (t=2441) lands on them.
- **Zap** (Evolved) — not visible in any event, but called out as a forced
  answer: "he had to go for the Evo Zap. And since he had to waste his evo zap,
  he was unable to take out my furnace or my goblin hut" (t=2402).
- **Rune Giant** — named in the deck call (t=2310) and once in play, where its
  knockback saves the tower for a moment: "the rune giant actually pushed the
  rhino away" (t=2445). No detector track for it (2024-era detector).
- **Fireball** — inferred from "He fireballed and I timed it with the piggies
  very nicely" (t=2418).
- Detector-only, low confidence: `little-prince(e)` (t=2304, t=2362),
  `inferno-dragon(e)` (t=2334, t=2382), `tombstone(e)` (t=2420, t=2448),
  `heal-spirit(e)` (t=2292), `magic-archer(e)`, `phoenix-small(e)`,
  `golden-knight(e)`. None of these is confirmed by commentary and several are
  probably mislabels of the Rune Giant / Valkyrie / Little Prince line-up.

## Lessons stated by Ryley

- t=2278-2295: "There's no point of sitting there sulking in the middle of a
  duel when, you know, I'm still able to win the duel set. Of course, I can look
  over mistakes later... there's no point of doing it in the moment cuz all
  that's going to do is affect me for the next game and just make it harder to
  win in the future."
- t=2337-2346: keep the threat in cycle to freeze the beatdown player — "he
  couldn't really lava hound because I had the evo piggies in cycle and he still
  had to deal with the furnace. I was just kind of abusing the fact that he
  couldn't really spell my furnace away because he had to watch out for the evo
  piggies."
- t=2351-2361: against Lava Hound + Valkyrie, spam first — "I was being really
  aggressive in this game just over and over again. I was just constantly
  spamming him because I don't want to let him get many lava hound pushes."
- t=2372-2378: bridge pressure is also spell/defence denial — "I instantly went
  barbarian barrel plus goblins at the bridge, so he wasn't able to go for a
  lava hound. I knew I'd force out the skeleton dragons there cuz obviously he
  didn't want to play anything else."
- t=2393-2412: pre-place against a card you know is coming, and run two
  spawners so one spell cannot answer both — "I even pre-dark Prince predicting
  his Valkyrie... since he had to waste his evo zap, he was unable to take out
  my furnace or my goblin hut cuz obviously I was going to cycle back to a new
  set of goblin hut and furnace."
- t=2422-2426: place the Furnace high when the opponent is pinned on defence —
  "I knew that he would have to continue defending the piggies, so the fire
  spirit jumps over the bridge, which is what I was planning."
- t=2427-2436: use one lane to move their answer — "I even went for the barb
  barrel plus goblins opposite lane kind of just knowing that he'd have to
  Valkyrie on the left side."
- t=2439: count spell range before committing — "His tower was in a two fireball
  range so I knew that the odds of me winning this game were very high."

## Data gaps

- **No clock at all.** This is broadcast footage; every event carries
  `clock None`, so all references in this file are video seconds.
- **Side attribution in the pipeline is scrambled.** Ryley is the bottom
  ("own") player — settled by commentary (his Furnace/Goblin Hut/Royal Hogs/
  Dark Prince narration matches bottom-side deploys, and the enemy Valkyrie at
  t=2444 walks into the "own" left tower he says he is protecting). But many of
  his plays are logged as `opponent`: Royal Hogs at [2, 17]/[2, 18]/[10, 18],
  Dark Prince ('kPRiNce') at [2, 17], Barbarian Barrel at [4, 17], plus the
  events explicitly flagged "on own half" (Furnace [9, 12], Goblins [14, 12],
  Electro Spirit [6, 14], Barbarian Barrel [8, 10]/[13, 14]/[14, 14]). All of
  those are re-attributed to Ryley here. The generated "opponent cards seen"
  list (Barbarians, Barbarian Barrel, Royal Hogs, Furnace, Lava Hound, Skeleton
  Dragons, Goblins, Electro Spirit) is therefore mostly Ryley's own deck and
  was not used.
- **HUD hand reads discarded wholesale** (mean confidence 0.449). Over the clip
  the reader produced Golem, Bowler, Tombstone, Mini P.E.K.K.A., Cannon Cart,
  Ice Wizard, Goblin Barrel, Mortar, Wall Breakers, Royal Giant, Bandit, Flying
  Machine, Battle Ram, Zappies, Inferno Tower, Lightning, Poison, Berserker,
  Skeletons, Knight, Bomb Tower, Giant, Wizard, Princess and Ice Spirit — none
  of which is consistent with the deck Ryley narrates. Every `hud`-sourced play
  (22 events: Royal Giant t=2257, Wizard t=2272/2324, Mini P.E.K.K.A. t=2379,
  Giant t=2388/2436, Bowler t=2401, Cannon t=2405, Poison t=2416, Princess
  t=2334/2421/2448, Skeletons t=2397, Berserker t=2353, Zap t=2361/2367, Ice
  Spirit t=2292/2319/2335, Fireball t=2393/2441, Goblins t=2444) is treated as
  unreliable; only the two whose deploy label agrees (Fireball at t=2441,
  labelled at [15, 27]) were used. The own-elixir trace is correspondingly
  noisy (abs_mean drift 1.06, max 7.51).
- **'BaRbaRiaN' labels read as Barbarians** at t=2250, t=2253 and t=2424
  ([12, 17]) are almost certainly truncated `Barbarian Barrel` reads — the
  t=2424 one is confirmed by Ryley narrating "the barb barrel plus goblins
  opposite lane" three seconds later with Goblins at [15, 17]. The t=2250/2253
  pair (logged as two 5-elixir opponent plays at the same tile) is unresolved;
  it may be a duplicate read of one deploy and could belong to either player, so
  Barbarians is not credited to either deck.
- **'SpiRit' at [8, 15] (t=2292)** was logged as Ice Spirit off a HUD slot;
  given the deck read it is more likely the Electro Spirit. Ice Spirit is not
  credited to the deck.
- **13 unidentified events.** Six are opponent damage with no unit in range
  (own_left -151 x3 around t=2275-2278, own_right -273/-153/-479 at
  t=2283-2287, own_right -172 at t=2419, own_left -266 x2 at t=2444-2446) —
  most likely Lava Hound / lava-pup contact or spells that were not read. Seven
  are own elixir drops with no readable hand change (t=2278 -5, t=2280 -2,
  t=2347 -6, t=2397 -2), which on cost alone are consistent with Royal Hogs,
  Dark Prince ability, Goblins or Electro Spirit but cannot be assigned.
- **Cards named in prose but not credited as slugs anywhere:** none missing
  from the KB in this game (Rune Giant, Little Prince, Heroic Dark Prince and
  Heroic Valkyrie all have files). Hero variants are not linked as card slugs:
  the Dark Prince ability (Destructive Dismount, `heroes/dark-prince-hero.md`)
  and the Valkyrie ability (Wild Whirlwind, `heroes/valkyrie-hero.md`) are
  discussed in prose only.
- **Partial clip.** 204 s of readable footage. Own towers read full (3052/3052)
  at t=2250 so the opening is probably intact, but the enemy right tower already
  reads 238 at t=2250 and 474 at t=2444, and enemy tower reads are `None` for
  most of the clip, so enemy crown-tower HP cannot be tracked. Tower values at
  t=2452 (own king 1111, enemy left 1305) look like the transition into game
  three. The result (win) rests on Ryley's own words at t=2448-2455, not on a
  readable end screen.
- **Opponent elixir** is an estimate throughout and swings implausibly (e.g.
  0.24 at t=2392 to 7.75 at t=2414); it was not used for any claim here.
