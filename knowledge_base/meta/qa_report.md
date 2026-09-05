# QA report

Generated 2026-09-02T03:50:13+00:00

## Summary

- cards: 126 done, 0 pending, 0 failed
- evolutions: 42 done, 0 pending, 0 failed
- heroes: 16 done, 0 pending, 0 failed

## Checks

- PASS: Card file count: 126 files in cards/ vs 126 cards in card_index.md
- PASS: Card frontmatter: 126/126 cards have all required fields non-empty
- PASS: Card images: 126/126 image_path files exist, are PNG and > 1KB
- PASS: Card prose: 126/126 cards have Strong/Weak/Notes sections filled
- PASS: Evolutions: 42/42 cards with has_evolution:true have an evolutions/ file
- PASS: Evolution files: 42/42 pass frontmatter/image/placeholder/link checks
- PASS: Heroes: 16/16 heroes in the Heroes index have a heroes/ file
- PASS: Hero files: 16/16 pass frontmatter/image/placeholder/link checks
- PASS: Cross-links: every card<->evolution and card<->hero pair links both ways

## Per-item problems

None.

## Sections marked "Not specified on source page" (76 files)

These are expected outcomes (the wiki page has no such content), listed so they are visible.

- `cards/archers.md`: Abilities and special mechanics
- `cards/arrows.md`: Abilities and special mechanics
- `cards/balloon.md`: Abilities and special mechanics
- `cards/bandit.md`: Abilities and special mechanics
- `cards/barbarian-barrel.md`: Abilities and special mechanics
- `cards/barbarians.md`: Abilities and special mechanics
- `cards/bats.md`: Abilities and special mechanics
- `cards/battle-healer.md`: Abilities and special mechanics
- `cards/battle-ram.md`: Abilities and special mechanics
- `cards/bomb-tower.md`: Abilities and special mechanics
- `cards/bomber.md`: Abilities and special mechanics
- `cards/bowler.md`: Abilities and special mechanics
- `cards/cannon-cart.md`: Abilities and special mechanics
- `cards/cannon.md`: Abilities and special mechanics
- `cards/cannoneer.md`: Abilities and special mechanics
- `cards/clone.md`: Abilities and special mechanics
- `cards/dagger-duchess.md`: Abilities and special mechanics
- `cards/dark-prince.md`: Abilities and special mechanics
- `cards/earthquake.md`: Abilities and special mechanics
- `cards/electro-dragon.md`: Abilities and special mechanics
- `cards/electro-giant.md`: Abilities and special mechanics
- `cards/elite-barbarians.md`: Abilities and special mechanics
- `cards/elixir-collector.md`: Abilities and special mechanics
- `cards/fire-spirit.md`: Abilities and special mechanics
- `cards/firecracker.md`: Abilities and special mechanics
- `cards/freeze.md`: Abilities and special mechanics
- `cards/giant-skeleton.md`: Abilities and special mechanics
- `cards/goblin-cage.md`: Abilities and special mechanics
- `cards/goblin-curse.md`: Abilities and special mechanics
- `cards/goblin-gang.md`: Abilities and special mechanics
- `cards/goblin-machine.md`: Abilities and special mechanics
- `cards/goblins.md`: Abilities and special mechanics
- `cards/guards.md`: Abilities and special mechanics
- `cards/heal-spirit.md`: Abilities and special mechanics
- `cards/hog-rider.md`: Abilities and special mechanics
- `cards/ice-golem.md`: Abilities and special mechanics
- `cards/inferno-dragon.md`: Abilities and special mechanics
- `cards/lightning.md`: Attributes, Abilities and special mechanics
- `cards/lumberjack.md`: Abilities and special mechanics
- `cards/magic-archer.md`: Abilities and special mechanics
- `cards/mega-minion.md`: Abilities and special mechanics
- `cards/miner.md`: Abilities and special mechanics
- `cards/mini-p-e-k-k-a.md`: Abilities and special mechanics
- `cards/minion-horde.md`: Abilities and special mechanics
- `cards/minions.md`: Abilities and special mechanics
- `cards/mirror.md`: Abilities and special mechanics
- `cards/phoenix.md`: Abilities and special mechanics
- `cards/poison.md`: Attributes
- `cards/prince.md`: Abilities and special mechanics
- `cards/ronin.md`: Abilities and special mechanics
- `cards/royal-chef.md`: Abilities and special mechanics
- `cards/royal-ghost.md`: Abilities and special mechanics
- `cards/royal-hogs.md`: Abilities and special mechanics
- `cards/royal-recruits.md`: Abilities and special mechanics
- `cards/rune-giant.md`: Weak against
- `cards/skeleton-army.md`: Abilities and special mechanics
- `cards/skeleton-barrel.md`: Abilities and special mechanics
- `cards/skeleton-dragons.md`: Abilities and special mechanics
- `cards/skeletons.md`: Abilities and special mechanics
- `cards/sparky.md`: Abilities and special mechanics
- `cards/spear-goblins.md`: Abilities and special mechanics
- `cards/spirit-empress.md`: Abilities and special mechanics
- `cards/tesla.md`: Abilities and special mechanics
- `cards/tornado.md`: Abilities and special mechanics
- `cards/tower-princess.md`: Abilities and special mechanics
- `cards/valkyrie.md`: Abilities and special mechanics
- `cards/void.md`: Abilities and special mechanics
- `cards/wall-breakers.md`: Abilities and special mechanics
- `cards/zap.md`: Abilities and special mechanics
- `evolutions/elite-barbarians-evolution.md`: Notes
- `evolutions/minion-horde-evolution.md`: Overview, Notes
- `heroes/barbarian-barrel-hero.md`: Weak against
- `heroes/berserker-hero.md`: Strong against, Weak against
- `heroes/goblins-hero.md`: Strong against, Weak against
- `heroes/ice-golem-hero.md`: Weak against
- `heroes/tombstone-hero.md`: Overview, Strong against, Weak against

## Tower troop coverage check (2026-09-02)

The wiki category "Tower Troop Cards" lists five pages: Tower Princess, Cannoneer, Dagger Duchess, Royal Chef and Baby Goblins. The first four are live cards and are in `cards/` with `card_type: Tower Troop` (Tower Princess Common, Cannoneer Epic, Dagger Duchess and Royal Chef Legendary; tower troops have no elixir cost, so `elixir_cost: n/a`). Baby Goblins is tagged `RemovedContent` on the wiki: a temporary tower troop from the June 2024 Goblin Queen's Journey event, not in the live game, so it is intentionally excluded (Card Overviews, the ground-truth list, omits it too). Tower troop rules from the wiki: chosen before battle, rarities as above, upgradable to level 16 but never past the player's King Tower level.

## Hero coverage gap (noted 2026-09-02 during Phase 4)

Ryley's video `nSXIs16M7Ag` ("Hero Ice Wizard is SKILL and OVERPOWERED!!") plays a
Hero Ice Wizard, and `Ice Wizard/Hero`, `Mega Knight/Hero` and `Battle Healer/Hero`
pages exist on the wiki, but none is listed on the `Heroes` page and their content is
a stub ("Coming soon...") or a deletion placeholder. No hero file could be written for
them; `cards/ice-wizard.md` therefore still says `has_hero_variant: false`. From the
video commentary the Hero Ice Wizard ability costs 2 elixir (unverified against the
wiki). Re-run `scripts/kb/enumerate_cards.py` + `fetch_cache.py` once the wiki pages
are filled in; the enumerator only reads the `Heroes` list table, so add a probe of
`<card>/Hero` pages if the list keeps lagging.

## Card coverage gap: Minion Giant (noted 2026-09-02 during Phase 4)

Ryley's video `CInrqMTlVkg` ("Minion Giant is the Most OVERPOWERED Card!!") is built
around Minion Giant, which is not in the `Card Overviews` list and whose wiki page is a
stub ("Coming soon..."). There is no `cards/minion-giant.md`; the hand reader reads the
card as Giant and the unit detector cannot see it. Match files for that video name it
in prose only. (Boss Bandit, Ronin, Rune Giant and Spirit Empress do have card files;
only Minion Giant is missing as of this note.)

## Phase 4 coverage (ryleycr1 creator videos, closed 2026-09-02)

Data collection was stopped on the user's instruction once each machine finished its in-flight video. Generated by `tools/phase4_summary.py`.

- Videos with match files: 33 of 48 downloaded; match files: 197
- Results recorded: win 162, unknown 19, loss 16
- Deck files: 68 total, 52 created from creator matches, 56 carrying creator match blocks
- Card files with creator insights: 124 of 126
- Per video (id, title, games):
  - `-V4H_YeMGGk` My Opinion on the BIGGEST Balance Changes EVER!!: 6
  - `32pkLm4-QMc` This Mega Knight Bait Deck is EXTREMELY BROKEN!!: 6
  - `5bU83eIW8Yg` TOP 5 Decks in Clash Royale for the New Meta: 5
  - `5nOaTBlYLlg` I QUALIFIED for the Clash Royale League WORLD FINALS!!: 6
  - `6qYQNJ1Uaeg` This Hero Berserker Deck has a 100% WIN Rate : 7
  - `7R5FP2IYWsw` Classic Log Bait RETURNS to Clash Royale!!: 6
  - `9tx2iYsSmMw` Ryley's BEST Games of CRL 2026: 11
  - `A5qMlVwLS1M` Playing the 8 BEST Cards in a SINGLE Deck: 7
  - `B7SW-i94v8s` My MAIN Deck for Season End in Clash Royale : 6
  - `CInrqMTlVkg` Minion Giant is the Most OVERPOWERED Card!!: 4
  - `CdlYCOppdXQ` Jynxzi Challenged Me in C.H.A.O.S Mode: 1
  - `GQmC6dsl6Go` #1 BEST Log Bait Deck in Clash Royale: 8
  - `LGb0mz8Sb8w` TOP 5 BEST Decks After New Balance Changes: 5
  - `O97T50_dNGc` I Made it to DAY 2 of Clash Royale League Monthly Finals: 8
  - `OKA_QNEkJIU` This SUPER FAST 2.0 Log Bait Cycle is BUSTED!!: 7
  - `PuJls1qTZsU` This Piggies Lightning Deck is UNSTOPPABLE!!: 7
  - `SsOmbv1PfUg` This NEW Xbow Deck is TAKING OVER the Meta: 6
  - `VyADrwRPJz8` #1 TOP Ladder Push with Log Bait 🌎🏆: 7
  - `X2zRpx5TN2U` This Rune Giant Deck is TOTALLY OVERPOWERED!!: 8
  - `XM5NEb9SeBY` NERF the Hog Rider NOW!!!: 6
  - `Z8-4VhLjrGU` EMERGENCY Balance Changes Announced!!: 4
  - `Zt6Onn89EVk` Hero Valkyrie is the Most BROKEN Hero of All Time: 5
  - `eEwTOkVPoFM` Ronin Made This Log Bait Deck 100X MORE BROKEN: 7
  - `kiN_D3bRu34` #1 BEST Ronin Deck in Clash Royale: 8
  - `klMJBAH2Zx8` I Am #1 in the World with this GAME BREAKING Deck 🌎🏆: 6
  - `nSXIs16M7Ag` Hero Ice Wizard is SKILL and OVERPOWERED!!: 3
  - `qHWw4rM-N2I` The Most UNDERRATED Card in Clash Royale!!: 7
  - `rTamPgdaLUs` Hero Berserker is INVINCIBLE!!!: 4
  - `tizujccrTvE` I Got a 96% WIN Rate with this BROKEN Deck: 8
  - `vp7jP_vjxcU` Ryley's BEST Predictions of 2026!: 2
  - `yOGPFqpzNu4` TOP 10 Best C.H.A.O.S Decks in Clash Royale: 5
  - `ynafcLtQWDQ` I Got a 98% WIN Rate with this Royal Hogs Deck: 5
  - `zimGbIvEs1s` This Elite Barbs Evolution Deck is UNBEATEN: 6
- Downloaded but no match file:
  - `TSK-yPHwwtQ` I Am UNDEFEATED in Ken's K.H.A.O.S Mode
  - `kWTdrSkGfr8` This Hero Valkyrie Deck is BROKEN!!
  - `EYZBR5GxSec` Evolved Elite Barbarians DESTROY EVERYTHING!!
  - `l4m-QFYIREk` Playing the Most BRAINDEAD Deck in Clash Royale
  - `yCHG5misFWQ` 35 Cards Are Getting CHANGED (Spirits are DEAD!!)
  - `3P-7Euti0qE` This TOXIC Deck Has ZERO Bad Matchups
  - `uwu19wDpHOU` BEATING the BEST Players in Clash Royale League
  - `tVtbO4bKfqw` I Played the BEST Chaos Modifier for Every Rarity
  - `7Fa2CsEXXSM` I COACHED Jynxzi to Reach Ultimate Champion 
  - `YxwncpRk2eU` The JYNXZI and RYLEY DUO is BACK!!! 
  - `p2MSIPdQqnM` This Royal Hogs Evolution Deck Has a 100% WIN Rate
  - `qIcm9pQ8RlM` The Newly BUFFED Baby Dragon is BROKEN!!
  - `vhuWno40_jI` TOP 10 Decks for the New Meta After Balance Changes
  - `LCn9x-_8F-w` The Most BROKEN Modifier in C.H.A.O.S Mode
  - `3aBW7pN5sg0` Can I Get TOP 1 in the World 🌎🏆

Notes on the uncovered videos: `7Fa2CsEXXSM` is a landscape coaching stream with no readable game panel; `TSK-yPHwwtQ` and `LCn9x-_8F-w` are C.H.A.O.S.-mode videos (modified arena; the portrait-panel fix landed after the first was scanned and it was not re-run); the rest were simply not reached before the stop. Pipeline: `tools/batch_process.py` (perception, contexts), one Opus agent per game per `scripts/kb/AGENT_INSTRUCTIONS_MATCHES.md`, `tools/merge_insights.py` (card/deck blocks), `tools/ingest_worker.py` (worker branches `phase4/worker-{1,3,4}`).

## Phase 2 data: game-state enrichment (closed 2026-09-05)

Every sample in `knowledge_base/states/*.jsonl` carries an `enrichment` block written by one Opus agent per game (five remote worker sessions, `phase5/sworker-1..5`, ingested with `tools/ingest_states.py`). Validator: `python3 tools/validate_states.py --summary` reports 0 errors.

| kind | samples | enriched | low-confidence |
|---|---|---|---|
| key | 3,539 | 3,539 | 92 (2.6%) |
| play | 6,901 | 6,901 | 2,636 (38.2%) |
| periodic | 483 | 483 | 139 (28.8%) |
| total | 10,923 | 10,923 | 2,867 (26.2%) |

- Fields present: `situation_read`, `reaction`, `confidence`, `tags` on all samples; `pro_action_rationale`, `principle`, `alternatives` on all key samples (2 key samples lack principle/alternatives); `outcome_note` on 6,089.
- Median enrichment size: key 1,332 bytes, play 406, periodic 453.
- Outcome verdicts (from the extractor, not the agents): neutral 5,717, positive 3,150, negative 2,056.
- Confidence: high 2,392, medium 5,664, low 2,867. Low confidence in play samples is dominated by `unknown-action`/`unknown-play` (hand read unusable, card off-deck) and by known perception noise the agents called out in `outcome_note` (tower-HP flicker, side mis-attribution, double-counted spells).
- Tags: 3,070 distinct, free-vocabulary. The `overtime` tag agrees with `state.phase` on 98% of samples. Top tags: overtime, triple-elixir, cycle, hold, counter-push, tower-race, low-elixir, double-elixir, defend-bridge-push, punish.

Known limitations for retrieval use:
- Tag vocabulary is uncontrolled; normalise (or drop tags) before using them as filters.
- `state.phase` is null on 2,017 samples (no clock read); those samples still have `match_seconds` and `clock: null`.
- Play samples with `action.type = unknown` (475 tagged) have a reaction written from the visible state only; treat them as weaker supervision than key samples.
- Worker cost: about $540 across the five sessions (sub-agents included).
