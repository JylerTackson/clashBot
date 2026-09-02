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
