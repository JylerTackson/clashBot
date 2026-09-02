# Match context: Playing the 8 BEST Cards in a SINGLE Deck (video A5qMlVwLS1M, match 0.2)

- Video time 569.7s to 733.7s (164.0s readable). Calibration: towers.
- Own deck observed (Ryley): Barbarian Barrel, Mother Witch, Goblinstein, Zap, Goblins, Heal Spirit, Tornado, Battle Ram (deck_key `barbarian-barrel-battle-ram-goblins-goblinstein-heal-spirit-mother-witch-tornado-zap`)
- Own deck, video-level consensus across 7 game(s) of this video: Barbarian Barrel, Heal Spirit, Zap, Cannon Cart, Battle Ram, Elite Barbarians, Mother Witch, Goblinstein (deck_key `barbarian-barrel-battle-ram-cannon-cart-elite-barbarians-goblinstein-heal-spirit-mother-witch-zap`); use this when the per-game read above is incomplete unless the commentary says he switched decks
- Hero variants possible: Barbarian Barrel as a Hero has the ability 'Rowdy Reroll' costing 1 elixir (an own elixir drop of 1 with no hand change is usually this ability, not a play)
- Cards named in this game's commentary (mentions): Battle Ram 7, Mother Witch 6, Lava Hound 6, Berserker 5, Zap 5, Fireball 4, Tombstone 3, Mortar 3, Barbarian Barrel 2, Royal Hogs 2, Valkyrie 2, Elite Barbarians 1, Spirit Empress 1, Goblin Barrel 1, Goblin Drill 1
- Opponent cards seen: Tombstone, Skeletons, Mother Witch, Witch, Battle Ram, Lava Hound, Elite Barbarians, Barbarian Barrel (complete)
- Quality: events 54 (6 unidentified), sources {'hud': 19, 'deploy_label': 19, 'arena': 10, 'inferred': 6}, hand confidence 0.519, own-elixir drift {'n': 300, 'mean': 0.375, 'abs_mean': 1.004, 'max_abs': 8.106, 'last': -1.285}

Tile coordinates: (col 0-17 left to right, row 0-31 bottom to top); Ryley's half is rows 0-14, river 15-16, opponent half 17-31. Unit positions are Kalman-tracked estimates (heading and 2 s prediction from a constant-velocity model seeded with the card's speed class); 'THREATS' lists enemy units within ~8 s of a tower. Detections come from a detector frozen in 2024 (newer cards may be missing or 'unknown_unit'); deployments from the in-game deploy label are reliable for any card.

## Plays

- t=572.6 clock 2:48 **own** Battle Ram at tile [2, 13] (elixir 9.0->6.0) [hud/medium] slot 3 emptied (battle-ram) but elixir -3 != cost 4.0; deploy label 'BattleRaM' at (2, 13) (score 1.0)
- t=585.8 clock 2:35 **own** Barbarian Barrel at tile [3, 13] (elixir 10.0->8.0) [deploy_label/medium] deploy label 'aRbaRiaN BaRRel' -> barbarian-barrel (score 0.966) at (3, 13) explains elixir -2 (hand read disagreed)
- t=592.6 clock 2:27 **opponent** Tombstone at tile [8, 20] (elixir 10.0->7.0) [arena/high] enemy-side track
- t=592.3 clock 2:28 **own** Barbarian Barrel at tile None (elixir 9.0->5.0) [hud/medium] slot 1 emptied (barbarian-barrel) but elixir -4 != cost 2.0
- t=590.6 clock None **own** Goblinstein at tile [15, 1] (elixir None->None) [deploy_label/medium] deploy label 'GobliNste' at (15, 1) (score 0.9); no HUD slot change read (hand read missed it)
- t=594.6 clock 2:25 **opponent** Skeletons at tile [8, 19] (elixir 7.71->6.71) [arena/high] enemy-side track
- t=591.6 clock None **own** Goblins at tile [16, 1] (elixir None->None) [deploy_label/medium] deploy label 'GobliNs' at (16, 1) (score 1.0); no HUD slot change read (hand read missed it)
- t=598.6 clock 2:21 **opponent** Skeletons at tile [5, 16] (elixir 8.14->7.14) [arena/high] enemy-side track
- t=605.1 clock 2:16 **own** Elite Barbarians at tile [14, 13] (elixir 9.0->3.0) [hud/high] slot 1 emptied (elite-barbarians), elixir -6; deploy label 'Elite BaRbaRiaNs' at (14, 13) (score 1.0)
- t=610.8 clock 2:11 **own** UNIDENTIFIED at tile None (elixir 5.0->3.0) [inferred/low] elixir dropped by 2 with no readable hand change
- t=612.6 clock None **own** Mother Witch at tile [8, 6] (elixir None->None) [deploy_label/medium] deploy label 'MotHeR WitcH' at (8, 6) (score 1.0); no HUD slot change read (hand read missed it)
- t=619.6 clock 2:02 **opponent** Mother Witch at tile [8, 13] (elixir 10.0->6.0) [deploy_label/high] deploy label 'MotHeR WitcH' on own half, lvl 16
- t=621.6 clock 2:00 **opponent** Witch at tile [10, 15] (elixir 6.95->1.95) [deploy_label/high] deploy label 'WitCH'
- t=622.6 clock 1:59 **opponent** Battle Ram at tile [13, 14] (elixir 4.0->0.0) [deploy_label/high] deploy label 'Battle RaM' on own half, lvl 16
- t=623.6 clock 1:58 **opponent** Battle Ram at tile [3, 13] (elixir 4.0->0.0) [deploy_label/high] deploy label 'Battle RaM' on own half, lvl 16
- t=621.5 clock 2:00 **own** Tornado at tile None (elixir 6.0->3.0) [hud/high] slot 0 emptied (tornado), elixir -3
- t=622.6 clock 1:57 **opponent** Skeletons at tile [4, 14] (elixir 1.0->0.0) [arena/medium] enemy-side track but spawned on own half (miner/barrel/spell?)
- t=624.3 clock 1:57 **own** UNIDENTIFIED at tile None (elixir 4.0->1.0) [inferred/low] elixir dropped by 3 with no readable hand change
- t=626.6 clock 1:53 **opponent** Lava Hound at tile [17, 26] (elixir 7.0->0.0) [arena/high] enemy-side track
- t=631.5 clock 1:50 **own** Heal Spirit at tile None (elixir 2.0->1.0) [hud/high] slot 3 emptied (heal-spirit), elixir -1
- t=628.6 clock None **own** Barbarian Barrel at tile [3, 15] (elixir None->None) [deploy_label/medium] deploy label 'aRbaRiaNBaRRel' at (3, 15) (score 0.966); no HUD slot change read (hand read missed it)
- t=649.6 clock 1:32 **opponent** Elite Barbarians at tile [4, 20] (elixir 10.0->4.0) [deploy_label/high] deploy label 'Elite BagbaRiaNS'
- t=647.8 clock 1:34 **own** Heal Spirit at tile [3, 15] (elixir 6.0->5.0) [hud/high] slot 3 emptied (heal-spirit), elixir -1; deploy label 'Heal SpiRit' at (3, 15) (score 1.0)
- t=649.8 clock 1:32 **own** Fireball at tile None (elixir 5.0->1.0) [hud/high] slot 3 emptied (fireball), elixir -4
- t=655.8 clock 1:26 **own** Heal Spirit at tile None (elixir 2.0->1.0) [hud/high] slot 2 emptied (heal-spirit), elixir -1
- t=666.7 clock 1:15 **opponent** UNIDENTIFIED at tile None (elixir None->None) [inferred/low] own_right lost 579 HP with no enemy unit in range: unidentified spell
- t=667.7 clock 1:14 **opponent** UNIDENTIFIED at tile None (elixir None->None) [inferred/low] own_right lost 494 HP with no enemy unit in range: unidentified spell
- t=671.7 clock 1:10 **opponent** Barbarian Barrel at tile [11, 2] (elixir 10.0->8.0) [deploy_label/high] deploy label 'BaRbaRiaN BaRRel' on own half, lvl 16
- t=670.7 clock 1:11 **own** Mega Minion at tile None (elixir 5.0->2.0) [hud/high] slot 1 emptied (mega-minion), elixir -3
- t=672.3 clock 1:10 **own** Heal Spirit at tile None (elixir 2.0->1.0) [hud/high] slot 2 emptied (heal-spirit), elixir -1
- t=669.7 clock None **own** Mother Witch at tile [13, 2] (elixir None->None) [deploy_label/medium] deploy label 'MotHeR WitCH' at (13, 2) (score 1.0); no HUD slot change read (hand read missed it)
- t=685.7 clock 0:56 **opponent** Goblinstein at tile [8, 19] (elixir 10.0->5.0) [deploy_label/high] deploy label 'GobliNsteiN', lvl 16
- t=686.7 clock 0:54 **own** UNIDENTIFIED at tile None (elixir 6.0->2.0) [inferred/low] elixir dropped by 4 with no readable hand change
- t=687.7 clock 0:54 **opponent** Battle Ram at tile [3, 19] (elixir 7.14->3.14) [deploy_label/high] deploy label 'BattleRaM', lvl 16
- t=689.5 clock 0:52 **own** The Log at tile None (elixir 3.0->1.0) [hud/high] slot 0 emptied (the-log), elixir -2
- t=691.2 clock 0:50 **own** Heal Spirit at tile [8, 18] (elixir 2.0->1.0) [hud/high] slot 1 emptied (heal-spirit), elixir -1; deploy label 'Heal SpiRit' at (8, 18) (score 1.0)
- t=693.9 clock 0:47 **own** Heal Spirit at tile None (elixir 2.0->1.0) [hud/high] slot 0 emptied (heal-spirit), elixir -1
- t=694.7 clock 0:45 **opponent** Inferno Tower at tile [8, 18] (elixir 10.0->5.0) [arena/high] enemy-side track
- t=692.7 clock None **own** Zap at tile [9, 22] (elixir None->None) [deploy_label/medium] deploy label 'Zap' at (9, 22) (score 1.0); no HUD slot change read (hand read missed it)
- t=696.4 clock 0:45 **own** Heal Spirit at tile None (elixir 2.0->1.0) [hud/high] slot 1 emptied (heal-spirit), elixir -1
- t=695.7 clock None **own** Barbarian Barrel at tile [8, 19] (elixir None->None) [deploy_label/medium] deploy label 'BaRbaRiaN BaRRel' at (8, 19) (score 1.0); no HUD slot change read (hand read missed it)
- t=704.7 clock 0:35 **opponent** Skeleton Dragons at tile [5, 24] (elixir 10.0->6.0) [arena/high] enemy-side track
- t=708.7 clock 0:32 **own** Tornado at tile None (elixir 8.0->5.0) [hud/high] slot 2 emptied (tornado), elixir -3
- t=710.9 clock 0:31 **own** Mother Witch at tile [2, 7] (elixir 6.0->2.0) [hud/high] slot 2 emptied (mother-witch), elixir -4; deploy label 'THeR WiTCH' at (2, 7) (score 0.9)
- t=711.9 clock 0:28 **own** Heal Spirit at tile None (elixir 2.0->1.0) [hud/high] slot 2 emptied (heal-spirit), elixir -1
- t=709.7 clock None **own** Goblinstein at tile [5, 8] (elixir None->None) [deploy_label/medium] deploy label 'GobliNsteiN' at (5, 8) (score 1.0); no HUD slot change read (hand read missed it)
- t=717.6 clock 0:25 **own** UNIDENTIFIED at tile None (elixir 4.0->1.0) [inferred/low] elixir dropped by 3 with no readable hand change
- t=722.2 clock 0:19 **own** Heal Spirit at tile [8, 20] (elixir 3.0->2.0) [hud/high] slot 0 emptied (heal-spirit), elixir -1; deploy label at (8, 20) (score 0.842)
- t=722.7 clock 0:17 **opponent** Tombstone at tile [10, 20] (elixir 10.0->7.0) [arena/high] enemy-side track
- t=723.9 clock 0:18 **own** Fire Spirit at tile None (elixir 3.0->2.0) [hud/high] slot 1 emptied (fire-spirit), elixir -1
- t=723.7 clock None **own** Barbarian Barrel at tile [8, 20] (elixir None->None) [deploy_label/medium] deploy label 'BaRbaRiaN BaRRel' at (8, 20) (score 1.0); no HUD slot change read (hand read missed it)
- t=724.7 clock None **own** Zap at tile [6, 27] (elixir None->None) [deploy_label/medium] deploy label 'Zap' at (6, 27) (score 1.0); no HUD slot change read (hand read missed it)
- t=728.7 clock 0:13 **opponent** Skeletons at tile [11, 18] (elixir 10.0->9.0) [arena/high] enemy-side track
- t=728.7 clock 0:13 **opponent** Barbarians at tile [10, 21] (elixir 9.0->4.0) [arena/high] enemy-side track

## Timeline with commentary

Every 2 s: clock, Ryley's elixir / opponent estimate, hand, units on the field (class, side, tile). Commentary lines (from the auto-transcript) are placed at the time they were spoken.

> [567s] That was close. We're in the next match
> [570s] here, guys. up against Marsh. Just going
- 570s clock 2:52 single_elixir | elixir 9 / opp~5.0 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Battle Ram'] next Cannon Cart | towers own {'king': None, 'left': None, 'right': None} enemy {'king': None, 'left': None, 'right': None} | 
- 570s clock None single_elixir | elixir 9 / opp~5.12 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Battle Ram'] next Cannon Cart | towers own {'king': None, 'left': None, 'right': None} enemy {'king': None, 'left': None, 'right': None} | 
> [572s] to start off with the battle ram here at
- 572s clock 2:50 single_elixir | elixir 9 / opp~5.83 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Barbarian Barrel'] next Cannon Cart | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | 
> [574s] the bridge. This guy usually plays with
- 574s clock 2:47 single_elixir | elixir 6 / opp~6.55 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | 
> [576s] Goblin Drill Bowler, but last time I
- 576s clock 2:47 single_elixir | elixir 6 / opp~7.26 | hand ['Mother Witch', 'Barbarian Barrel', 'Heal Spirit', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | 
> [578s] played him, he did have this like mortar
- 578s clock 2:43 single_elixir | elixir 7 / opp~7.98 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(3, 14) advancing left lane
> [580s] berserker deck. Actually, maybe didn't
- 580s clock 2:42 single_elixir | elixir 8 / opp~8.69 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(3, 18) advancing left lane
> [582s] have the Berserker, but I know that he
- 582s clock 2:38 single_elixir | elixir 9 / opp~9.41 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(3, 21) advancing left lane, elite-barbarian(a)@(2, 20) advancing left lane
> [583s] had some type of mortar deck, but mortar
- 584s clock 2:37 single_elixir | elixir 9 / opp~10.0 | hand ['-', 'Barbarian Barrel', 'Heal Spirit', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(3, 19) advancing left lane, elite-barbarian(a)@(2, 22) advancing left lane
> [585s] doesn't usually run Zap, so I'm not
> [586s] really too sure what this could be. I
- 586s clock 2:35 single_elixir | elixir 8 / opp~10.0 | hand ['-', 'Goblinstein', 'Heal Spirit', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(4, 20) advancing left lane, elite-barbarian(a)@(3, 18) retreating
> [587s] will barbarian barrel though. Berserker,
- 588s clock 2:33 single_elixir | elixir 8 / opp~10.0 | hand ['Mother Witch', 'Goblinstein', 'Heal Spirit', 'Cannon Cart'] next Flying Machine | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(2, 22) advancing left lane, elite-barbarian(a)@(3, 18) retreating
> [589s] Evo, Zap, and Tombstone. Okay, looks
- 590s clock 2:32 single_elixir | elixir 9 / opp~10.0 | hand ['-', 'Goblinstein', 'Heal Spirit', 'Cannon Cart'] next Flying Machine | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(2, 24) advancing left lane
> [591s] like Lava Hound. I'm going to
- 592s clock 2:28 single_elixir | elixir 5 / opp~10.0 | hand ['-', 'Goblinstein', 'Heal Spirit', 'Cannon Cart'] next Flying Machine | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(8, 19) moving right (crossing lanes)
> [593s] goblinstein in the back here just
- 594s clock 2:27 single_elixir | elixir 5 / opp~10.0 | hand ['-', 'Executioner', 'Heal Spirit', 'Cannon Cart'] next Minions | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(10, 18) moving right (crossing lanes), tombstone(e)@(8, 20), elite-barbarian(a)@(4, 19)
> [595s] because if he has lava then I can get a
- 596s clock 2:25 single_elixir | elixir 6 / opp~7.54 | hand ['-', 'Elite Barbarians', 'Heal Spirit', 'Cannon Cart'] next Minions | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(7, 19) retreating, tombstone(e)@(8, 20), skeleton(e)@(8, 19) moving right (crossing lanes), skeleton(e)@(7, 18) advancing left lane, elite-barbarian(a)@(5, 21) advancing left lane
> [597s] really nice spam push right now because
- 598s clock 2:24 single_elixir | elixir 7 / opp~7.25 | hand ['-', 'Elite Barbarians', 'Heal Spirit', 'Cannon Cart'] next Minions | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | battle-ram(a)@(7, 18) retreating, tombstone(e)@(8, 20), skeleton(e)@(9, 19) moving right (crossing lanes), skeleton(e)@(4, 15) moving left (crossing lanes), elite-barbarian(a)@(5, 23) advancing left lane, unknown_unit(a)@(14, 3) advancing right lane
> [599s] he doesn't have the tombstone and cycle
- 600s clock 2:22 single_elixir | elixir 7 / opp~7.97 | hand ['-', 'Elite Barbarians', 'Heal Spirit', 'Cannon Cart'] next Minions | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | tombstone(e)@(8, 20), skeleton(e)@(7, 19), skeleton(e)@(3, 14) moving left (crossing lanes), unknown_unit(a)@(13, 4) advancing right lane, skeleton(e)@(3, 10) advancing left lane | THREATS: skeleton(e) advancing left lane, tower in 2.6s
> [601s] for my ebar. He's going to be in a weird
- 602s clock 2:20 single_elixir | elixir 8 / opp~7.68 | hand ['-', 'Elite Barbarians', 'Heal Spirit', 'Cannon Cart'] next Minions | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | tombstone(e)@(8, 20), skeleton(e)@(7, 19), skeleton(e)@(4, 15) moving left (crossing lanes), skeleton(e)@(2, 6) advancing left lane, skeleton(e)@(4, 15) advancing left lane | THREATS: skeleton(e) advancing left lane, tower in 0.8s; skeleton(e) advancing left lane, tower in 7.9s
> [603s] spot and he can't fireball that my
> [604s] scientist way either. Just go for the
- 604s clock 2:18 single_elixir | elixir 9 / opp~8.4 | hand ['-', 'Goblinstein', 'Heal Spirit', 'Cannon Cart'] next Minions | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | tombstone(e)@(8, 20), skeleton(e)@(7, 19), skeleton(e)@(4, 15) moving left (crossing lanes), skeleton(e)@(2, 10) advancing left lane | THREATS: skeleton(e) advancing left lane, tower in 2.2s
> [606s] ebarbs here at the bridge. He can't
- 606s clock 2:16 single_elixir | elixir 3 / opp~9.11 | hand ['-', 'Zap', 'Heal Spirit', 'Heal Spirit'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | tombstone(e)@(8, 20), skeleton(e)@(7, 19), skeleton(e)@(4, 16), skeleton(e)@(0, 6) advancing left lane, skeleton(e)@(4, 16) moving left (crossing lanes) | THREATS: skeleton(e) advancing left lane, tower in 1.0s
> [607s] really lava right now either. Like he
- 608s clock 2:14 single_elixir | elixir 4 / opp~9.83 | hand ['-', 'Zap', 'Heal Spirit', 'Cannon Cart'] next Battle Ram | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | tombstone(e)@(8, 20), skeleton(e)@(7, 19), skeleton(e)@(4, 16), skeleton(e)@(2, 10) advancing left lane, unknown_unit(a)@(3, 13), unknown_unit(a)@(14, 15) advancing right lane | THREATS: skeleton(e) advancing left lane, tower in 2.3s
> [609s] can't lava the bridge to protect or
- 610s clock 2:12 single_elixir | elixir 5 / opp~10.0 | hand ['-', 'Zap', 'Heal Spirit', 'Cannon Cart'] next Battle Ram | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | tombstone(e)@(8, 20), skeleton(e)@(7, 19), skeleton(e)@(4, 16), skeleton(e)@(0, 6) advancing left lane, unknown_unit(a)@(3, 13), unknown_unit(a)@(13, 17) advancing right lane, skeleton(e)@(4, 15) advancing left lane | THREATS: skeleton(e) advancing left lane, tower in 1.0s; skeleton(e) advancing left lane, tower in 7.7s
> [610s] anything like that. Okay, let's go for
> [611s] the Stein ability here just to make sure
- 612s clock 2:10 single_elixir | elixir 4 / opp~10.0 | hand ['Tornado', 'Zap', 'Heal Spirit', 'Fireball'] next Battle Ram | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4858} | tombstone(e)@(8, 20), skeleton(e)@(9, 19) moving right (crossing lanes), skeleton(e)@(4, 16), unknown_unit(a)@(3, 13), knight(a)@(11, 19) advancing right lane, skeleton(e)@(2, 10) advancing left lane | THREATS: skeleton(e) advancing left lane, tower in 2.3s
> [613s] that we take out that Spirit Empress.
- 614s clock 2:07 single_elixir | elixir 4 / opp~10.0 | hand ['-', 'Zap', 'Heal Spirit', 'Cannon Cart'] next Battle Ram | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4546} | tombstone(e)@(8, 20), skeleton(e)@(10, 19) moving right (crossing lanes), unknown_unit(a)@(3, 13), knight(a)@(10, 21) advancing right lane, skeleton(e)@(0, 6) advancing left lane | THREATS: skeleton(e) advancing left lane, tower in 1.1s
- 616s clock 2:05 single_elixir | elixir 5 / opp~10.0 | hand ['-', 'Zap', 'Heal Spirit', 'Cannon Cart'] next Battle Ram | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4402} | tombstone(e)@(8, 20), skeleton(e)@(10, 20) moving right (crossing lanes), knight(a)@(8, 21) moving left (crossing lanes), unknown_unit(a)@(11, 19) advancing right lane, unknown_unit(a)@(14, 11) retreating, unknown_unit(a)@(12, 19) advancing right lane
> [617s] And I'm tempted to go for a Mother Witch
- 618s clock 2:04 single_elixir | elixir 6 / opp~10.0 | hand ['-', 'Zap', 'Heal Spirit', 'Cannon Cart'] next Battle Ram | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4366} | tombstone(e)@(8, 20), skeleton(e)@(11, 20) moving right (crossing lanes), knight(a)@(7, 22) moving left (crossing lanes), elite-barbarian(a)@(11, 17) retreating, skeleton(a)@(14, 15) advancing right lane, unknown_unit(a)@(12, 21) advancing right lane, skeleton(e)@(10, 16) advancing right lane
> [618s] on these tombstone skellies. H. Yeah,
- 620s clock 2:02 single_elixir | elixir 6 / opp~6.18 | hand ['Fire Spirit', 'Zap', 'Heal Spirit', 'Cannon Cart'] next Battle Ram | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4366} | tombstone(e)@(8, 20), skeleton(e)@(8, 20) moving left (crossing lanes), elite-barbarian(a)@(10, 19) advancing right lane, skeleton(a)@(15, 16) advancing right lane, skeleton(e)@(11, 14) advancing right lane | THREATS: skeleton(e) advancing right lane, tower in 7.8s
> [621s] let's do it to be honest because we can
- 622s clock 2:00 double_elixir | elixir 3 / opp~2.31 | hand ['Fire Spirit', 'Zap', 'Heal Spirit', 'Heal Spirit'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 4366} | tombstone(e)@(8, 20), skeleton(e)@(7, 19) moving left (crossing lanes), elite-barbarian(a)@(5, 20) moving left (crossing lanes), skeleton(a)@(15, 17) advancing right lane
> [623s] just go for a battle ram. Yes, let's go
- 624s clock 1:58 double_elixir | elixir 1 / opp~0.36 | hand ['Fire Spirit', 'Zap', 'Heal Spirit', 'Zap'] next Barbarian Barrel | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 3934} | skeleton(e)@(4, 16) moving left (crossing lanes), elite-barbarian(a)@(3, 22) moving left (crossing lanes), skeleton(e)@(4, 15) advancing left lane | THREATS: skeleton(e) moving left (crossing lanes), tower in 7.1s
> [624s] for a battle ram. Perfect. I had a
> [626s] feeling there was a chance he might go
- 626s clock 1:56 double_elixir | elixir 1 / opp~1.07 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 3718} | skeleton(e)@(4, 15) advancing left lane, elite-barbarian(a)@(2, 16) moving left (crossing lanes), skeleton(e)@(3, 13) advancing left lane, unknown_unit(a)@(16, 18) moving right (crossing lanes), skeleton(e)@(2, 10) advancing left lane | THREATS: skeleton(e) advancing left lane, tower in 7.2s; skeleton(e) advancing left lane, tower in 3.0s
> [628s] for the lava hound in the back as well.
- 628s clock 1:54 double_elixir | elixir 2 / opp~2.5 | hand ['Giant', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4858, 'right': 3502} | skeleton(e)@(3, 14) advancing left lane, elite-barbarian(a)@(0, 15) moving left (crossing lanes), unknown_unit(a)@(17, 20) moving right (crossing lanes), skeleton(e)@(2, 7) advancing left lane, lava-hound(e)@(17, 26) advancing right lane | THREATS: skeleton(e) advancing left lane, tower in 1.3s
> [630s] [clears throat] That's huge. He's going
- 630s clock 1:52 double_elixir | elixir 2 / opp~1.07 | hand ['Skeletons', 'Zap', 'Heal Spirit', 'Heal Spirit'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 4518, 'right': 3502} | lava-hound(e)@(17, 24) advancing right lane, battle-ram(a)@(3, 17) advancing left lane, unknown_unit(a)@(4, 12) moving left (crossing lanes)
> [630s] to pop the berserker ability just to
- 632s clock 1:50 double_elixir | elixir 2 / opp~2.5 | hand ['Fire Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 3347, 'right': 3502} | lava-hound(e)@(16, 23) advancing right lane, battle-ram(a)@(3, 23) advancing left lane, unknown_unit(a)@(1, 21) moving left (crossing lanes), unknown_unit(a)@(2, 12) moving left (crossing lanes)
> [632s] keep it alive. But I'm going to
> [633s] barbarian barrel the bridge too just to
- 634s clock 1:49 double_elixir | elixir 2 / opp~3.93 | hand ['Heal Spirit', 'Zap', 'Heal Spirit', 'Heal Spirit'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 3262, 'right': 3502} | lava-hound(e)@(17, 21) advancing right lane, battle-ram(a)@(3, 27) advancing left lane, elite-barbarian(a)@(3, 20) moving right (crossing lanes)
> [635s] stay tanky from my mother witch cuz
- 636s clock 1:46 double_elixir | elixir 2 / opp~5.36 | hand ['Heal Spirit', 'Zap', 'Heal Spirit', 'Heal Spirit'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 3092, 'right': 3502} | lava-hound(e)@(17, 19) advancing right lane, elite-barbarian(a)@(3, 20) retreating, barbarian-barrel(e)@(3, 15)
> [637s] that's going to spawn a piggy in a
> [638s] second. And yes, it's going to be a
- 638s clock 1:45 double_elixir | elixir 3 / opp~4.79 | hand ['Heal Spirit', 'Zap', 'Heal Spirit', 'Heal Spirit'] next Skeletons | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 3092, 'right': 3502} | lava-hound(e)@(16, 18) advancing right lane, elite-barbarian(a)@(3, 16) retreating, barbarian-barrel(e)@(3, 15)
> [640s] crazy amount of damage. That might even
- 640s clock 1:42 double_elixir | elixir 3 / opp~6.22 | hand ['Heal Spirit', 'Zap', 'Heal Spirit', 'Fireball'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 2192, 'right': 3502} | lava-hound(e)@(15, 16) advancing right lane, elite-barbarian(a)@(3, 14) retreating
> [641s] be tower down. I'm not 100%. Oh, no. I
- 642s clock 1:40 double_elixir | elixir 4 / opp~7.65 | hand ['-', 'Zap', 'Heal Spirit', 'Cannon Cart'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 1598, 'right': 3502} | lava-hound(e)@(16, 13) advancing right lane, unknown_unit(a)@(16, 14) retreating
- 644s clock 1:38 double_elixir | elixir 5 / opp~9.08 | hand ['Goblinstein', 'Zap', 'Heal Spirit', 'Cannon Cart'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 1301, 'right': 3502} | lava-hound(e)@(16, 12) advancing right lane, unknown_unit(a)@(16, 13) retreating
> [644s] am 100% sure. That is most definitely
> [646s] tower down. What a beautiful predict
- 646s clock 1:36 double_elixir | elixir 6 / opp~10.0 | hand ['Goblinstein', 'Zap', 'Heal Spirit', 'Cannon Cart'] next ? | towers own {'king': None, 'left': 4858, 'right': 4858} enemy {'king': None, 'left': 495, 'right': 3502} | lava-hound(e)@(16, 10) advancing right lane, firecracker(a)@(2, 24) advancing left lane | THREATS: lava-hound(e) advancing right lane, tower in 5.9s
> [648s] right And I'm even going to heal spear
- 648s clock 1:34 double_elixir | elixir 5 / opp~10.0 | hand ['Goblinstein', 'Zap', 'The Log', 'Cannon Cart'] next ? | towers own {'king': None, 'left': 4858, 'right': 4773} enemy {'king': None, 'left': 283, 'right': 3502} | lava-hound(e)@(15, 9) advancing right lane, firecracker(a)@(1, 29) advancing left lane | THREATS: lava-hound(e) advancing right lane, tower in 3.9s
> [649s] and go for the evil Earbs on the king
- 650s clock 1:32 double_elixir | elixir 1 / opp~4.36 | hand ['Zap', 'Mother Witch', 'The Log', 'Cannon Cart'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4603} enemy {'king': None, 'left': 283, 'right': 3502} | firecracker(a)@(3, 29) advancing left lane
> [650s] tower because I can't really defend a
- 652s clock 1:30 double_elixir | elixir 2 / opp~5.79 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Cannon Cart | towers own {'king': None, 'left': 4858, 'right': 4518} enemy {'king': None, 'left': 283, 'right': 3502} | firecracker(a)@(3, 31) advancing left lane, unknown_unit(a)@(16, 11) advancing right lane
> [653s] pusher now. And just by doing this, I'm
> [654s] just forcing him to go in defensively,
- 654s clock 1:28 double_elixir | elixir 2 / opp~7.22 | hand ['Heal Spirit', 'Zap', 'Heal Spirit', 'Heal Spirit'] next Cannon Cart | towers own {'king': None, 'left': 4858, 'right': 4348} enemy {'king': None, 'left': 283, 'right': 3502} | unknown_unit(a)@(16, 11)
> [656s] which is good for us. Let's even zap
- 656s clock 1:26 double_elixir | elixir 1 / opp~8.65 | hand ['Zap', 'Berserker', 'Tornado', 'Heal Spirit'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 4263} enemy {'king': None, 'left': 283, 'right': 3502} | unknown_unit(a)@(16, 11)
> [657s] that here just to get some nice damage
- 658s clock 1:24 double_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next ? | towers own {'king': None, 'left': 4858, 'right': 4093} enemy {'king': None, 'left': 283, 'right': 3502} | unknown_unit(a)@(16, 11), lava-hound(e)@(16, 11), spear-goblin(a)@(9, 25) advancing right lane
> [659s] on the king tower because I know that
- 660s clock 1:22 double_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Barbarian Barrel | towers own {'king': None, 'left': 4858, 'right': 4008} enemy {'king': None, 'left': 283, 'right': 3502} | lava-hound(e)@(16, 11), spear-goblin(a)@(11, 27) advancing right lane, skeleton-dragon(e)@(9, 25) moving left (crossing lanes)
> [661s] defense is not easy for us. And we have
- 662s clock 1:20 double_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Lumberjack'] next Barbarian Barrel | towers own {'king': None, 'left': 4858, 'right': 8585} enemy {'king': None, 'left': 283, 'right': 3502} | lava-hound(e)@(16, 11), skeleton-dragon(e)@(7, 25) moving left (crossing lanes), knight(a)@(7, 26) advancing left lane
> [662s] a very good chance to win if we just get
- 664s clock 1:18 double_elixir | elixir 3 / opp~10.0 | hand ['Heal Spirit', 'Mini P.E.K.K.A.', 'Heal Spirit', 'Heal Spirit'] next Barbarian Barrel | towers own {'king': None, 'left': 4858, 'right': 3753} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(11, 21) advancing right lane, knight(a)@(7, 27) advancing left lane
> [664s] his king tower very low. Not bad. We're
- 666s clock 1:16 double_elixir | elixir 4 / opp~10.0 | hand ['Heal Spirit', 'Cannon Cart', 'Heal Spirit', 'Cannon Cart'] next Barbarian Barrel | towers own {'king': None, 'left': 4858, 'right': 3583} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(12, 15) advancing right lane, skeleton-dragon(e)@(13, 15) advancing right lane, unknown_unit(a)@(12, 20) advancing right lane | THREATS: skeleton-dragon(e) advancing right lane, tower in 5.3s; skeleton-dragon(e) advancing right lane, tower in 5.9s
> [667s] going to have to go for a mother witch
- 668s clock 1:14 double_elixir | elixir 4 / opp~10.0 | hand ['-', 'Cannon Cart', 'Heal Spirit', 'Cannon Cart'] next Barbarian Barrel | towers own {'king': None, 'left': 4858, 'right': 2510} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(13, 11) advancing right lane, skeleton-dragon(e)@(13, 9) advancing right lane, unknown_unit(a)@(11, 25) advancing right lane | THREATS: skeleton-dragon(e) advancing right lane, tower in 2.7s; skeleton-dragon(e) advancing right lane, tower in 1.9s
> [668s] here on defense. So just to not get
> [670s] three crowned cuz being three crowned is
- 670s clock 1:12 double_elixir | elixir 4 / opp~10.0 | hand ['Goblinstein', 'Cannon Cart', 'Heal Spirit', 'Cannon Cart'] next Barbarian Barrel | towers own {'king': None, 'left': 4858, 'right': 1690} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(15, 7) advancing right lane, skeleton-dragon(e)@(14, 5) advancing right lane | THREATS: skeleton-dragon(e) advancing right lane, tower in 1.0s; skeleton-dragon(e) advancing right lane, tower in 0.0s
> [672s] actually a very big worry in this type
- 672s clock 1:10 double_elixir | elixir 1 / opp~8.24 | hand ['Zap', 'Zap', 'Heal Spirit', 'Heal Spirit'] next Heal Spirit | towers own {'king': None, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(16, 3) advancing right lane, skeleton-dragon(e)@(14, 8) advancing right lane, lava-pup(e)@(14, 7) advancing right lane, lava-pup(e)@(16, 7) advancing right lane, unknown_unit(a)@(12, 11) advancing right lane | THREATS: skeleton-dragon(e) advancing right lane, tower in 0.2s; skeleton-dragon(e) advancing right lane, tower in 3.8s; lava-pup(e) advancing right lane, tower in 1.3s; lava-pup(e) advancing right lane, tower in 2.7s
> [673s] of situation. Okay, that was a bad
- 674s clock 1:08 double_elixir | elixir 2 / opp~9.67 | hand ['Zap', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Flying Machine | towers own {'king': 7487, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(11, 5) advancing right lane, lava-pup(e)@(13, 5) advancing right lane, lava-pup(e)@(15, 6) advancing right lane, unknown_unit(a)@(7, 13) moving left (crossing lanes), skeleton-dragon(e)@(11, 7) advancing right lane, lava-pup(e)@(14, 8) advancing right lane | THREATS: skeleton-dragon(e) advancing right lane, tower in 1.1s; lava-pup(e) advancing right lane, tower in 0.1s; lava-pup(e) advancing right lane, tower in 0.7s; skeleton-dragon(e) advancing right lane, tower in 2.3s; lava-pup(e) advancing right lane, tower in 3.8s
> [675s] Valkyrie though because after that
- 676s clock 1:07 double_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Flying Machine | towers own {'king': 6511, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(10, 3) advancing right lane, unknown_unit(a)@(4, 15) moving left (crossing lanes), skeleton-dragon(e)@(11, 4) advancing right lane, lava-pup(e)@(14, 7) advancing right lane, unknown_unit(a)@(13, 7) moving left (crossing lanes), skeleton-evolution(e)@(17, 0) advancing right lane | THREATS: skeleton-dragon(e) advancing right lane, tower in 0.9s; skeleton-dragon(e) advancing right lane, tower in 0.8s; lava-pup(e) advancing right lane, tower in 1.8s; skeleton-evolution(e) advancing right lane, tower in 2.4s
> [676s] Valkyrie, he's not going to be able to
> [677s] afford a fireball on the mother witch. I
- 678s clock 1:05 double_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Flying Machine | towers own {'king': 5041, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(12, 5), skeleton-dragon(e)@(11, 2) advancing right lane, skeleton-dragon(a)@(10, 6) moving left (crossing lanes), skeleton-evolution(e)@(17, 0) advancing right lane, unknown_unit(a)@(14, 12) advancing right lane | THREATS: skeleton-dragon(e) advancing right lane, tower in 1.3s; skeleton-evolution(e) advancing right lane, tower in 4.1s
> [679s] don't even know if he has fireball cuz
> [680s] he hasn't used it yet, of course. But he
- 680s clock 1:02 double_elixir | elixir 3 / opp~10.0 | hand ['Heal Spirit', 'Goblin Barrel', 'Heal Spirit', 'Heal Spirit'] next Flying Machine | towers own {'king': 4559, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(12, 6), skeleton-dragon(e)@(11, 7) retreating, skeleton-dragon(a)@(9, 4) moving left (crossing lanes), unknown_unit(a)@(14, 13) advancing right lane, minion(a)@(15, 17) advancing right lane | THREATS: skeleton-dragon(e) retreating, tower in 6.7s
> [681s] hasn't really gotten an opportunity to
- 682s clock 1:01 double_elixir | elixir 4 / opp~10.0 | hand ['Heal Spirit', 'Cannon Cart', 'Heal Spirit', 'Cannon Cart'] next Flying Machine | towers own {'king': 4318, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3502} | skeleton-dragon(e)@(12, 6), skeleton-dragon(e)@(11, 8) retreating, skeleton-dragon(a)@(7, 3) moving left (crossing lanes), unknown_unit(a)@(14, 19) advancing right lane, minion(a)@(15, 20) advancing right lane, lava-pup(e)@(15, 8) moving right (crossing lanes) | THREATS: lava-pup(e) moving right (crossing lanes), tower in 4.3s
> [683s] play with it either. So, we really are
- 684s clock 0:58 triple_elixir | elixir 5 / opp~10.0 | hand ['Goblinstein', 'Cannon Cart', 'Heal Spirit', 'Cannon Cart'] next Flying Machine | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3332} | skeleton-dragon(e)@(11, 8) retreating, skeleton-dragon(e)@(11, 8) retreating, skeleton-dragon(a)@(8, 0) retreating, skeleton-evolution(a)@(14, 25) advancing right lane, minion(a)@(15, 24) advancing right lane, lava-pup(e)@(16, 7) moving right (crossing lanes), unknown_unit(a)@(13, 17) advancing right lane | THREATS: skeleton-dragon(e) retreating, tower in 7.5s; lava-pup(e) moving right (crossing lanes), tower in 4.6s
> [685s] just in the dark right now. Let's go for
- 686s clock 0:56 triple_elixir | elixir 5 / opp~5.36 | hand ['Tornado', 'Cannon Cart', 'Heal Spirit', 'Cannon Cart'] next Flying Machine | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3162} | skeleton-dragon(e)@(11, 9) retreating, skeleton-dragon(a)@(8, 0) retreating, skeleton-evolution(a)@(14, 30) advancing right lane, unknown_unit(a)@(13, 21) advancing right lane, unknown_unit(a)@(14, 31)
> [686s] the Stein in the pocket here, though.
> [688s] And then we can go for a evil ram on the
- 688s clock 0:54 triple_elixir | elixir 3 / opp~3.5 | hand ['Heal Spirit', 'Berserker', 'Heal Spirit', 'Heal Spirit'] next Minions | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3162} | unknown_unit(a)@(14, 30), unknown_unit(a)@(9, 0)
> [689s] king cuz he might berserker on top of
- 690s clock 0:52 triple_elixir | elixir 1 / opp~5.65 | hand ['Zap', 'Fire Spirit', 'Heal Spirit', 'Heal Spirit'] next ? | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 3077} | unknown_unit(a)@(14, 30), unknown_unit(a)@(9, 0)
> [690s] this. Yeah, we can go for the heal
> [692s] spirit. He might go for the ability.
- 692s clock 0:50 triple_elixir | elixir 2 / opp~7.79 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next ? | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 2822} | unknown_unit(a)@(15, 30)
> [693s] That's fine with me. Let's even go for
- 694s clock 0:48 triple_elixir | elixir 1 / opp~9.94 | hand ['Zap', 'Barbarian Barrel', 'Tornado', 'Zap'] next Barbarian Barrel | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 2567} | unknown_unit(a)@(15, 30), unknown_unit(a)@(9, 16) moving left (crossing lanes)
> [694s] the zap here just to make sure our
- 696s clock 0:47 triple_elixir | elixir 1 / opp~10.0 | hand ['Heal Spirit', 'Skeletons', 'Heal Spirit', 'Heal Spirit'] next Heal Spirit | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 2397} | unknown_unit(a)@(10, 16) moving right (crossing lanes), inferno-tower(e)@(8, 18), unknown_unit(a)@(15, 13) advancing right lane, unknown_unit(a)@(13, 22) advancing right lane
> [697s] scientist stays alive. And we barb.
- 698s clock 0:44 triple_elixir | elixir 2 / opp~6.43 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Battle Ram | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 2108} | unknown_unit(a)@(10, 17) moving right (crossing lanes), inferno-tower(e)@(8, 18), unknown_unit(a)@(15, 16) advancing right lane, rocket(a)@(12, 22) moving left (crossing lanes)
> [698s] Nice. Beautiful. And look at the battle
> [700s] ram, guys. The king tower is a massive
- 700s clock 0:42 triple_elixir | elixir 3 / opp~8.58 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Cannon Cart | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1615} | unknown_unit(a)@(13, 21) advancing right lane, rocket(a)@(11, 23) moving left (crossing lanes), rascal-girl(a)@(6, 26)
> [702s] threat now. He has to watch out for
- 702s clock 0:40 triple_elixir | elixir 4 / opp~10.0 | hand ['Executioner', 'Zap', 'Mother Witch', 'Cannon Cart'] next Cannon Cart | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | unknown_unit(a)@(14, 23) advancing right lane, rocket(a)@(10, 20) retreating, rascal-girl(a)@(5, 24) retreating
> [703s] getting three crown. The barbs. The
- 704s clock 0:38 triple_elixir | elixir 5 / opp~10.0 | hand ['Elite Barbarians', 'Goblinstein', 'Mother Witch', 'Cannon Cart'] next Cannon Cart | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | rocket(a)@(9, 19) retreating, rascal-girl(a)@(1, 22) moving left (crossing lanes)
> [704s] barbs. Okay, nice. We only need a tiny
- 706s clock 0:36 triple_elixir | elixir 7 / opp~10.0 | hand ['Elite Barbarians', 'Goblinstein', 'The Log', 'Cannon Cart'] next Battle Ram | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | rascal-girl(a)@(0, 19) moving left (crossing lanes), skeleton-dragon(e)@(6, 24) moving left (crossing lanes)
> [706s] bit more damage. I'm going to go for the
> [708s] mother witch off to the side cuz he
- 708s clock 0:34 triple_elixir | elixir 4 / opp~7.43 | hand ['Elite Barbarians', 'Goblinstein', 'The Log', 'Cannon Cart'] next Cannon Cart | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | rascal-girl(a)@(2, 21) retreating, skeleton-dragon(e)@(5, 24) moving left (crossing lanes), skeleton-dragon(e)@(2, 19) advancing left lane
> [709s] might lava in the pocket. Yep. Let's go
- 710s clock 0:32 triple_elixir | elixir 6 / opp~9.58 | hand ['Elite Barbarians', 'Fire Spirit', 'Battle Ram', 'Cannon Cart'] next Heal Spirit | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | rascal-girl(a)@(2, 20) retreating, skeleton-dragon(e)@(2, 15) advancing left lane, skeleton-dragon(e)@(4, 14) advancing left lane, valkyrie-evolution(e)@(14, 13) advancing right lane, unknown_unit(a)@(9, 0) advancing right lane | THREATS: skeleton-dragon(e) advancing left lane, tower in 5.4s; skeleton-dragon(e) advancing left lane, tower in 5.1s; valkyrie-evolution(e) advancing right lane, tower in 6.2s
> [711s] for the stein like this. And we can even
> [712s] go for the ability right now instantly
- 712s clock 0:31 triple_elixir | elixir 1 / opp~10.0 | hand ['Zap', 'Heal Spirit', 'Heal Spirit', 'Zap'] next Minions | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | skeleton-dragon(e)@(2, 10) advancing left lane, skeleton-dragon(e)@(4, 12) advancing left lane, valkyrie-evolution(e)@(14, 11) advancing right lane, unknown_unit(a)@(9, 0) advancing right lane, unknown_unit(a)@(15, 31) | THREATS: skeleton-dragon(e) advancing left lane, tower in 2.4s; skeleton-dragon(e) advancing left lane, tower in 4.2s; valkyrie-evolution(e) advancing right lane, tower in 4.2s
- 714s clock 0:28 triple_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Heal Spirit', 'Heal Spirit'] next Minions | towers own {'king': 4077, 'left': 4858, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | skeleton-dragon(e)@(2, 6) advancing left lane, skeleton-dragon(e)@(3, 9) advancing left lane, valkyrie-evolution(e)@(14, 8) advancing right lane, unknown_unit(a)@(5, 7) advancing left lane, witch(e)@(2, 5) retreating, unknown_unit(a)@(15, 31), unknown_unit(a)@(2, 9) advancing left lane | THREATS: skeleton-dragon(e) advancing left lane, tower in 0.4s; skeleton-dragon(e) advancing left lane, tower in 2.2s; valkyrie-evolution(e) advancing right lane, tower in 2.6s; witch(e) retreating, tower in 1.1s
> [714s] cuz if we insta ability, he can't
> [716s] fireball right away. And then we can
- 716s clock 0:27 triple_elixir | elixir 3 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Barbarian Barrel', 'Heal Spirit'] next Minions | towers own {'king': 4077, 'left': 4498, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | skeleton-dragon(e)@(4, 10) advancing left lane, valkyrie-evolution(e)@(14, 6) advancing right lane, unknown_unit(a)@(3, 10) advancing left lane, witch(e)@(2, 6) retreating, unknown_unit(a)@(2, 11) advancing left lane | THREATS: valkyrie-evolution(e) advancing right lane, tower in 1.0s; witch(e) retreating, tower in 2.0s
> [717s] even go for a battle ram here. And this
- 718s clock 0:25 triple_elixir | elixir 1 / opp~10.0 | hand ['Zap', 'Heal Spirit', 'The Log', 'Zap'] next ? | towers own {'king': 4077, 'left': 4413, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | skeleton-dragon(e)@(4, 9) advancing left lane, valkyrie-evolution(e)@(11, 6) moving left (crossing lanes), witch(e)@(9, 10) moving right (crossing lanes), ice-golem(a)@(4, 17) advancing left lane | THREATS: skeleton-dragon(e) advancing left lane, tower in 7.2s; valkyrie-evolution(e) moving left (crossing lanes), tower in 2.1s; witch(e) moving right (crossing lanes), tower in 3.3s
> [718s] is fully defended. He has to watch out
> [720s] for the mother witch again as well cuz
- 720s clock 0:22 triple_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Zap', 'Heal Spirit'] next Barbarian Barrel | towers own {'king': 4077, 'left': 4328, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | valkyrie-evolution(e)@(8, 7) moving left (crossing lanes), witch(e)@(12, 12) moving right (crossing lanes), ice-golem(a)@(5, 21) advancing left lane, unknown_unit(a)@(8, 9) advancing left lane | THREATS: valkyrie-evolution(e) moving left (crossing lanes), tower in 3.5s; witch(e) moving right (crossing lanes), tower in 3.5s
> [721s] it's going to spawn a bunch of piggies.
- 722s clock 0:20 triple_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Zap', 'Heal Spirit'] next Barbarian Barrel | towers own {'king': 4077, 'left': 4198, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | valkyrie-evolution(e)@(5, 7) moving left (crossing lanes), ice-golem(a)@(7, 22) advancing left lane, unknown_unit(a)@(10, 13) advancing right lane, battle-ram(a)@(2, 22) advancing left lane, lava-pup(e)@(6, 9) advancing left lane, lava-pup(e)@(6, 9) advancing left lane, lava-pup(e)@(7, 6) advancing left lane, lava-pup(a)@(8, 9) advancing left lane | THREATS: valkyrie-evolution(e) moving left (crossing lanes), tower in 1.9s; lava-pup(e) advancing left lane, tower in 4.6s; lava-pup(e) advancing left lane, tower in 6.3s; lava-pup(e) advancing left lane, tower in 2.7s
> [723s] Let's go for the heal spear here to heal
> [724s] up the big guy. We can go for the barb
- 724s clock 0:18 triple_elixir | elixir 2 / opp~10.0 | hand ['Heal Spirit', 'Goblinstein', 'Zap', 'Heal Spirit'] next ? | towers own {'king': 4077, 'left': 4198, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | ice-golem(a)@(8, 24) advancing left lane, unknown_unit(a)@(9, 22) advancing right lane, battle-ram(a)@(4, 22) moving right (crossing lanes), lava-pup(e)@(5, 7) advancing left lane, lava-pup(e)@(7, 9) advancing left lane, lava-pup(e)@(8, 4) advancing left lane, lava-pup(a)@(4, 10) moving left (crossing lanes), tombstone(e)@(10, 20) | THREATS: lava-pup(e) advancing left lane, tower in 2.6s; lava-pup(e) advancing left lane, tower in 2.6s
> [725s] barrel. Let me go for the zap here as
- 726s clock 0:16 triple_elixir | elixir 2 / opp~8.43 | hand ['Executioner', 'Heal Spirit', 'Heal Spirit', 'Zap'] next ? | towers own {'king': 4077, 'left': 4198, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | ice-golem(a)@(9, 26) advancing right lane, unknown_unit(a)@(10, 28) advancing right lane, battle-ram(a)@(5, 23) moving right (crossing lanes), lava-pup(e)@(7, 8) advancing left lane, lava-pup(a)@(1, 12) moving left (crossing lanes), tombstone(e)@(10, 20), fireball(a)@(10, 19) advancing right lane
> [727s] well. Battle ram. Connect the the barb.
- 728s clock 0:14 triple_elixir | elixir 3 / opp~10.0 | hand ['Heal Spirit', 'Heal Spirit', 'Tornado', 'Lumberjack'] next Battle Ram | towers own {'king': 4077, 'left': 4198, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | battle-ram(a)@(5, 24) advancing left lane, tombstone(e)@(10, 20), fireball(a)@(11, 24) advancing right lane, bomber-evolution(a)@(3, 11) advancing left lane
> [729s] The barb the barb. Let's go. Good game.
- 730s clock 0:13 triple_elixir | elixir 9 / opp~10.0 | hand ['Giant', '-', 'Fireball', 'Fire Spirit'] next Giant | towers own {'king': 4077, 'left': 4198, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | battle-ram(a)@(6, 24) advancing left lane, tombstone(e)@(10, 20), fireball(a)@(11, 29) advancing right lane, bomber-evolution(a)@(3, 13) advancing left lane, skeleton(e)@(11, 18) advancing right lane, barbarian(e)@(10, 21) advancing right lane
> [730s] Very nice win. We played that great.
- 732s clock 0:13 triple_elixir | elixir 9 / opp~5.43 | hand ['Giant', 'Giant', 'Giant', 'Giant'] next Giant | towers own {'king': 407, 'left': 4198, 'right': 584} enemy {'king': None, 'left': 283, 'right': 1411} | battle-ram(a)@(9, 24) moving right (crossing lanes), tombstone(e)@(10, 20), bomber-evolution(a)@(4, 15) advancing left lane, skeleton(e)@(11, 17) advancing right lane, barbarian(e)@(11, 20) advancing right lane
> [733s] We're in the next match up here, guys,
