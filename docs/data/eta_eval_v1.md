# ETA error (predicted - actual seconds to be in reach of our tower, + = too late)

data/videos/Z8-4VhLjrGU/Z8-4VhLjrGU.mp4 start 60.0 s, 240.0 s, stride 3, 2398 frames in 227.2 s; 70 enemy unit tracks, 11 reached a tower; tracks / arrived by kind: melee 36/4, ranged 18/6, flyer 8/1, unknown 8/0

### All frames before the arrival

| kind | rule | median | p25 | p75 | share abs err <= 1 s | median abs err | n |
|---|---|---|---|---|---|---|---|
| melee | old | 1.30 | 0.80 | 2.40 | 38% | 1.30 | 175 |
| melee | new | 0.30 | -0.10 | 1.19 | 74% | 0.30 | 175 |
| ranged | old | 2.19 | 0.59 | 3.20 | 33% | 2.19 | 192 |
| ranged | new | -0.10 | -0.60 | 0.09 | 83% | 0.40 | 192 |
| flyer | old | 2.30 | 2.30 | 2.40 | 0% | 2.30 | 17 |
| flyer | new | -0.20 | -0.20 | -0.10 | 100% | 0.20 | 17 |
| unknown | old | - | - | - | - | - | 0 |
| unknown | new | - | - | - | - | - | 0 |
| all | old | 1.49 | 0.80 | 3.00 | 34% | 1.49 | 384 |
| all | new | -0.00 | -0.20 | 0.40 | 80% | 0.30 | 384 |

### Frames within 10 s of the arrival

| kind | rule | median | p25 | p75 | share abs err <= 1 s | median abs err | n |
|---|---|---|---|---|---|---|---|
| melee | old | 1.30 | 0.80 | 2.40 | 38% | 1.30 | 175 |
| melee | new | 0.30 | -0.10 | 1.19 | 74% | 0.30 | 175 |
| ranged | old | 2.19 | 0.59 | 3.20 | 33% | 2.19 | 192 |
| ranged | new | -0.10 | -0.60 | 0.09 | 83% | 0.40 | 192 |
| flyer | old | 2.30 | 2.30 | 2.40 | 0% | 2.30 | 17 |
| flyer | new | -0.20 | -0.20 | -0.10 | 100% | 0.20 | 17 |
| unknown | old | - | - | - | - | - | 0 |
| unknown | new | - | - | - | - | - | 0 |
| all | old | 1.49 | 0.80 | 3.00 | 34% | 1.49 | 384 |
| all | new | -0.00 | -0.20 | 0.40 | 80% | 0.30 | 384 |

### Acceptance (new median abs error <= limit and better than old)

- melee: old 1.3 -> new 0.3 (limit 1.0): PASS
- flyer: old 2.3 -> new 0.2 (limit 1.0): PASS
- ranged: old 2.19 -> new 0.4 (limit 1.5): PASS
wrote runs/smoke/AL/eta.json
