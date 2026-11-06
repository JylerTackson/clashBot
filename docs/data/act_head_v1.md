# Act-head evaluation (offline, D-041)

model `db/models/d1-ft-act`, threshold 0.50 (val balanced accuracy 0.573)

## val  (n=3008, pro act rate 0.233)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| learned | 0.554 | 0.573 | 0.496 | 0.286 | 0.608 | 0.818 | 0.538 |
| rules | 0.506 | 0.504 | 0.493 | 0.236 | 0.499 | 0.769 | 0.509 |
| combined | 0.469 | 0.528 | 0.595 | 0.250 | 0.638 | 0.791 | 0.418 |
| always | 0.233 | 0.500 | 1.000 | 0.233 | 1.000 | n/a | 0.000 |
| never | 0.767 | 0.500 | 0.000 | n/a | 0.000 | 0.767 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=1629, pro act rate 0.258

| policy | act rate | act recall | act prec |
|---|---|---|---|
| learned | 0.579 | 0.675 | 0.301 |
| rules | 0.566 | 0.563 | 0.257 |
| combined | 0.656 | 0.682 | 0.268 |
| always | 1.000 | 1.000 | 0.258 |
| never | 0.000 | 0.000 | n/a |

## test  (n=3221, pro act rate 0.206)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| learned | 0.580 | 0.558 | 0.429 | 0.250 | 0.522 | 0.828 | 0.595 |
| rules | 0.552 | 0.535 | 0.450 | 0.232 | 0.507 | 0.815 | 0.564 |
| combined | 0.506 | 0.545 | 0.540 | 0.233 | 0.611 | 0.826 | 0.478 |
| always | 0.206 | 0.500 | 1.000 | 0.206 | 1.000 | n/a | 0.000 |
| never | 0.794 | 0.500 | 0.000 | n/a | 0.000 | 0.794 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=1745, pro act rate 0.228

| policy | act rate | act recall | act prec |
|---|---|---|---|
| learned | 0.493 | 0.577 | 0.266 |
| rules | 0.513 | 0.554 | 0.246 |
| combined | 0.598 | 0.665 | 0.253 |
| always | 1.000 | 1.000 | 0.228 |
| never | 0.000 | 0.000 | n/a |

