# Act-head evaluation (offline, D-041)

model `db/models/d1-ft-act`, threshold 0.50 (val balanced accuracy 0.573)

## val  (n=3008, pro act rate 0.233)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| learned | 0.554 | 0.573 | 0.496 | 0.286 | 0.608 | 0.818 | 0.538 |
| rules | 0.624 | 0.490 | 0.254 | 0.219 | 0.238 | 0.762 | 0.742 |
| combined | 0.586 | 0.503 | 0.344 | 0.237 | 0.349 | 0.768 | 0.658 |
| always | 0.233 | 0.500 | 1.000 | 0.233 | 1.000 | n/a | 0.000 |
| never | 0.767 | 0.500 | 0.000 | n/a | 0.000 | 0.767 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=1629, pro act rate 0.258

| policy | act rate | act recall | act prec |
|---|---|---|---|
| learned | 0.579 | 0.675 | 0.301 |
| rules | 0.278 | 0.261 | 0.243 |
| combined | 0.347 | 0.335 | 0.250 |
| always | 1.000 | 1.000 | 0.258 |
| never | 0.000 | 0.000 | n/a |

## test  (n=3221, pro act rate 0.206)

| policy | accuracy | balanced acc | act rate | act prec | act recall | wait prec | wait recall |
|---|---|---|---|---|---|---|---|
| learned | 0.580 | 0.558 | 0.429 | 0.250 | 0.522 | 0.828 | 0.595 |
| rules | 0.676 | 0.513 | 0.215 | 0.226 | 0.237 | 0.800 | 0.790 |
| combined | 0.639 | 0.512 | 0.277 | 0.219 | 0.296 | 0.799 | 0.728 |
| always | 0.206 | 0.500 | 1.000 | 0.206 | 1.000 | n/a | 0.000 |
| never | 0.794 | 0.500 | 0.000 | n/a | 0.000 | 0.794 | 1.000 |

reaction slice (since_opp_play <= 3 s): n=1745, pro act rate 0.228

| policy | act rate | act recall | act prec |
|---|---|---|---|
| learned | 0.493 | 0.577 | 0.266 |
| rules | 0.230 | 0.270 | 0.267 |
| combined | 0.273 | 0.322 | 0.268 |
| always | 1.000 | 1.000 | 0.228 |
| never | 0.000 | 0.000 | n/a |

