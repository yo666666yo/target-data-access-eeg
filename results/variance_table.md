# Per-run variance table

All quantities are computed inside a single run and, where the paper
quotes one number, pooled across runs on the **variance** scale.
Table I of the paper instead reports the standard deviation of
run-averaged per-subject accuracy, which is a different pooling order
and a slightly smaller number; the two are not interchangeable.

`noise` is the conditionally binomial estimate
`N^-1 sum_s p_s(1-p_s)/(m_s-1)`; it cannot exceed `bound`,
`max_s 0.25/(m_s-1)`. `benchmark_var` is the random-regrouping null,
which removes the subject structure as well as the trial noise and so
is not the same estimand.

| dataset | cond | run | N | m | raw_var | noise | bound | corrected_var | benchmark_var |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BCI IV 2a | SRC | 0 | 9 | 288 | 0.02898341 | 0.00077711 | 0.00087108 | 0.02820630 | 0.00086916 |
| BCI IV 2a | SRC | 1 | 9 | 288 | 0.02946801 | 0.00076876 | 0.00087108 | 0.02869925 | 0.00086247 |
| BCI IV 2a | SRC | 2 | 9 | 288 | 0.03508023 | 0.00075061 | 0.00087108 | 0.03432962 | 0.00087202 |
| BCI IV 2a | SUP | 0 | 9 | 288 | 0.03822291 | 0.00074209 | 0.00087108 | 0.03748082 | 0.00086905 |
| BCI IV 2a | SUP | 1 | 9 | 288 | 0.03746939 | 0.00075092 | 0.00087108 | 0.03671847 | 0.00086537 |
| BCI IV 2a | SUP | 2 | 9 | 288 | 0.04260539 | 0.00073215 | 0.00087108 | 0.04187324 | 0.00085947 |
| BNCI2014-002 | SRC | 0 | 14 | 80 | 0.01811556 | 0.00252755 | 0.00316456 | 0.01558801 | 0.00268154 |
| BNCI2014-002 | SRC | 1 | 14 | 80 | 0.01315333 | 0.00248997 | 0.00316456 | 0.01066336 | 0.00260283 |
| BNCI2014-002 | SRC | 2 | 14 | 80 | 0.01311212 | 0.00261881 | 0.00316456 | 0.01049331 | 0.00273713 |
| BNCI2014-002 | SUP | 0 | 14 | 80 | 0.01999056 | 0.00218849 | 0.00316456 | 0.01780207 | 0.00240783 |
| BNCI2014-002 | SUP | 1 | 14 | 80 | 0.02022665 | 0.00219117 | 0.00316456 | 0.01803548 | 0.00239156 |
| BNCI2014-002 | SUP | 2 | 14 | 80 | 0.01859289 | 0.00199494 | 0.00316456 | 0.01659795 | 0.00221169 |
| PhysionetMI | SRC | 0 | 105 | 44-45 | 0.01805489 | 0.00517785 | 0.00581395 | 0.01287704 | 0.00542043 |
| PhysionetMI | SRC | 1 | 105 | 44-45 | 0.01991564 | 0.00520062 | 0.00581395 | 0.01471502 | 0.00554046 |
| PhysionetMI | SRC | 2 | 105 | 44-45 | 0.01565375 | 0.00523745 | 0.00581395 | 0.01041630 | 0.00548407 |
| PhysionetMI | SUP | 0 | 105 | 44-45 | 0.02911771 | 0.00502507 | 0.00581395 | 0.02409265 | 0.00557182 |
| PhysionetMI | SUP | 1 | 105 | 44-45 | 0.02470760 | 0.00512680 | 0.00581395 | 0.01958080 | 0.00555892 |
| PhysionetMI | SUP | 2 | 105 | 44-45 | 0.02889187 | 0.00503073 | 0.00581395 | 0.02386114 | 0.00555232 |

## Pooled over runs (variance scale)

| dataset | cond | raw SD | corrected SD | variance share |
|---|---|---:|---:|---:|
| BCI IV 2a | SRC | 0.176571 | 0.174390 | 2.46% |
| BCI IV 2a | SUP | 0.198576 | 0.196700 | 1.88% |
| BNCI2014-002 | SRC | 0.121629 | 0.110672 | 17.21% |
| BNCI2014-002 | SUP | 0.140012 | 0.132206 | 10.84% |
| PhysionetMI | SRC | 0.133697 | 0.112559 | 29.12% |
| PhysionetMI | SUP | 0.166049 | 0.150038 | 18.35% |
