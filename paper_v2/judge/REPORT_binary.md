| judge | set | n | agreement | kappa | 95% bootstrap |
|---|---|---|---|---|---|
| deepseek_v1 | all 50 | 50 | 0.920 | 0.802 | 0.59 to 0.95 |
| deepseek_v1 | non-example 37 | 37 | 0.892 | 0.649 | 0.25 to 0.92 |
| deepseek_v2 | all 50 | 50 | 0.920 | 0.792 | 0.56 to 0.95 |
| deepseek_v2 | non-example 37 | 37 | 0.919 | 0.720 | 0.32 to 1.00 |
| mistral_v1 | all 50 | 50 | 0.940 | 0.840 | 0.63 to 1.00 |
| mistral_v1 | non-example 37 | 37 | 0.946 | 0.801 | 0.44 to 1.00 |

| judge | same-object labels | same-object n | same-object CAAP-TP | different-object labels | different-object n | different-object CAAP-TP | gap (pp) |
|---|---|---|---|---|---|---|---|
| deepseek_v1 | 224 | 3,945 | 82.4% | 551 | 7,533 | 12.5% | 69.8 |
| deepseek_v2 | 189 | 3,622 | 86.3% | 584 | 7,854 | 13.6% | 72.8 |
| mistral_v1 | 185 | 3,585 | 86.8% | 590 | 7,893 | 13.7% | 73.1 |

Original undocumented judge, same collapse: same-object 4,299 / 77.4% (synonym 2,143 at 89.3% + finer-grained 2,156 at 65.6%), different-object 7,179 / 12.0%.