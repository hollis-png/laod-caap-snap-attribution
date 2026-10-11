# Judge report

## Run health

| run | model returned | labels | parse failures | unsure | max attempts |
|---|---|---|---|---|---|
| deepseek_v1 | {'deepseek/deepseek-v4-flash-0731': 775} | 775 | 0 | 0 | 1 |
| deepseek_v2 | {'deepseek/deepseek-v4-flash-0731': 775} | 775 | 0 | 2 | 1 |
| mistral_v1 | {'mistralai/mistral-large-4-0': 775} | 775 | 0 | 0 | 1 |

## Judge vs human (B1, 50 labels; 37 not in prompt)

| run | set | n | accuracy | Cohen kappa | kappa 95% bootstrap |
|---|---|---|---|---|---|
| deepseek_v1 | all 50 | 50 | 0.840 | 0.640 | 0.44 to 0.83 |
| deepseek_v1 | non-example 37 | 37 | 0.838 | 0.507 | 0.18 to 0.79 |
| deepseek_v2 | all 50 | 50 | 0.860 | 0.666 | 0.45 to 0.86 |
| deepseek_v2 | non-example 37 | 37 | 0.892 | 0.648 | 0.28 to 0.91 |
| mistral_v1 | all 50 | 50 | 0.920 | 0.803 | 0.58 to 0.95 |
| mistral_v1 | non-example 37 | 37 | 0.946 | 0.810 | 0.44 to 1.00 |

Confusion, deepseek_v1 (rows judge, cols human), non-example set:

| judge \ human | synonym | finer_grained | out_of_ontology |
|---|---|---|---|
| synonym | 2 | 2 | 2 |
| finer_grained | 0 | 1 | 1 |
| out_of_ontology | 0 | 1 | 28 |

Disagreements (deepseek_v1, all 50): surfer->person judge=finer_grained human=synonym; badminton racket->tennis racket judge=out_of_ontology human=finer_grained; stove->oven judge=synonym human=out_of_ontology; face->person judge=finer_grained human=out_of_ontology; pigeons->bird judge=synonym human=finer_grained; man->person judge=synonym human=finer_grained; monitor->tv judge=synonym human=out_of_ontology; table->dining table judge=synonym human=finer_grained

## Judge vs judge (all labels both classified)

| pair | n | agreement | kappa |
|---|---|---|---|
| deepseek_v1 vs deepseek_v2 | 775 | 0.907 | 0.783 |
| deepseek_v1 vs mistral_v1 | 775 | 0.854 | 0.660 |
| deepseek_v2 vs mistral_v1 | 775 | 0.898 | 0.744 |

## Section 3.3 table recomputed (weights n_with_gt_match; rate = TP / n)

| judge | category | labels | n matched detections | CAAP-TP rate |
|---|---|---|---|---|
| deepseek_v1 | synonym | 139 | 2,788 | 85.0% |
| deepseek_v1 | finer_grained | 85 | 1,157 | 76.1% |
| deepseek_v1 | out_of_ontology | 551 | 7,533 | 12.5% |
| deepseek_v2 | synonym | 96 | 1,776 | 79.9% |
| deepseek_v2 | finer_grained | 93 | 1,846 | 92.5% |
| deepseek_v2 | out_of_ontology | 584 | 7,854 | 13.6% |
| deepseek_v2 | unsure | 2 | 2 | 0.0% |
| mistral_v1 | synonym | 48 | 1,018 | 88.6% |
| mistral_v1 | finer_grained | 137 | 2,567 | 86.1% |
| mistral_v1 | out_of_ontology | 590 | 7,893 | 13.7% |

Original (undocumented) judge: synonym 2,143 / 89.3%, finer_grained 2,156 / 65.6%, out_of_ontology 7,179 / 12.0%.

## Re-check of the three caveats (deepseek_v1)

- clothing: suit: out_of_ontology (TP 91%, n=33); dress: out_of_ontology (TP 61%, n=28); jacket: out_of_ontology (TP 50%, n=60); shirt: out_of_ontology (TP 44%, n=135); t-shirt: out_of_ontology (TP 25%, n=12); coat: out_of_ontology (TP 86%, n=7); pants: out_of_ontology (TP 4%, n=45); jeans: out_of_ontology (TP 15%, n=41)
- person descriptors: man: synonym (TP 98%, n=782); woman: finer_grained (TP 97%, n=282); boy: finer_grained (TP 98%, n=121); girl: finer_grained (TP 95%, n=19); men: synonym (TP 100%, n=19); women: finer_grained (TP 95%, n=20); child: finer_grained (TP 97%, n=33); surfer: finer_grained (TP 100%, n=34)
- part-whole: faucet: out_of_ontology (TP 1%, n=80); shelves: out_of_ontology (TP 12%, n=16); cushions: out_of_ontology (TP 18%, n=17); cushion: out_of_ontology (TP 10%, n=42); pillow: out_of_ontology (TP 6%, n=150); pillows: out_of_ontology (TP 1%, n=85)
- finer_grained bimodality: of 1,157 matched detections, 59% are in labels with TP >= 0.9 and 8% in labels with TP < 0.25

## Decision rule

kappa(deepseek_v1, human; non-example) = 0.507; kappa(deepseek_v1, deepseek_v2) = 0.783. **DOWNGRADE section 3.3 to a descriptive statement.**