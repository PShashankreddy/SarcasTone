# Phase 1 Significance Tests (locked test set, n=104)

Exact McNemar on discordant pairs + paired bootstrap (2000 iters, resampled macro-F1).

| A | B | n01 (A better) | n10 (B better) | McNemar p | dF1 (A-B) | bootstrap 95% CI | bootstrap p |
|---|---|---|---|---|---|---|---|
| lrtfidf | lrbertcls | 16 | 22 | 0.4177 | -0.0583 | [-0.1729, +0.0577] | 0.8336 |
| lrtfidf | bert_ctx | 24 | 28 | 0.6778 | -0.0391 | [-0.1746, +0.0950] | 0.7076 |
| lrtfidf | roberta | 15 | 21 | 0.405 | -0.0582 | [-0.1692, +0.0595] | 0.8531 |
| lrtfidf | roberta_boosted | 16 | 26 | 0.1641 | -0.0911 | [-0.2109, +0.0327] | 0.9195 |
| lrbertcls | bert_ctx | 21 | 19 | 0.8746 | +0.0192 | [-0.1041, +0.1376] | 0.3853 |
| lrbertcls | roberta | 15 | 15 | 1.0 | +0.0001 | [-0.1077, +0.1080] | 0.5047 |
| lrbertcls | roberta_boosted | 14 | 18 | 0.5966 | -0.0328 | [-0.1411, +0.0762] | 0.7086 |
| bert_ctx | roberta | 21 | 23 | 0.8804 | -0.0191 | [-0.1435, +0.1061] | 0.6227 |
| bert_ctx | roberta_boosted | 14 | 20 | 0.3915 | -0.0520 | [-0.1596, +0.0588] | 0.8121 |
| roberta | roberta_boosted | 10 | 14 | 0.5413 | -0.0329 | [-0.1270, +0.0581] | 0.7546 |

Model key: lrtfidf=TF-IDF+LogReg, lrbertcls=frozen BERT-CLS+LR, bert_ctx=BERT(+context), roberta=RoBERTa, roberta_boosted=champion.

_Note: n=104. The champion's raw F1 gains over TF-IDF and frozen BERT are the largest
effects, but no pair reaches p<0.05 - every difference sits inside the n=104 noise band.
McNemar p is two-sided exact; bootstrap p is the fraction of resamples where A <= B._