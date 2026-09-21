# Phase 2 Significance Tests (locked test set, n=104)

Speech models: lr=Praat summary+LogReg, cnn=1D-CNN on MFCC, bigru=BiGRU on MFCC.
Exact McNemar (discordant pairs) + paired bootstrap (2000 iters, macro-F1).

| A | B | n01 (A better) | n10 (B better) | McNemar p | dF1 (A-B) | bootstrap 95% CI | bootstrap p |
|---|---|---|---|---|---|---|---|
| lr_praat | cnn | 13 | 20 | 0.2962 | -0.0653 | [-0.1746, +0.0430] | 0.8896 |
| lr_praat | bigru | 24 | 19 | 0.5424 | +0.0553 | [-0.0693, +0.1830] | 0.1869 |
| cnn | bigru | 31 | 19 | 0.1189 | +0.1206 | [-0.0087, +0.2524] | 0.0365 |

_Note: n=104. CNN > LR is the largest effect; significance is limited by test size_