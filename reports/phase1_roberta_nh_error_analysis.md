# Phase 1 errors: fine-tuned roberta_nh

- total samples: **2862**
- false positives (predicted sarcasm, actually sincere): **69**
- false negatives (missed sarcasm): **161**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| microsoft unveils next best thing to teleportation | non-sarc | sarcasm | 0.9923 | onion/huffpost |
| trump reads fake version of own speech | non-sarc | sarcasm | 0.9876 | onion/huffpost |
| jeb bush severing 'problematic' connections | non-sarc | sarcasm | 0.9706 | onion/huffpost |
| actual soccer team loses game 46-0 | non-sarc | sarcasm | 0.9681 | onion/huffpost |
| 9/11 health program now officially on borrowed time | non-sarc | sarcasm | 0.968 | onion/huffpost |
| report: nsa's intercepted data mostly not from intended targets | non-sarc | sarcasm | 0.9448 | onion/huffpost |
| man travels to historic art locations just to paint the patterns on his shirts | non-sarc | sarcasm | 0.9397 | onion/huffpost |
| taking up arms where birds feast on buffet of salmon | non-sarc | sarcasm | 0.9382 | onion/huffpost |
| man records every detail of his life for 5 years | non-sarc | sarcasm | 0.9355 | onion/huffpost |
| achieving presentation zen | non-sarc | sarcasm | 0.9293 | onion/huffpost |
| scientists crack mystery of tiny traveling plants | non-sarc | sarcasm | 0.9147 | onion/huffpost |
| reply all email creates havoc for case western students' inboxes | non-sarc | sarcasm | 0.9053 | onion/huffpost |
| hilarious moms lament never being in family photos | non-sarc | sarcasm | 0.8984 | onion/huffpost |
| general mills releases tiny toast, its first new cereal in 15 years | non-sarc | sarcasm | 0.8925 | onion/huffpost |
| veterans finding a new outlook outdoors | non-sarc | sarcasm | 0.8904 | onion/huffpost |
| team of sherpas first to scale everest in 2 years | non-sarc | sarcasm | 0.8876 | onion/huffpost |
| astronomers discover most distant galaxy yet | non-sarc | sarcasm | 0.8822 | onion/huffpost |
| family rewrites 'in da club' to celebrate back-to-school season | non-sarc | sarcasm | 0.8733 | onion/huffpost |
| permanent white house staff on edge about the 2016 presidential race | non-sarc | sarcasm | 0.8577 | onion/huffpost |
| pork roll ice cream a hot item at new jersey farm | non-sarc | sarcasm | 0.8444 | onion/huffpost |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| now you can see into your future. and it's pretty darn scary. | sarcasm | non-sarc | 0.0026 | onion/huffpost |
| are we meeting the needs of our nation's rich? | sarcasm | non-sarc | 0.0026 | onion/huffpost |
| those we lost in 2011 | sarcasm | non-sarc | 0.0047 | onion/huffpost |
| historical archives: a puzzle for the mind | sarcasm | non-sarc | 0.005 | onion/huffpost |
| following death of adam yauch, grieving china frees tibet | sarcasm | non-sarc | 0.0106 | onion/huffpost |
| 8-year-old boy surprises marine dad during firefight in afghanistan | sarcasm | non-sarc | 0.0111 | onion/huffpost |
| devin nunes threatens defamation lawsuit after reputation ruined by his official twitter account | sarcasm | non-sarc | 0.0125 | onion/huffpost |
| 340 million social security numbers obtained by federal government in massive personal data breach | sarcasm | non-sarc | 0.0127 | onion/huffpost |
| stolen tour bus leads police on chase of historic downtown philadelphia | sarcasm | non-sarc | 0.0166 | onion/huffpost |
| retirees speak out on crucial lawn care issues | sarcasm | non-sarc | 0.018 | onion/huffpost |
| ann coulter attacks trump for cowardly backing down from full on race war | sarcasm | non-sarc | 0.0215 | onion/huffpost |
| calcutta fire marshal: many indian homes lack bride extinguisher | sarcasm | non-sarc | 0.0244 | onion/huffpost |
| tucker carlson challenges alexandria ocasio-cortez to a date | sarcasm | non-sarc | 0.0271 | onion/huffpost |
| fox news apologizes for mistaking patti labelle for aretha franklin | sarcasm | non-sarc | 0.0278 | onion/huffpost |
| tom gilbert, actor who portrays tv's regis philbin, to leave 'regis & kelly' show | sarcasm | non-sarc | 0.0316 | onion/huffpost |
| jonathan lipnicki to star as young 'dark helmet' in spaceballs prequel | sarcasm | non-sarc | 0.0321 | onion/huffpost |
| nader supporters blame electoral defeat on bush, kerry | sarcasm | non-sarc | 0.0321 | onion/huffpost |
| owls are assholes | sarcasm | non-sarc | 0.0348 | onion/huffpost |
| lee greenwood urges u.s. to take military action against iraq | sarcasm | non-sarc | 0.0353 | onion/huffpost |
| the american dream: what does that part about kissing the gym teacher mean? | sarcasm | non-sarc | 0.0384 | onion/huffpost |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
