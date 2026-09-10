# Phase 2 errors: speech CNN

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **9**
- false negatives (missed sarcasm): **20**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Okay, but If he asks, I am not going to lie. | non-sarc | sarcasm | 0.99 | CHANDLER |
| Oh yeah he has a caretaker his older brother, Ernie. You can't make this stuff up! | non-sarc | sarcasm | 0.9475 | CHANDLER |
| No, I'm behind on my Wiki-reading. I'm kind of on a John Grisham kick right now. | non-sarc | sarcasm | 0.8906 | RAJ |
| I'm not going to bother him, I'm going to talk to him. | non-sarc | sarcasm | 0.7808 | SHELDON |
| I really don't wanna sit with Allen Iverson over there . | non-sarc | sarcasm | 0.7563 | CHANDLER |
| What?!?! | non-sarc | sarcasm | 0.7499 | MONICA |
| Oh, you know, Goth stuff. Goth magazines, Goth music. | non-sarc | sarcasm | 0.7317 | HOWARD |
| I'm sorry, Sheldon bought a loom and learned how to weave? | non-sarc | sarcasm | 0.6767 | PERSON3 |
| Cover yourself up! | non-sarc | sarcasm | 0.5515 | MONICA |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Exactly what it was half an hour ago. It's like you're not even trying to get better. | sarcasm | non-sarc | 0.0189 | SHELDON |
| Why? Because she can sing and play guitar and do both at the same time? | sarcasm | non-sarc | 0.063 | PHOEBE |
| And protected them from a tornado? | sarcasm | non-sarc | 0.0638 | CHANDLER |
| Was that place the sun? | sarcasm | non-sarc | 0.0877 | CHANDLER |
| Oh, of course he is. She's very interesting. Did you know, when she was 14, she severed the webbing between her own toes? | sarcasm | non-sarc | 0.1045 | SHELDON |
| Joey if I go first, I want to be looking for my keys | sarcasm | non-sarc | 0.1132 | CHANDLER |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.1255 | CHANDLER |
| I'm listening to you snore. I'm wondering how I'll ever sleep without it. | sarcasm | non-sarc | 0.1287 | SHELDON |
| I am not freaking out, why would I be freaking out? A woman named Hildy called and said we will get married, but that happens everyday | sarcasm | non-sarc | 0.2165 | CHANDLER |
| Well, I'm a headhunter. I hook up out of work Soviet scientists with rogue third-world nations. Hi Rasputin! | sarcasm | non-sarc | 0.2527 | CHANDLER |
| Oh Oh, I am convinced! | sarcasm | non-sarc | 0.2621 | CHANDLER |
| No, I'm just allergic to people who get Nobel Prizes for no good reason. | sarcasm | non-sarc | 0.272 | SHELDON |
| The blunt instrument that will be the focus of my murder trial? | sarcasm | non-sarc | 0.2805 | LEONARD |
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.2973 | PERSON |
| Nooo! | sarcasm | non-sarc | 0.3053 | MODERATOR |
| Dear god, this parachute is a knapsack | sarcasm | non-sarc | 0.348 | CHANDLER |
| is an event that's like a heart attack. | sarcasm | non-sarc | 0.4003 | SHELDON |
| But if she dumps you, she'll have a new boyfriend by tomorrow morning and you'll have a new girlfriend when you figure out how to build one. | sarcasm | non-sarc | 0.4163 | HOWARD |
| No, you're right, we should do what you do. Have our mom send us pants from the Walmart in Houston. | sarcasm | non-sarc | 0.4652 | PENNY |
| Excellent hole, Joe. | sarcasm | non-sarc | 0.4925 | CHANDLER |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
