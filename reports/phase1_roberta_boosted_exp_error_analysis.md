# Phase 1 errors: fine-tuned roberta_boosted_exp

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **5**
- false negatives (missed sarcasm): **28**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Pheebs? | non-sarc | sarcasm | 0.9781 | CHANDLER |
| So, you're just Bing? | non-sarc | sarcasm | 0.8314 | JOEY |
| Oh, I think we were looking at creepy in the rearview mirror when I put up that camera. | non-sarc | sarcasm | 0.8226 | HOWARD |
| No, but maybe she wants a man with a pocket watch. | non-sarc | sarcasm | 0.7271 | SHELDON |
| His legs flail about as if independent from his body | non-sarc | sarcasm | 0.6505 | CHANDLER |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Okay, here's an idea. What if I change my name and go live with my cousin and her husband, Avi, in Israel? | sarcasm | non-sarc | 0.0171 | HOWARD |
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.0253 | PERSON |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.0407 | CHANDLER |
| Oh, sure. And while we're at it, why don't we put our hands behind our backs, have an old-fashioned eating contest? | sarcasm | non-sarc | 0.076 | SHELDON |
| Nooo! | sarcasm | non-sarc | 0.0772 | MODERATOR |
| I don't think I'll be able to stop thinking about it. | sarcasm | non-sarc | 0.0788 | PENNY |
| I am not freaking out, why would I be freaking out? A woman named Hildy called and said we will get married, but that happens everyday | sarcasm | non-sarc | 0.0836 | CHANDLER |
| The British are coming? | sarcasm | non-sarc | 0.1249 | CHANDLER |
| You see betrayal in others, but not yourself. | sarcasm | non-sarc | 0.1377 | SHELDON |
| Well, if you don't mind looking like an orange traffic cone, great. | sarcasm | non-sarc | 0.1418 | BERNADETTE |
| I guess he wants to do a little dance, you know, make a little love. Well pretty much get down to that. | sarcasm | non-sarc | 0.2103 | CHANDLER |
| Dear god, this parachute is a knapsack | sarcasm | non-sarc | 0.2414 | CHANDLER |
| Obviously, waitressing at The Cheese Cake Factory is a complex socio-economic activity that requires a great deal of analysis and planning. | sarcasm | non-sarc | 0.2516 | SHELDON |
| Yeah, my parents felt that naming me Leonard and putting me in Advanced Placement classes wasn't getting me beaten up enough. | sarcasm | non-sarc | 0.2706 | LEONARD |
| No, I'm just allergic to people who get Nobel Prizes for no good reason. | sarcasm | non-sarc | 0.3127 | SHELDON |
| Exactly what it was half an hour ago. It's like you're not even trying to get better. | sarcasm | non-sarc | 0.3147 | SHELDON |
| Why? Because she can sing and play guitar and do both at the same time? | sarcasm | non-sarc | 0.3199 | PHOEBE |
| Oh, of course he is. She's very interesting. Did you know, when she was 14, she severed the webbing between her own toes? | sarcasm | non-sarc | 0.3224 | SHELDON |
| Well, I'm a headhunter. I hook up out of work Soviet scientists with rogue third-world nations. Hi Rasputin! | sarcasm | non-sarc | 0.3284 | CHANDLER |
| Darn. If you weren't busy, I'd ask you to join us. | sarcasm | non-sarc | 0.3293 | SHELDON |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
