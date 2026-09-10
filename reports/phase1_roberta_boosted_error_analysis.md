# Phase 1 errors: fine-tuned roberta_boosted

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **9**
- false negatives (missed sarcasm): **23**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Pheebs? | non-sarc | sarcasm | 0.9086 | CHANDLER |
| His legs flail about as if independent from his body | non-sarc | sarcasm | 0.9056 | CHANDLER |
| Oh, I think we were looking at creepy in the rearview mirror when I put up that camera. | non-sarc | sarcasm | 0.8312 | HOWARD |
| No, but maybe she wants a man with a pocket watch. | non-sarc | sarcasm | 0.7817 | SHELDON |
| I'm sorry, Sheldon bought a loom and learned how to weave? | non-sarc | sarcasm | 0.6462 | PERSON3 |
| That woman looks exactly like the pictures of Princess Panchali in the book. How often does one see a beloved fictional character come to life? | non-sarc | sarcasm | 0.6421 | SHELDON |
| Wow, there's a Denny's in Vegas you can actually get married in. | non-sarc | sarcasm | 0.6357 | PENNY |
| Look now, Phoebe remember, hey, their just fulfilling their Christmas.... | non-sarc | sarcasm | 0.567 | JOEY |
| As you may know, I've been experimenting with elevated anxiety levels, and I thought, what better way to increase my discomfort than to subject myself to an evening of tasteless uncensored crotch talk? | non-sarc | sarcasm | 0.5342 | SHELDON |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Nooo! | sarcasm | non-sarc | 0.1259 | MODERATOR |
| The British are coming? | sarcasm | non-sarc | 0.1265 | CHANDLER |
| Okay, here's an idea. What if I change my name and go live with my cousin and her husband, Avi, in Israel? | sarcasm | non-sarc | 0.1293 | HOWARD |
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.1384 | PERSON |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.2164 | CHANDLER |
| Are you still enjoying your nap? | sarcasm | non-sarc | 0.2442 | CHANDLER |
| I am not freaking out, why would I be freaking out? A woman named Hildy called and said we will get married, but that happens everyday | sarcasm | non-sarc | 0.2599 | CHANDLER |
| Darn. If you weren't busy, I'd ask you to join us. | sarcasm | non-sarc | 0.2619 | SHELDON |
| I guess he wants to do a little dance, you know, make a little love. Well pretty much get down to that. | sarcasm | non-sarc | 0.2646 | CHANDLER |
| And protected them from a tornado? | sarcasm | non-sarc | 0.2647 | CHANDLER |
| I was saying the actual words | sarcasm | non-sarc | 0.2679 | CHANDLER |
| Exactly what it was half an hour ago. It's like you're not even trying to get better. | sarcasm | non-sarc | 0.2752 | SHELDON |
| Excellent hole, Joe. | sarcasm | non-sarc | 0.3043 | CHANDLER |
| You see betrayal in others, but not yourself. | sarcasm | non-sarc | 0.3341 | SHELDON |
| I am sorry, we donÕt have your sheep. | sarcasm | non-sarc | 0.3374 | CHANDLER |
| You're right, the party's fantastic. Please, tell me more. I haven't heard enough about it all week because hearing about that never gets old! | sarcasm | non-sarc | 0.3528 | HOWARD |
| Why? Because she can sing and play guitar and do both at the same time? | sarcasm | non-sarc | 0.37 | PHOEBE |
| We were wondering what was taking so long with the gift, but now we understand you were doing this. | sarcasm | non-sarc | 0.3999 | CHANDLER |
| Obviously, waitressing at The Cheese Cake Factory is a complex socio-economic activity that requires a great deal of analysis and planning. | sarcasm | non-sarc | 0.4182 | SHELDON |
| Oh, sure. And while we're at it, why don't we put our hands behind our backs, have an old-fashioned eating contest? | sarcasm | non-sarc | 0.4698 | SHELDON |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
