# Phase 1 baseline errors: TF-IDF + LogReg

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **19**
- false negatives (missed sarcasm): **23**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Yes, I know, but her friend sounds like such a... | non-sarc | sarcasm | 0.7874 | CHANDLER |
| Oh, I think we were looking at creepy in the rearview mirror when I put up that camera. | non-sarc | sarcasm | 0.7865 | HOWARD |
| Technically, yes. But, if you'll notice... It's reversible! | non-sarc | sarcasm | 0.7724 | LEONARD |
| His legs flail about as if independent from his body | non-sarc | sarcasm | 0.7533 | CHANDLER |
| That woman looks exactly like the pictures of Princess Panchali in the book. How often does one see a beloved fictional character come to life? | non-sarc | sarcasm | 0.7059 | SHELDON |
| Do you think Penny will come here and take care of us? | non-sarc | sarcasm | 0.6316 | SHELDON |
| I don't think so. Hello?  When you get in there | non-sarc | sarcasm | 0.6225 | ROSS |
| Yeah it would! | non-sarc | sarcasm | 0.6118 | ROSS |
| No, I'm behind on my Wiki-reading. I'm kind of on a John Grisham kick right now. | non-sarc | sarcasm | 0.5795 | RAJ |
| I'm the supply manager. | non-sarc | sarcasm | 0.5623 | PERSON |
| Ok! | non-sarc | sarcasm | 0.5524 | JOEY |
| He's right, even if it's to say something complementary. | non-sarc | sarcasm | 0.5409 | ROSS |
| Well, he said it was a tribble. It could be a toupee, but either way, it's pretty cool. | non-sarc | sarcasm | 0.5404 | PERSON |
| Wow, there's a Denny's in Vegas you can actually get married in. | non-sarc | sarcasm | 0.531 | PENNY |
| I know but it's so hard! Nothing rhymes with your stupid name! | non-sarc | sarcasm | 0.5308 | PHOEBE |
| ...to all the good times we had, like, uh, when we went camping and spent that night telling each other all our secrets. I told him I'm addicted to pedicures and he told me he lost his virginity to his cousin. | non-sarc | sarcasm | 0.5294 | RAJ |
| Pheebs? | non-sarc | sarcasm | 0.5223 | CHANDLER |
| All right, so technically it's not a dinner date. I suppose you could call it a, uh, dinfast date. But if you did, you'd open yourself to peer-based mocking, such as, "Hey, Leonard, how was your dinfast with Priya last night?" | non-sarc | sarcasm | 0.5154 | SHELDON |
| I'm not going to bother him, I'm going to talk to him. | non-sarc | sarcasm | 0.5138 | SHELDON |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| No, this is just part of a daredevil game that I play called wait until last moment before I burst and die | sarcasm | non-sarc | 0.1964 | CHANDLER |
| Why? Because she can sing and play guitar and do both at the same time? | sarcasm | non-sarc | 0.203 | PHOEBE |
| Yeah, my parents felt that naming me Leonard and putting me in Advanced Placement classes wasn't getting me beaten up enough. | sarcasm | non-sarc | 0.2039 | LEONARD |
| Excellent hole, Joe. | sarcasm | non-sarc | 0.2389 | CHANDLER |
| I guess he wants to do a little dance, you know, make a little love. Well pretty much get down to that. | sarcasm | non-sarc | 0.2751 | CHANDLER |
| No, I'm just allergic to people who get Nobel Prizes for no good reason. | sarcasm | non-sarc | 0.3184 | SHELDON |
| I'm just inferring this is a couch because the evidence suggests the coffee table is having a tiny garage sale. | sarcasm | non-sarc | 0.3491 | SHELDON |
| No, you're right, we should do what you do. Have our mom send us pants from the Walmart in Houston. | sarcasm | non-sarc | 0.3519 | PENNY |
| Dear god, this parachute is a knapsack | sarcasm | non-sarc | 0.3623 | CHANDLER |
| I thought if I littered, that crying Indian might come by and save us. | sarcasm | non-sarc | 0.3641 | CHANDLER |
| Okay, here's an idea. What if I change my name and go live with my cousin and her husband, Avi, in Israel? | sarcasm | non-sarc | 0.3699 | HOWARD |
| Nooo! | sarcasm | non-sarc | 0.3841 | MODERATOR |
| Was that place the sun? | sarcasm | non-sarc | 0.3844 | CHANDLER |
| Oh no, now its not gonna make any sense | sarcasm | non-sarc | 0.3857 | CHANDLER |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.389 | CHANDLER |
| You see betrayal in others, but not yourself. | sarcasm | non-sarc | 0.4189 | SHELDON |
| Joey if I go first, I want to be looking for my keys | sarcasm | non-sarc | 0.4258 | CHANDLER |
| Obviously, waitressing at The Cheese Cake Factory is a complex socio-economic activity that requires a great deal of analysis and planning. | sarcasm | non-sarc | 0.4441 | SHELDON |
| I'm listening to you snore. I'm wondering how I'll ever sleep without it. | sarcasm | non-sarc | 0.4442 | SHELDON |
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.4638 | PERSON |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
