# Phase 1 errors: fine-tuned roberta

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **19**
- false negatives (missed sarcasm): **17**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Pheebs? | non-sarc | sarcasm | 0.8835 | CHANDLER |
| Oh, I think we were looking at creepy in the rearview mirror when I put up that camera. | non-sarc | sarcasm | 0.8706 | HOWARD |
| No, but maybe she wants a man with a pocket watch. | non-sarc | sarcasm | 0.8655 | SHELDON |
| Well, he said it was a tribble. It could be a toupee, but either way, it's pretty cool. | non-sarc | sarcasm | 0.8544 | PERSON |
| That woman looks exactly like the pictures of Princess Panchali in the book. How often does one see a beloved fictional character come to life? | non-sarc | sarcasm | 0.8007 | SHELDON |
| As you may know, I've been experimenting with elevated anxiety levels, and I thought, what better way to increase my discomfort than to subject myself to an evening of tasteless uncensored crotch talk? | non-sarc | sarcasm | 0.7778 | SHELDON |
| Oh yeah he has a caretaker his older brother, Ernie. You can't make this stuff up! | non-sarc | sarcasm | 0.7776 | CHANDLER |
| Wow, there's a Denny's in Vegas you can actually get married in. | non-sarc | sarcasm | 0.7668 | PENNY |
| His legs flail about as if independent from his body | non-sarc | sarcasm | 0.7642 | CHANDLER |
| I'm sorry, Sheldon bought a loom and learned how to weave? | non-sarc | sarcasm | 0.7513 | PERSON3 |
| Anyway, to this day, I still can't see a box of crayons without crossing my legs. | non-sarc | sarcasm | 0.6991 | PENNY |
| Look now, Phoebe remember, hey, their just fulfilling their Christmas.... | non-sarc | sarcasm | 0.6882 | JOEY |
| Do you think Penny will come here and take care of us? | non-sarc | sarcasm | 0.6403 | SHELDON |
| Yes, I know, but her friend sounds like such a... | non-sarc | sarcasm | 0.6148 | CHANDLER |
| Terrific, it's just that... I don't think Penny wants me to go. | non-sarc | sarcasm | 0.5843 | LEONARD |
| ...to all the good times we had, like, uh, when we went camping and spent that night telling each other all our secrets. I told him I'm addicted to pedicures and he told me he lost his virginity to his cousin. | non-sarc | sarcasm | 0.5646 | RAJ |
| I don't think so. Hello?  When you get in there | non-sarc | sarcasm | 0.5446 | ROSS |
| All right, so technically it's not a dinner date. I suppose you could call it a, uh, dinfast date. But if you did, you'd open yourself to peer-based mocking, such as, "Hey, Leonard, how was your dinfast with Priya last night?" | non-sarc | sarcasm | 0.5364 | SHELDON |
| Yeah well that's because uh .. I stayed in my room. Yeah, you don't want to look in my hamper. | non-sarc | sarcasm | 0.5327 | JOEY |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Oh no, now its not gonna make any sense | sarcasm | non-sarc | 0.1541 | CHANDLER |
| Nooo! | sarcasm | non-sarc | 0.1955 | MODERATOR |
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.2375 | PERSON |
| I don't think I'll be able to stop thinking about it. | sarcasm | non-sarc | 0.2415 | PENNY |
| Darn. If you weren't busy, I'd ask you to join us. | sarcasm | non-sarc | 0.2438 | SHELDON |
| Exactly what it was half an hour ago. It's like you're not even trying to get better. | sarcasm | non-sarc | 0.2471 | SHELDON |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.2684 | CHANDLER |
| Okay, here's an idea. What if I change my name and go live with my cousin and her husband, Avi, in Israel? | sarcasm | non-sarc | 0.3337 | HOWARD |
| I am not freaking out, why would I be freaking out? A woman named Hildy called and said we will get married, but that happens everyday | sarcasm | non-sarc | 0.347 | CHANDLER |
| Oh Oh, I am convinced! | sarcasm | non-sarc | 0.3763 | CHANDLER |
| I was saying the actual words | sarcasm | non-sarc | 0.3855 | CHANDLER |
| No, I'm just allergic to people who get Nobel Prizes for no good reason. | sarcasm | non-sarc | 0.3925 | SHELDON |
| Are you still enjoying your nap? | sarcasm | non-sarc | 0.3928 | CHANDLER |
| Oh sure, she was probably up all night excited about the party she knows is happening. | sarcasm | non-sarc | 0.4346 | CHANDLER |
| I guess he wants to do a little dance, you know, make a little love. Well pretty much get down to that. | sarcasm | non-sarc | 0.4433 | CHANDLER |
| Why? Because she can sing and play guitar and do both at the same time? | sarcasm | non-sarc | 0.45 | PHOEBE |
| You see betrayal in others, but not yourself. | sarcasm | non-sarc | 0.4561 | SHELDON |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
