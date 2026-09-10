# Phase 2 errors: Praat features + LogReg

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **21**
- false negatives (missed sarcasm): **15**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| All right, so technically it's not a dinner date. I suppose you could call it a, uh, dinfast date. But if you did, you'd open yourself to peer-based mocking, such as, "Hey, Leonard, how was your dinfast with Priya last night?" | non-sarc | sarcasm | 0.8554 | SHELDON |
| That woman looks exactly like the pictures of Princess Panchali in the book. How often does one see a beloved fictional character come to life? | non-sarc | sarcasm | 0.8386 | SHELDON |
| Yeah well that's because uh .. I stayed in my room. Yeah, you don't want to look in my hamper. | non-sarc | sarcasm | 0.7933 | JOEY |
| As you may know, I've been experimenting with elevated anxiety levels, and I thought, what better way to increase my discomfort than to subject myself to an evening of tasteless uncensored crotch talk? | non-sarc | sarcasm | 0.7651 | SHELDON |
| Terrific, it's just that... I don't think Penny wants me to go. | non-sarc | sarcasm | 0.7379 | LEONARD |
| I'm sorry, Sheldon bought a loom and learned how to weave? | non-sarc | sarcasm | 0.7167 | PERSON3 |
| You don't really believe in that superstition, do you? | non-sarc | sarcasm | 0.6995 | SHELDON |
| Oh yeah he has a caretaker his older brother, Ernie. You can't make this stuff up! | non-sarc | sarcasm | 0.6988 | CHANDLER |
| Okay, but If he asks, I am not going to lie. | non-sarc | sarcasm | 0.6602 | CHANDLER |
| Okay, I got a bone to pick with you. | non-sarc | sarcasm | 0.66 | HOWARD |
| No, but maybe she wants a man with a pocket watch. | non-sarc | sarcasm | 0.6595 | SHELDON |
| Do you think Penny will come here and take care of us? | non-sarc | sarcasm | 0.6541 | SHELDON |
| Oh, you know, Goth stuff. Goth magazines, Goth music. | non-sarc | sarcasm | 0.6261 | HOWARD |
| Well, he said it was a tribble. It could be a toupee, but either way, it's pretty cool. | non-sarc | sarcasm | 0.621 | PERSON |
| Technically, yes. But, if you'll notice... It's reversible! | non-sarc | sarcasm | 0.6177 | LEONARD |
| His legs flail about as if independent from his body | non-sarc | sarcasm | 0.5758 | CHANDLER |
| What did you mean when you said you were going to miss me? | non-sarc | sarcasm | 0.5649 | LEONARD |
| No, I'm behind on my Wiki-reading. I'm kind of on a John Grisham kick right now. | non-sarc | sarcasm | 0.5572 | RAJ |
| ...to all the good times we had, like, uh, when we went camping and spent that night telling each other all our secrets. I told him I'm addicted to pedicures and he told me he lost his virginity to his cousin. | non-sarc | sarcasm | 0.5561 | RAJ |
| I'm not going to bother him, I'm going to talk to him. | non-sarc | sarcasm | 0.5191 | SHELDON |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| I am sorry, we donÕt have your sheep. | sarcasm | non-sarc | 0.2093 | CHANDLER |
| Exactly what it was half an hour ago. It's like you're not even trying to get better. | sarcasm | non-sarc | 0.2365 | SHELDON |
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.2457 | PERSON |
| Joey if I go first, I want to be looking for my keys | sarcasm | non-sarc | 0.2528 | CHANDLER |
| Why? Because she can sing and play guitar and do both at the same time? | sarcasm | non-sarc | 0.2744 | PHOEBE |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.2754 | CHANDLER |
| Excellent hole, Joe. | sarcasm | non-sarc | 0.3051 | CHANDLER |
| The British are coming? | sarcasm | non-sarc | 0.3053 | CHANDLER |
| And protected them from a tornado? | sarcasm | non-sarc | 0.3289 | CHANDLER |
| Was that place the sun? | sarcasm | non-sarc | 0.3772 | CHANDLER |
| Nooo! | sarcasm | non-sarc | 0.3784 | MODERATOR |
| Is it me or the greetings gone downhill around here? | sarcasm | non-sarc | 0.401 | CHANDLER |
| I was saying the actual words | sarcasm | non-sarc | 0.4437 | CHANDLER |
| Why? You never look down in the shower?  Oh please I am allowed one joke in the monkey is penis genre. | sarcasm | non-sarc | 0.4583 | CHANDLER |
| I'm listening to you snore. I'm wondering how I'll ever sleep without it. | sarcasm | non-sarc | 0.4615 | SHELDON |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
