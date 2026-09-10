# Phase 1 baseline errors: frozen BERT-CLS + LogReg

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **18**
- false negatives (missed sarcasm): **18**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Yeah well that's because uh .. I stayed in my room. Yeah, you don't want to look in my hamper. | non-sarc | sarcasm | 0.9792 | JOEY |
| Oh, I think we were looking at creepy in the rearview mirror when I put up that camera. | non-sarc | sarcasm | 0.9666 | HOWARD |
| No, but maybe she wants a man with a pocket watch. | non-sarc | sarcasm | 0.9345 | SHELDON |
| Wow, there's a Denny's in Vegas you can actually get married in. | non-sarc | sarcasm | 0.9253 | PENNY |
| So, you're just Bing? | non-sarc | sarcasm | 0.9101 | JOEY |
| Oh, you know, Goth stuff. Goth magazines, Goth music. | non-sarc | sarcasm | 0.8808 | HOWARD |
| I'm sorry, Sheldon bought a loom and learned how to weave? | non-sarc | sarcasm | 0.8626 | PERSON3 |
| I know but it's so hard! Nothing rhymes with your stupid name! | non-sarc | sarcasm | 0.8608 | PHOEBE |
| Pheebs? | non-sarc | sarcasm | 0.8607 | CHANDLER |
| Technically, yes. But, if you'll notice... It's reversible! | non-sarc | sarcasm | 0.8446 | LEONARD |
| Look now, Phoebe remember, hey, their just fulfilling their Christmas.... | non-sarc | sarcasm | 0.7871 | JOEY |
| That woman looks exactly like the pictures of Princess Panchali in the book. How often does one see a beloved fictional character come to life? | non-sarc | sarcasm | 0.7812 | SHELDON |
| Anyway, to this day, I still can't see a box of crayons without crossing my legs. | non-sarc | sarcasm | 0.7307 | PENNY |
| Well, he said it was a tribble. It could be a toupee, but either way, it's pretty cool. | non-sarc | sarcasm | 0.6767 | PERSON |
| Okay, I got a bone to pick with you. | non-sarc | sarcasm | 0.5676 | HOWARD |
| Is that why you had to take him to Office Depot last night? | non-sarc | sarcasm | 0.558 | PENNY |
| Yes, I know, but her friend sounds like such a... | non-sarc | sarcasm | 0.5569 | CHANDLER |
| No, I'm behind on my Wiki-reading. I'm kind of on a John Grisham kick right now. | non-sarc | sarcasm | 0.534 | RAJ |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Okay, here's an idea. What if I change my name and go live with my cousin and her husband, Avi, in Israel? | sarcasm | non-sarc | 0.0067 | HOWARD |
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.0214 | PERSON |
| I guess he wants to do a little dance, you know, make a little love. Well pretty much get down to that. | sarcasm | non-sarc | 0.0481 | CHANDLER |
| The British are coming? | sarcasm | non-sarc | 0.1364 | CHANDLER |
| Nooo! | sarcasm | non-sarc | 0.1666 | MODERATOR |
| is an event that's like a heart attack. | sarcasm | non-sarc | 0.1789 | SHELDON |
| I was saying the actual words | sarcasm | non-sarc | 0.1858 | CHANDLER |
| I am not freaking out, why would I be freaking out? A woman named Hildy called and said we will get married, but that happens everyday | sarcasm | non-sarc | 0.2098 | CHANDLER |
| Yes, we all know how cruel a parent can be, about the flatness of a child's pillow. | sarcasm | non-sarc | 0.2966 | CHANDLER |
| Joey if I go first, I want to be looking for my keys | sarcasm | non-sarc | 0.306 | CHANDLER |
| Yeah, my parents felt that naming me Leonard and putting me in Advanced Placement classes wasn't getting me beaten up enough. | sarcasm | non-sarc | 0.3586 | LEONARD |
| I don't think I'll be able to stop thinking about it. | sarcasm | non-sarc | 0.3839 | PENNY |
| We were wondering what was taking so long with the gift, but now we understand you were doing this. | sarcasm | non-sarc | 0.3893 | CHANDLER |
| I'm listening to you snore. I'm wondering how I'll ever sleep without it. | sarcasm | non-sarc | 0.3945 | SHELDON |
| Oh no, now its not gonna make any sense | sarcasm | non-sarc | 0.4086 | CHANDLER |
| You're right, the party's fantastic. Please, tell me more. I haven't heard enough about it all week because hearing about that never gets old! | sarcasm | non-sarc | 0.4155 | HOWARD |
| Darn. If you weren't busy, I'd ask you to join us. | sarcasm | non-sarc | 0.4656 | SHELDON |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.4922 | CHANDLER |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
