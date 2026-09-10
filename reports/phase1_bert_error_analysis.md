# Phase 1 errors: fine-tuned bert

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **19**
- false negatives (missed sarcasm): **19**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Oh, you know, Goth stuff. Goth magazines, Goth music. | non-sarc | sarcasm | 0.7279 | HOWARD |
| No, but maybe she wants a man with a pocket watch. | non-sarc | sarcasm | 0.7271 | SHELDON |
| You don't really believe in that superstition, do you? | non-sarc | sarcasm | 0.717 | SHELDON |
| Is that why you had to take him to Office Depot last night? | non-sarc | sarcasm | 0.7055 | PENNY |
| Do you think Penny will come here and take care of us? | non-sarc | sarcasm | 0.6922 | SHELDON |
| I'm sorry, Sheldon bought a loom and learned how to weave? | non-sarc | sarcasm | 0.6879 | PERSON3 |
| I really don't wanna sit with Allen Iverson over there . | non-sarc | sarcasm | 0.6872 | CHANDLER |
| Anyway, to this day, I still can't see a box of crayons without crossing my legs. | non-sarc | sarcasm | 0.6872 | PENNY |
| No, I'm behind on my Wiki-reading. I'm kind of on a John Grisham kick right now. | non-sarc | sarcasm | 0.6859 | RAJ |
| His legs flail about as if independent from his body | non-sarc | sarcasm | 0.6841 | CHANDLER |
| He's right, even if it's to say something complementary. | non-sarc | sarcasm | 0.6708 | ROSS |
| That woman looks exactly like the pictures of Princess Panchali in the book. How often does one see a beloved fictional character come to life? | non-sarc | sarcasm | 0.6627 | SHELDON |
| Okay, but If he asks, I am not going to lie. | non-sarc | sarcasm | 0.6609 | CHANDLER |
| As you may know, I've been experimenting with elevated anxiety levels, and I thought, what better way to increase my discomfort than to subject myself to an evening of tasteless uncensored crotch talk? | non-sarc | sarcasm | 0.6565 | SHELDON |
| I just have a question. Does Bernadette ever talk about me? | non-sarc | sarcasm | 0.6075 | HOWARD |
| Oh yeah he has a caretaker his older brother, Ernie. You can't make this stuff up! | non-sarc | sarcasm | 0.6069 | CHANDLER |
| No!! You have been screwing us all day! | non-sarc | sarcasm | 0.605 | MONICA |
| Okay, I got a bone to pick with you. | non-sarc | sarcasm | 0.562 | HOWARD |
| So, you're just Bing? | non-sarc | sarcasm | 0.5006 | JOEY |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.2922 | PERSON |
| Oh, sure. And while we're at it, why don't we put our hands behind our backs, have an old-fashioned eating contest? | sarcasm | non-sarc | 0.3055 | SHELDON |
| Excellent hole, Joe. | sarcasm | non-sarc | 0.321 | CHANDLER |
| Well, I'm a headhunter. I hook up out of work Soviet scientists with rogue third-world nations. Hi Rasputin! | sarcasm | non-sarc | 0.3266 | CHANDLER |
| Darn. If you weren't busy, I'd ask you to join us. | sarcasm | non-sarc | 0.3271 | SHELDON |
| Okay, here's an idea. What if I change my name and go live with my cousin and her husband, Avi, in Israel? | sarcasm | non-sarc | 0.3357 | HOWARD |
| Oh Oh, I am convinced! | sarcasm | non-sarc | 0.3399 | CHANDLER |
| You're right, the party's fantastic. Please, tell me more. I haven't heard enough about it all week because hearing about that never gets old! | sarcasm | non-sarc | 0.3551 | HOWARD |
| Yeah, my parents felt that naming me Leonard and putting me in Advanced Placement classes wasn't getting me beaten up enough. | sarcasm | non-sarc | 0.367 | LEONARD |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.3866 | CHANDLER |
| The British are coming? | sarcasm | non-sarc | 0.3932 | CHANDLER |
| Are you still enjoying your nap? | sarcasm | non-sarc | 0.394 | CHANDLER |
| Oh, of course he is. She's very interesting. Did you know, when she was 14, she severed the webbing between her own toes? | sarcasm | non-sarc | 0.4184 | SHELDON |
| I thought if I littered, that crying Indian might come by and save us. | sarcasm | non-sarc | 0.4203 | CHANDLER |
| Nooo! | sarcasm | non-sarc | 0.4271 | MODERATOR |
| The blunt instrument that will be the focus of my murder trial? | sarcasm | non-sarc | 0.446 | LEONARD |
| And protected them from a tornado? | sarcasm | non-sarc | 0.448 | CHANDLER |
| No, you're right, we should do what you do. Have our mom send us pants from the Walmart in Houston. | sarcasm | non-sarc | 0.4521 | PENNY |
| Is it me or the greetings gone downhill around here? | sarcasm | non-sarc | 0.4578 | CHANDLER |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
