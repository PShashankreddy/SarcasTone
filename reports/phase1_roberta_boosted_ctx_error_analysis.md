# Phase 1 errors: fine-tuned roberta_boosted_ctx

- total samples: **104**
- false positives (predicted sarcasm, actually sincere): **7**
- false negatives (missed sarcasm): **29**

## Top false positives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| His legs flail about as if independent from his body | non-sarc | sarcasm | 0.9934 | CHANDLER |
| I really don't wanna sit with Allen Iverson over there . | non-sarc | sarcasm | 0.9747 | CHANDLER |
| Do you think Penny will come here and take care of us? | non-sarc | sarcasm | 0.8882 | SHELDON |
| I'm sorry, Sheldon bought a loom and learned how to weave? | non-sarc | sarcasm | 0.8369 | PERSON3 |
| Is that why you had to take him to Office Depot last night? | non-sarc | sarcasm | 0.556 | PENNY |
| Anyway, to this day, I still can't see a box of crayons without crossing my legs. | non-sarc | sarcasm | 0.5431 | PENNY |
| Oh yeah he has a caretaker his older brother, Ernie. You can't make this stuff up! | non-sarc | sarcasm | 0.5356 | CHANDLER |

## Top false negatives (most confident mistakes)

| text | true | pred | P(sarcasm) | speaker |
|---|---|---|---|---|
| Almost! But first, we gotta start. | sarcasm | non-sarc | 0.0427 | PERSON |
| But if she dumps you, she'll have a new boyfriend by tomorrow morning and you'll have a new girlfriend when you figure out how to build one. | sarcasm | non-sarc | 0.0784 | HOWARD |
| Darn. If you weren't busy, I'd ask you to join us. | sarcasm | non-sarc | 0.1133 | SHELDON |
| Well, I'm a headhunter. I hook up out of work Soviet scientists with rogue third-world nations. Hi Rasputin! | sarcasm | non-sarc | 0.1153 | CHANDLER |
| Oh, sure. And while we're at it, why don't we put our hands behind our backs, have an old-fashioned eating contest? | sarcasm | non-sarc | 0.1457 | SHELDON |
| I'm listening to you snore. I'm wondering how I'll ever sleep without it. | sarcasm | non-sarc | 0.1497 | SHELDON |
| You're right, the party's fantastic. Please, tell me more. I haven't heard enough about it all week because hearing about that never gets old! | sarcasm | non-sarc | 0.1635 | HOWARD |
| And protected them from a tornado? | sarcasm | non-sarc | 0.1664 | CHANDLER |
| Why? Because she can sing and play guitar and do both at the same time? | sarcasm | non-sarc | 0.1789 | PHOEBE |
| Well, y'know I'm 29. I mean who needs a savings account. | sarcasm | non-sarc | 0.1817 | CHANDLER |
| Okay, here's an idea. What if I change my name and go live with my cousin and her husband, Avi, in Israel? | sarcasm | non-sarc | 0.2239 | HOWARD |
| How's this? "Pleased to meet you, Dr. Gablehauser. "How fortunate for you that the university's "chosen to hire you, despite the fact "that you've done no original research in 25 years, "and instead have written a series of popular books "that reduce the great concepts of science "to a series of anecdotes, "each one dumbed down to accommodate the duration "of an average bowel movement. | sarcasm | non-sarc | 0.2357 | SHELDON |
| Are you still enjoying your nap? | sarcasm | non-sarc | 0.2361 | CHANDLER |
| The blunt instrument that will be the focus of my murder trial? | sarcasm | non-sarc | 0.2624 | LEONARD |
| Obviously, waitressing at The Cheese Cake Factory is a complex socio-economic activity that requires a great deal of analysis and planning. | sarcasm | non-sarc | 0.272 | SHELDON |
| Excellent hole, Joe. | sarcasm | non-sarc | 0.2726 | CHANDLER |
| We were wondering what was taking so long with the gift, but now we understand you were doing this. | sarcasm | non-sarc | 0.2791 | CHANDLER |
| The British are coming? | sarcasm | non-sarc | 0.2922 | CHANDLER |
| You see betrayal in others, but not yourself. | sarcasm | non-sarc | 0.318 | SHELDON |
| Dear god, this parachute is a knapsack | sarcasm | non-sarc | 0.3556 | CHANDLER |

> Fusion hypothesis to check: do FN/FP clusters differ between text-only and
> multimodal models? Compare sample IDs across phase reports.
