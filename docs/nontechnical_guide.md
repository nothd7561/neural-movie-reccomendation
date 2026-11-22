# How I Built and Tested a Movie Recommender

## Why recommendations matter

Imagine opening a streaming app with thousands of movies and half an hour to pick one. A list of popular titles gives you somewhere to start. A personalized list can help you find something that suits your taste without scrolling through the whole catalog.

I wanted to know whether a neural recommendation system could rank movies better than simpler approaches. Building a neural network was only part of the project. I also needed a fair comparison to find out whether it was worth the extra work.

## Learning from movie ratings

MovieLens 1M contains about one million ratings from 6,040 users across 3,706 rated movies. Each record identifies a user and a movie, gives a rating from one to five stars, and records when the rating was made. I treated four and five stars as positive feedback in the ranking experiments.

Here is a fictional example. These are illustrations, not MovieLens records.

| Viewer | Movie | Rating |
|---|---|---:|
| Viewer 17 | The Winter Express | 5 stars |
| Viewer 17 | Saturday at the Lake | 2 stars |
| Viewer 42 | The Winter Express | 4 stars |
| Viewer 42 | The Last Ferry | 5 stars |

The shared interest in *The Winter Express* makes *The Last Ferry* a plausible suggestion for Viewer 17. A few ratings are only clues. Across many viewers and movies, those clues can reveal patterns. The model needs consistent IDs to connect the ratings in each viewer's history. It can do that without names or personal identities.

## Four ways to recommend a movie

### Popularity

Think of a bookstore's bestseller shelf. This model counts how many four- and five-star ratings each movie received in training and starts with the titles with the highest counts. It skips movies the viewer has already rated. Two people with different tastes still start from the same overall ranking. Counting ratings is inexpensive, and the results give the other models a benchmark to beat.

### Matrix Factorization

Suppose viewers who like the same spy movies also tend to like the same crime dramas. Matrix Factorization can pick up that pattern from their ratings. It learns a short numerical profile for each viewer and each movie, then uses how well the profiles fit together to estimate a star rating. A profile can capture something like a taste for suspense, but the model learns the numbers from ratings and gives them no genre labels. Its way of combining the profiles is simpler than the neural models'.

### Neural CF + BCE

Neural Collaborative Filtering, or Neural CF, uses a network to combine viewer and movie profiles more flexibly. One version learns using binary cross-entropy, or BCE. Its training question is simple: given a movie the viewer actually rated, was that rating four stars or higher? The network adjusts its scores when its answer is wrong. It can get better at classifying past ratings while still struggling to put the best unseen options first.

### Neural CF + Dynamic BPR

Bayesian Personalized Ranking, or BPR, changes the training question. If Viewer 17 gave Movie A five stars and has never rated Movie B, the network practices scoring Movie A above Movie B. The viewer's opinion of Movie B is unknown. The unrated title gives the network a comparison to learn from. Repeating this exercise teaches the network to order movies for a particular viewer.

| Approach | Basic idea | Personalization | What it learns | Main tradeoff |
|---|---|---|---|---|
| Popularity | Start with broadly liked titles | Same ranking before history is excluded | Count positive ratings | Fast and strong, with limited room for individual tastes |
| Matrix Factorization | Match viewer and movie profiles | Individual | Predict star ratings | Learns personal patterns with a simple matching rule |
| Neural CF + BCE | Score individual viewer–movie pairs | Individual | Classify past ratings as positive or lower | Classification can improve without improving the top 10 |
| Neural CF + Dynamic BPR | Compare pairs of movies | Individual | Rank liked movies above sampled unrated ones | Learns an ordering from comparisons with uncertain preferences |

## How I prepared the data fairly

I split each viewer's ratings in time order:

**Earlier ratings · about 70% for training → Middle ratings · about 15% for validation → Later ratings · about 15% for testing**

Training ratings teach the model. The middle ratings gave me a way to check progress and choose which saved version of a neural model to keep. The latest ratings were held back for the final comparison. If the model learned from those later ratings first, it would have access to answers it was supposed to predict.

All four models faced the same test conditions. A viewer had to appear in training and have at least one positive test rating. A movie needed at least 50 training ratings to be eligible for recommendation, and movies the viewer had already rated in training or validation were skipped. Agreeing on who and what could be recommended mattered as much as choosing the models.

## How the neural network learned

The network starts with two short lists of numbers, one for the viewer and one for the movie. These learned profiles are called *embeddings*. The network combines them into a single recommendation score. During training, it adjusts both the profiles and the way it combines them so its classifications or rankings improve.

Dynamic BPR gives the network a liked movie and a comparison movie the viewer did not rate in training. At the start of each *epoch*, or pass through training, the sampler redraws the comparisons. This lets the network practice against a wider variety of movies than it would see from one fixed set. A movie can still be drawn again by chance.

## How I measured recommendation quality

Getting a past rating right tells me little about where an interesting movie will appear in a list. I evaluated ten recommendations per viewer against the movies they later rated four or five stars.

Take a fictional viewer with four positive test movies. In a ten-movie list, a checkmark marks a hit and a dash marks a movie with no positive test rating.

- **List A** — ✓ ✓ – – – – – – – –
- **List B** — – – – – – – – – ✓ ✓

Both lists have **Precision@10 of 20%** because two of ten suggestions were hits. Both have **Recall@10 of 50%** because they found two of the four positive test movies. **NDCG@10** gives List A more credit because the hits arrive first. I used it as the main metric because someone browsing the list may only see the first few titles.

## Final results

These are the held-out test scores across the same 5,930 viewers for all four models.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.0637 | 0.0468 | **0.0747** |
| Matrix Factorization | 0.0390 | 0.0291 | 0.0464 |
| Neural CF + BCE | 0.0190 | 0.0134 | 0.0192 |
| Neural CF + Dynamic BPR | 0.0622 | 0.0452 | 0.0720 |

Popularity ranked first. Neural CF + Dynamic BPR came close and scored higher than Matrix Factorization. Its test NDCG@10 was about **275% higher** than Neural CF + BCE, or about 3.75 times the BCE score. That percentage compares two ranking scores. The experiment measured neither a change in customer satisfaction nor a percentage increase in correct picks.

## The most important experiment

Sorting movies into “highly rated” and “rated lower” is a different job from deciding which of ten movies deserves the first slot. Changing the training exercise from classifying single ratings to comparing two movies gave the network a task closer to the one it would face at evaluation. Both neural models used the same architecture, so the stronger result came with a change in how the model learned rather than a bigger network.

During the final comparison run, BPR's training error kept trending down while its validation rankings rose and fell. I kept the epoch 7 version because it had the best ranking score on a fixed group of 200 validation viewers. Its validation NDCG@10 was 0.0832. The 0.0720 in the results table comes from the separate held-out test group.

## What the results leave open

Popularity scored slightly better overall and took much less work to train. The results show how well each model recovered later positive ratings in an offline test. They leave open whether viewers would enjoy the recommended titles or find them useful in a live product. An unrated movie might simply be one the viewer never encountered, and historical ratings capture only part of someone's taste.

New viewers have little history for the personalized models to learn from. New movies may lack learned profiles, and the 50-rating cutoff rules out less frequently rated titles. The results apply to the viewers and movie catalog used in this comparison.

## What I learned

Building these models in PyTorch taught me how to turn viewer and movie IDs into learned profiles and how to train the same network for two different jobs. Preparing time-ordered splits and comparing four strategies on the same users and movies taught me how easily a difference in evaluation rules could change the story.

The result I value most is twofold. Switching to pairwise training made the neural model competitive, and counting positive ratings still ranked first. Both findings helped me understand what the network added and where a simpler approach was enough.

## Explore the project

- [Interactive model results and training charts on Tableau Public](https://public.tableau.com/app/profile/lucas.lu7590/viz/NeuralNetworkModelEvaluation/Dashboard2)
- [Technical overview and reproduction steps](../README.md)
- [Popularity](../src/neural_recommendation/popularity.py), [Matrix Factorization](../src/neural_recommendation/matrix_factorization.py), and [Neural CF](../src/neural_recommendation/neural_cf.py)
- [Evaluation pipeline](../src/neural_recommendation/evaluation.py) and [full-precision test results](../results/metrics/final_model_comparison.csv)
- [Learning journal](../notes/learning_journal.md)
