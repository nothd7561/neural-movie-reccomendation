# Neural Movie Recommendation System

A movie platform can show everyone the same popular titles or use each person's rating history to suggest films they might enjoy. This project uses MovieLens 1M to ask whether a neural recommender ranks those films better than simpler approaches.

I built four recommenders, tested different ways to train the neural model, and compared the models on the same held-out users and candidate movies. Pairwise ranking made the neural model competitive, while popularity still led the final comparison.

| Dataset | Models compared | Primary metric | Best neural model | Best overall model |
|---|---:|---|---|---|
| MovieLens 1M | 4 | NDCG@10 | Neural CF + Dynamic BPR (0.0720) | Popularity (0.0747) |

[Final test results](#key-results) | [Nontechnical project guide](docs/nontechnical_guide.md) | [Model implementations](#modeling-approaches) | [Reproduce the experiment](#how-to-run)

## Key results

**Held-out test results** across 5,930 users. Every model recommended 10 unique movies from a shared catalog of titles with at least 50 training ratings. [Download the full-precision test results](results/metrics/final_model_comparison.csv).

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity | **0.0637** | **0.0468** | **0.0747** |
| Matrix Factorization | 0.0390 | 0.0291 | 0.0464 |
| Neural CF + BCE | 0.0190 | 0.0134 | 0.0192 |
| Neural CF + Dynamic BPR | 0.0622 | 0.0452 | 0.0720 |

Switching the neural training objective from binary cross-entropy (BCE) to Bayesian Personalized Ranking (BPR) corresponded to about **275% higher test NDCG@10**. Dynamic BPR also scored about 55% higher than Matrix Factorization on NDCG@10. Popularity finished about 0.0027 ahead of Dynamic BPR, a small absolute advantage for a model that requires only counting ratings. These offline scores measure how well the models recover highly rated held-out movies and place them near the top. User satisfaction would require separate evidence.

## Contents

- [Key results](#key-results)
- [How recommendations work](#how-recommendations-work)
- [Dataset and data processing](#dataset-and-data-processing)
- [Modeling approaches](#modeling-approaches)
- [Experiments and model selection](#experiments-and-model-selection)
- [Training history and visualizations](#training-history-and-visualizations)
- [Evaluation methodology](#evaluation-methodology)
- [Technologies used](#technologies-used)
- [Repository navigation](#repository-navigation)
- [How to run](#how-to-run)
- [Limitations and future work](#limitations-and-future-work)
- [Key takeaways](#key-takeaways)

## How recommendations work

Imagine two people opening a movie app. One regularly gives high ratings to thrillers and the other prefers comedies. A personalized recommender looks for patterns in each person's ratings and ranks unseen movies accordingly.

- **Popularity** starts with the movies that received the most high ratings across all users.
- **Matrix Factorization** learns short lists of numbers that represent users and movies. Their fit helps predict how someone might rate an unseen movie.
- **Neural CF + BCE** uses a neural network to combine those representations and score how likely a rating is to be positive.
- **Neural CF + Dynamic BPR** uses the same kind of network and learns which of two movies should rank higher. It practices on a movie the user rated highly and a sampled movie the user has never rated. That second movie is **unobserved** and serves as a training comparison even though the user's preference is unknown.

**Precision@10** asks how many of the ten suggestions were relevant. **Recall@10** asks how many of the user's relevant movies appeared in those suggestions. **NDCG@10** gives more credit when relevant movies appear near the top, where a user is most likely to see them.

## Dataset and data processing

MovieLens 1M contains **1,000,209 ratings from 6,040 users across 3,706 rated movies**. The models use user IDs, movie IDs, ratings from 1 to 5, and timestamps for splitting. Ratings of 4 or 5 count as positive feedback. [GroupLens MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/)

The loader checks for missing values, duplicate user–movie pairs, and ratings outside 1–5. It converts Unix timestamps to UTC. The splitter orders each user's ratings by timestamp, using movie ID to break ties, and assigns approximately 70% to training, 15% to validation, and 15% to test. Keeping later interactions out of training better reflects how a recommender uses earlier activity to predict future preferences. [Data loader](src/neural_recommendation/data_loader.py) · [Chronological splitter](src/neural_recommendation/preprocessing.py)

## Modeling approaches

The trainable models use Adam with a learning rate of 0.001, a batch size of 256, and 10 epochs in the final comparison.

### Popularity

Counts positive training ratings for each movie, ranks movies by that count, and skips titles the user already rated. A low-cost baseline like this sets a meaningful benchmark for the personalized models. [Implementation](src/neural_recommendation/popularity.py)

### Matrix Factorization

Predicts star ratings from the dot product of 16-dimensional user and movie embeddings, plus learned user and movie biases and the global training mean. It minimizes mean squared error, reports validation MSE, and uses the final epoch for test ranking. [Implementation](src/neural_recommendation/matrix_factorization.py)

### Neural CF + BCE

Concatenates 32-dimensional user and movie embeddings and passes them through dense layers of sizes 64, 32, and 1, with ReLU after the first two layers. BCE with logits trains the model to distinguish high ratings from lower ratings among observed interactions. Test ranking uses the raw network score. The trainer restores the weights with the lowest validation BCE loss. [Implementation](src/neural_recommendation/neural_cf.py#L224)

### Neural CF + Dynamic BPR

Uses the same network architecture and learns pairwise preferences. The loss penalizes the model when a positively rated movie fails to score above an unobserved comparison. For each positive interaction, the sampler draws from movies present in training that the user has never rated in training. Redrawing comparisons each epoch exposes the model to more pairs. The model outputs ranking scores, and the training objective provides no probability calibration. [Sampler and trainer](src/neural_recommendation/neural_cf.py#L149)

## Experiments and model selection

BCE learns to classify ratings on movies users already chose to rate. BPR learns a relative ordering, which more closely matches the task of ranking unseen movies. I also compared **fixed sampling**, which reuses the same pairs, with **dynamic sampling**, which draws new comparisons each epoch. Both preparation paths are in the code. Dynamic BPR is the variant used in the final four-model comparison.

The BPR trainer uses a seeded sample of 200 eligible validation users, keeps that cohort fixed, and restores the weights from the epoch with the highest validation NDCG@10. Training loss can keep falling after validation ranking peaks, so checkpoint selection uses the ranking metric. The final comparison run selected epoch 7 at 0.0832 NDCG@10 on the validation subset. That checkpoint scored 0.0720 NDCG@10 on the held-out test cohort. The learning journal records earlier validation experiments with embedding sizes, rating cutoffs, and the fixed-negative BPR run. [Learning journal](notes/learning_journal.md)

## Training history and visualizations

An earlier dynamic BPR run is recorded in the local `results/metrics/epoch_results.csv`. Training loss declined from 0.3909 to 0.2599, and its validation-subset NDCG@10 peaked at 0.0844 in epoch 9. The available records establish separate validation histories. The cause of the different checkpoint choices is undocumented.

The [exploratory analysis script](notebooks/eda.py) contains chart code for ratings and interaction counts by user and movie. The final comparison table and [results CSV](results/metrics/final_model_comparison.csv) provide the performance evidence for this project.

## Evaluation methodology

1. Fit models on the training split. Use validation for BCE and BPR checkpoint selection, and record validation MSE for Matrix Factorization. Open the test file after training and checkpoint selection.
2. Evaluate the same users for all models. Each user must appear in training and have at least one positive test rating.
3. Use the same candidate rules for all models. Each movie needs at least 50 ratings in training, and each user's train and validation histories are excluded from their recommendations. Rank the remaining candidates using the model's scores.
4. Require exactly 10 unique recommendations per user. Compare them with all of that user's positive test movies, including relevant movies outside the candidate catalog. Compute Precision@10, Recall@10, and NDCG@10 per user, then average across users.

NDCG@10 gives more weight to early hits and normalizes by the user's ideal ranking. The common catalog and cohort keep differences in user or movie eligibility from driving the model comparison. The 50-rating cutoff standardizes this test run and differs from the historical Popularity and Matrix Factorization validation setups. [Evaluation implementation](src/neural_recommendation/evaluation.py)

## Technologies used

| Technology | Role |
|---|---|
| Python | Data preparation, model training, and experiment entry points |
| pandas and NumPy | Rating tables, grouping, CSV exports, and seeded negative sampling |
| PyTorch | Embeddings, neural layers, losses, data loaders, and optimization |
| Matplotlib and Seaborn | Exploratory plotting code |

## Repository navigation

```text
.
|-- data/
|   |-- raw/                    # Local MovieLens files
|   `-- processed/              # Local cleaned ratings and splits
|-- notebooks/
|   `-- eda.py                  # Exploratory rating and activity distributions
|-- notes/
|   `-- learning_journal.md     # Experiment notes and implementation decisions
|-- results/
|   |-- checkpoints/
|   |-- figures/
|   `-- metrics/               # Final test snapshot and local experiment logs
|-- src/neural_recommendation/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- popularity.py
|   |-- matrix_factorization.py
|   |-- neural_cf.py
|   |-- popularity_diagnostic.py
|   `-- evaluation.py
|-- requirements.txt
`-- README.md
```

| To explore | Start here |
|---|---|
| Project story and results in plain language | [Nontechnical guide](docs/nontechnical_guide.md) |
| Data ingestion and chronological splits | [data_loader.py](src/neural_recommendation/data_loader.py), [preprocessing.py](src/neural_recommendation/preprocessing.py) |
| Model implementations | [popularity.py](src/neural_recommendation/popularity.py), [matrix_factorization.py](src/neural_recommendation/matrix_factorization.py), [neural_cf.py](src/neural_recommendation/neural_cf.py) |
| Ranking metrics and final test runner | [evaluation.py](src/neural_recommendation/evaluation.py) |
| How popular the recommendations are in training | [popularity_diagnostic.py](src/neural_recommendation/popularity_diagnostic.py) |
| Final scores and experiment history | [Final test results CSV](results/metrics/final_model_comparison.csv), local `results/metrics/epoch_results.csv`, and the [learning journal](notes/learning_journal.md) |

## How to run

Run from the repository root in Windows PowerShell. Dependencies are listed in [requirements.txt](requirements.txt) without pinned versions.

```powershell
git clone https://github.com/nothd7561/neural-movie-reccomendation.git
cd neural-movie-reccomendation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Download `ml-1m.zip` and place the extracted `ratings.dat` and `movies.dat` directly in `data/raw/movie_data/`. The final comparison needs `ratings.dat`. The standalone popularity demo also reads `movies.dat` to display movie titles. [GroupLens download](https://grouplens.org/datasets/movielens/1m/)

Prepare the ratings and generate the three chronological splits in `data/processed/movie_data/`:

```powershell
python -m neural_recommendation.data_loader
python -m neural_recommendation.preprocessing
```

Run the full comparison:

```powershell
python -m neural_recommendation.evaluation --final-test
```

This command rebuilds the Popularity baseline, trains the other three models, and writes `results/metrics/final_model_comparison.csv`. The runner uses seed 42 and deterministic PyTorch on a single CPU thread. It selects weights during the run and keeps them in memory. Rerunning the command refreshes the results snapshot at that path.

For standalone validation runs:

```powershell
python -m neural_recommendation.evaluation
python -m neural_recommendation.matrix_factorization
python -m neural_recommendation.neural_cf
```

These run Popularity, Matrix Factorization, and Dynamic BPR respectively, with their standalone settings. BCE is available through the final comparison. Fixed sampling is available through `prepare_bpr_data()` for code-level experiments. The exploratory script has a legacy input path and a rating-count indexing issue, so it is outside the reproducible command sequence above.

## Limitations and future work

- **Coverage.** The minimum-rating cutoff removes lesser-rated titles, and the embedding models require training history for users and movies. The test cohort measures recommendations for known users with future positive ratings.
- **Feedback.** Unobserved movies may be enjoyable, and historical ratings reflect what users chose to watch and rate. The models use interaction history without contextual or content features.
- **Cost and measurement.** Popularity achieved the highest offline score with much less training work. Separate coverage and diversity measurements would help assess whether personalization provides benefits beyond this top-10 ranking benchmark.

## Key takeaways

Building the models in PyTorch made the role of user and movie embeddings concrete, from mapping raw IDs to scoring candidate movies. Chronological splits and validation-based checkpointing made it possible to compare ranking behavior on later interactions. The largest improvement within the neural models came from changing what the model learned to rank and how it sampled comparisons. The Popularity result shows why a simple baseline belongs in the experiment from the start.
