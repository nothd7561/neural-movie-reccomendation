
import pandas as pd
from math import log2
from pathlib import Path

from neural_recommendation.popularity import (
    rank_positive_ratings,
    get_user_history,
    get_recommendations
)

###FILE PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "movie_data" / "train_ratings.csv"
)

VAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "movie_data" / "val_ratings.csv"
)

###EVALUATION METRICS

def precision_at_k(recommendations: list, relevant_movies: set, k: int = 10):
    """Fraction of the top K recommendations that are relevant."""

    top_k = recommendations[:k]
    common_items = len(set(top_k) & relevant_movies)

    return common_items / k


def recall_at_k(recommendations: list, relevant_movies: set, k: int = 10):
    """Fraction of relevant movies found in the top K recommendations."""

    if not relevant_movies:
        return 0.0

    top_k = recommendations[:k]
    common_items = len(set(top_k) & relevant_movies)

    return common_items / len(relevant_movies)


def ndcg_at_k(recommendations: list, relevant_movies: set, k: int = 10):
    """Calculate normalized discounted cumulative gain at K."""

    if not relevant_movies:
        return 0.0

    top_k = recommendations[:k]

    # Actual discounted cumulative gain
    dcg = sum(
        1 / log2(rank + 2)
        for rank, movie_id in enumerate(top_k)
        if movie_id in relevant_movies
    )

    # Best possible discounted cumulative gain
    ideal_hits = min(len(relevant_movies), k)

    idcg = sum(
        1 / log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg


###MODEL EVALUATION

def evaluate_model(recommend_fn, user_histories: dict,
                   val_df: pd.DataFrame, k: int = 10):
    """Evaluate recommendations across users with positive validation ratings."""

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    # Map each user to their positively rated validation movies
    positive_val = val_df[val_df["is_positive"] == 1]

    relevant_by_user = {
        user_id: set(user_ratings["movie_id"])
        for user_id, user_ratings in positive_val.groupby("user_id")
    }

    for user_id, user_history in user_histories.items():

        relevant_movies = relevant_by_user.get(user_id, set())

        # Skip users without positive validation ratings
        if not relevant_movies:
            continue

        recommendations = recommend_fn(user_id, user_history, k)

        if len(recommendations) != k or len(set(recommendations)) != k:
            raise ValueError(f"Expected {k} unique recommendations for user {user_id}")

        precision_scores.append(
            precision_at_k(recommendations, relevant_movies, k)
        )

        recall_scores.append(
            recall_at_k(recommendations, relevant_movies, k)
        )

        ndcg_scores.append(
            ndcg_at_k(recommendations, relevant_movies, k)
        )

    if not precision_scores:
        raise ValueError("No eligible validation users were found.")

    return {
        "users_evaluated": len(precision_scores),
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores)
    }


###RUN THE EVALUATIONS FOR THE POPULARITY BASELINE

if __name__ == "__main__":

    # 1. Load data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    # 2. Build popularity baseline
    ranked_ratings = rank_positive_ratings(train_df)
    user_histories = get_user_history(train_df)

    # 3. Define how the popularity model recommends movies
    def popularity_recommend(user_id, user_history, k):
        return get_recommendations(
            user_history=user_history,
            ranked_ratings=ranked_ratings,
            top_n=k
        )

    # 4. Evaluate the model
    results = evaluate_model(
        recommend_fn=popularity_recommend,
        user_histories=user_histories,
        val_df=val_df,
        k=10
    )

    # 5. Display results
    print(f"Users evaluated: {results['users_evaluated']}")
    print(f"Precision@10: {results['precision_at_k']:.4f}")
    print(f"Recall@10:    {results['recall_at_k']:.4f}")
    print(f"NDCG@10:      {results['ndcg_at_k']:.4f}")
