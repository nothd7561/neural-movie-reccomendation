
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
    #count how many of the top k relevant movies are in the recommendations

    common_items = len(set(top_k) & relevant_movies)
    #calculate the precision by dividing the number of relevant items in the top k recommendations by k
 
    return common_items / k


def recall_at_k(recommendations: list, relevant_movies: set, k: int = 10):
    """Fraction of relevant movies found in the top K recommendations."""

    if not relevant_movies:
        return 0.0
    #if there are no relevant movies, we cannot calculate recall, so we return 0

    top_k = recommendations[:k]
    common_items = len(set(top_k) & relevant_movies)


    return common_items / len(relevant_movies)


def ndcg_at_k(recommendations: list, relevant_movies: set, k: int = 10):
    """Calculate normalized discounted cumulative gain at K."""
    #How highly did we rank the relevant movies?

    if not relevant_movies:
        return 0.0

    top_k = recommendations[:k]

    #actual discounted cumulative gain
    dcg = sum(
        1 / log2(rank + 2)
        for rank, movie_id in enumerate(top_k)
        if movie_id in relevant_movies
    )

    #best possible discounted cumilative gain
    ideal_hits = min(len(relevant_movies), k)

    idcg = sum(
        1 / log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg


###MODEL EVALUATION

def evaluate_model(recommend_function, user_histories: dict,
                   val_df: pd.DataFrame, k: int = 10):
    """Evaluate recommendations across users with positive validation ratings."""

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    #map each user to their positively rated validation movies
    positive_val = val_df[val_df["is_positive"] == 1]

    relevant_by_user = {}

    for user_id, user_ratings in positive_val.groupby("user_id"):
        relevant_by_user[user_id] = set(user_ratings['movie_id'])
        #create a dictionary of user_ids, and their positive rated movies from the validation set
        #this is so we can use the metric functions to compare the validation with our training predictions

    for user_id, user_history in user_histories.items():

        relevant_movies = relevant_by_user.get(user_id, set())
        #if the user exists in relevant_by_user, retrieve their positive validation movies.
        #if the user doesn't exist, return an empty set 

        # Skip users without positive validation ratings
        if not relevant_movies:
            continue

        recommendations = recommend_function(user_id, user_history, k)
        #get our recommendations from the training set

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


def run_final_test():
    """Evaluate the existing models with the same test users and candidate movies."""
    import random
    import numpy as np
    import torch
    from neural_recommendation import matrix_factorization as mf
    from neural_recommendation import neural_cf as ncf

    seed = 42
    k = 10
    min_ratings = 50
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    user_histories = get_user_history(train_df)
    movie_counts = train_df.groupby("movie_id").size()
    candidate_movies = set(movie_counts[movie_counts >= min_ratings].index)
    ranked_ratings = rank_positive_ratings(train_df)
    ranked_ratings = ranked_ratings[ranked_ratings.index.isin(candidate_movies)]

    def popularity_recommend(user_id, user_history, k):
        return get_recommendations(user_history, ranked_ratings, k)

    print("Training matrix factorization", flush=True)
    torch.manual_seed(seed)
    train_loader, mf_users, mf_movies = mf.prepare_data(train_df)
    val_loader = mf.prepare_validation_data(val_df, mf_users, mf_movies)
    mf_model = mf.MatrixFactorization(
        len(mf_users), len(mf_movies), embedding_dim=16, global_mean=train_df["rating"].mean())
    mf_model = mf.train_model(mf_model, train_loader, val_loader)
    mf_candidates = {movie: index for movie, index in mf_movies.items() if movie in candidate_movies}

    def mf_recommend(user_id, user_history, k):
        return mf.recommend_movies(mf_model, user_id, mf_users, mf_candidates, user_history, k)

    print("Training NCF + BCE", flush=True)
    torch.manual_seed(seed)
    train_loader, user_dict, movie_dict = ncf.prepare_ncf_data(train_df)
    val_loader = ncf.prepare_ncf_validation(val_df, user_dict, movie_dict)
    bce_model = ncf.NeuralCollaborativeFiltering(len(user_dict), len(movie_dict), embedding_dim=32)
    bce_model = ncf.train_ncf(bce_model, train_loader, val_loader)

    def bce_recommend(user_id, user_history, k):
        return ncf.recommend_ncf(
            bce_model, user_id, user_dict, movie_dict, user_history, movie_counts, k, min_ratings)

    print("Training NCF + dynamic BPR", flush=True)
    positive_by_user, unseen_by_user = ncf.prepare_bpr_sampling_data(train_df, user_dict, movie_dict)
    rng = np.random.default_rng(seed)

    def epoch_loader():
        return ncf.sample_bpr_epoch(positive_by_user, unseen_by_user, user_dict, rng, batch_size=256)

    torch.manual_seed(seed)
    bpr_model = ncf.NeuralCollaborativeFiltering(len(user_dict), len(movie_dict), embedding_dim=32)
    bpr_model = ncf.train_bpr(
        bpr_model, epoch_loader, user_dict, val_df, user_histories, movie_dict, movie_counts,
        min_ratings=min_ratings)

    def bpr_recommend(user_id, user_history, k):
        return ncf.recommend_ncf(
            bpr_model, user_id, user_dict, movie_dict, user_history, movie_counts, k, min_ratings)

    test_df = pd.read_csv(TRAIN_DATA_PATH.with_name("test_ratings.csv"))
    val_histories = get_user_history(val_df)
    test_users = set(test_df.loc[test_df["is_positive"] == 1, "user_id"])
    test_histories = {
        user_id: user_history | val_histories.get(user_id, set())
        for user_id, user_history in user_histories.items()
        if user_id in test_users
    }

    rows = []
    for model, recommend, checkpoint_rule in (
        ("Popularity", popularity_recommend, "Training positive-rating counts"),
        ("Matrix factorization", mf_recommend, "Final epoch"),
        ("NCF + BCE", bce_recommend, "Lowest validation BCE loss"),
        ("NCF + dynamic BPR", bpr_recommend, "Highest validation NDCG@10 on 200 users"),
    ):
        print(f"Scoring {model} on the test split", flush=True)

        def checked_recommend(user_id, user_history, k):
            recommendations = recommend(user_id, user_history, k)
            if any(movie not in candidate_movies or movie in user_history for movie in recommendations):
                raise ValueError(f"{model} returned an ineligible movie for user {user_id}")
            if len(recommendations) < k:
                raise ValueError(f"{model} returned {len(recommendations)} recommendations for user {user_id}")
            return recommendations

        scores = evaluate_model(checked_recommend, test_histories, test_df, k=k)
        rows.append(dict(
            model=model, **scores, checkpoint_rule=checkpoint_rule, min_training_ratings=min_ratings))

    results = pd.DataFrame(rows)
    output_path = PROJECT_ROOT / "results" / "metrics" / "final_model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))


###RUN THE EVALUATIONS FOR THE POPULARITY BASELINE

if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--final-test"]:
        run_final_test()
        raise SystemExit(0)

    #1. load data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    #2. build popularity baseline
    ranked_ratings = rank_positive_ratings(train_df)
    user_histories = get_user_history(train_df)

    #3. define how the popularity model ranks recommendations
    def popularity_recommend(user_id, user_history, k):
        return get_recommendations(
            user_history=user_history,
            ranked_ratings=ranked_ratings,
            top_n=k
        )

    #4. evaluate the model
    results = evaluate_model(
        recommend_function=popularity_recommend,
        user_histories=user_histories,
        val_df=val_df,
        k=10
    )

    #5. display the results
    print(f"Users evaluated: {results['users_evaluated']}")
    print(f"Precision@10: {results['precision_at_k']:.4f}")
    print(f"Recall@10:    {results['recall_at_k']:.4f}")
    print(f"NDCG@10:      {results['ndcg_at_k']:.4f}")
