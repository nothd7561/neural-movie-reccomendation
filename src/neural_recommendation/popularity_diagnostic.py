
import pandas as pd
import numpy as np

def analyze_recommendation_popularity(
    recommend_fn,
    user_histories,
    train_df,
    user_ids,
    k=10
):
    """
    Calculate the average training popularity of recommended movies.

    Popularity = number of training ratings received by a movie.
    """

    movie_popularity = train_df.groupby("movie_id").size()
    popularity_scores = []

    for user_id in user_ids:
        user_history = user_histories[user_id]
        recommendations = recommend_fn(user_id, user_history, k)

        for movie_id in recommendations:
            rating_count = movie_popularity.get(movie_id, 0)
            popularity_scores.append(rating_count)

    return {
        "mean_popularity": float(np.mean(popularity_scores)),
        "median_popularity": float(np.median(popularity_scores)),
        "num_recommendations": len(popularity_scores),
    }
