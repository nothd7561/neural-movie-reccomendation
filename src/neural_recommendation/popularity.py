import pandas as pd
from pathlib import Path
from neural_recommendation.data_loader import load_movies

PROJECT_ROOT = Path(__file__).resolve().parents[2]


PROCESSED_DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "movie_data" / "train_ratings.csv")

def rank_positive_ratings(train_df: pd.DataFrame): 
    """Count the number of positive ratings for each movie, then rank in descending order."""

    if 'is_positive' not in train_df.columns: #if our csv is missing the is postive column, raise an error
        raise ValueError("The 'is_positive' column is missing from the DataFrame.")

    ranked_ratings = train_df['is_positive'].groupby(
        train_df['movie_id']).sum().sort_values(
            ascending=False
        )
    #group the dataframe by movie_id, then sum the is_positive column to get the total number of positive ratings for each movie
    #then sort the values in descending order so we can see which movies have the most positive ratings
    
    return ranked_ratings

def get_user_history(train_df: pd.DataFrame):
    """Create a dictionary mapping each user to their rated movie IDs."""
    user_movies = {}

    for user_id, user_ratings in train_df.groupby('user_id'):
        user_movies[user_id] = set(user_ratings['movie_id'])

    return user_movies
    #returns a dict

def get_recommendations(user_history: set, ranked_ratings: pd.Series, top_n: int = 10):
#user_history is the set, since we plan to call this function for each user
    """Recommend movies to users based on popularity, excluding those they've already rated."""
    recommendations = []

    for movie_id in ranked_ratings.index:  #loop over the positive rating movie ids
        if movie_id not in user_history:  #if the movie id is not in the user's history, we can recommend it
            recommendations.append(movie_id) 
            if len(recommendations) >= top_n:
                break

    return recommendations
    #returns a list

if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_DATA_PATH)

    ranked_ratings = rank_positive_ratings(train_df)
    user_movies = get_user_history(train_df)

    user_id = 1

    recommendations = get_recommendations(
        user_history=user_movies[user_id],
        ranked_ratings=ranked_ratings,
        top_n=10)
    

    movie_df = load_movies()
    #load the movie data into a dataframe, using our load_movies function from data_loader.py

    recommended_movie_titles = movie_df.set_index('movie_id').loc[recommendations].reset_index()

    print(recommended_movie_titles.to_string(index=False))






