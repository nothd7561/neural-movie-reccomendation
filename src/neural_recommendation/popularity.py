import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


PROCESSED_DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "movie_data" / "train_ratings.csv")

def rank_positive_ratings(csv_path: Path = PROCESSED_DATA_PATH): 
    """Count the number of positive ratings for each movie, then rank in descending order."""
    if not csv_path.exists():  #check for the csv path, make sure it exists, if not, raise an error
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    ranked_ratings = pd.read_csv(csv_path) #read the csv file into a dataframe

    if 'is_positive' not in ranked_ratings.columns: #if our csv is missing the is postive column, raise an error
        raise ValueError("The 'is_positive' column is missing from the DataFrame.")

    ranked_ratings = ranked_ratings['is_positive'].groupby(
        ranked_ratings['movie_id']).sum().sort_values(
            ascending=False
        )
    #group the dataframe by movie_id, then sum the is_positive column to get the total number of positive ratings for each movie
    #then sort the values in descending order so we can see which movies have the most positive ratings
    
    return ranked_ratings

def get_user_history(train_df: pd.DataFrame):
    """Take away movies that a user has already rated, so we don't recommend them again."""
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

    ranked_ratings = rank_positive_ratings()
    user_movies = get_user_history(train_df)

    # Generate recommendations for individual users
    recommendations_1 = get_recommendations(
        user_movies[1], ranked_ratings
    )

    recommendations_2 = get_recommendations(
        user_movies[2], ranked_ratings
    )

    print("User 1:", recommendations_1)
    print("User 2:", recommendations_2)


