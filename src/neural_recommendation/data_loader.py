import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "movie_data" / "ratings.dat")

MOVIES_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "movie_data" / "movies.dat")

PROCESSED_DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "movie_data" / "ratings.csv")

def load_movies():
    """Load MovieLens movie .dat file"""
    movies = pd.read_csv(MOVIES_DATA_PATH,
                         sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )

    return movies

def load_ratings(positive_threshold: int = 4):
    """Load MovieLens ratings and perform preprocessing"""


    ratings = pd.read_csv(RAW_DATA_PATH,
                      sep='::',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'],
                      engine='python')
    #load the .dat file in as a dataframe object, separating by ::

    # Validate required fields before preprocessing
    if ratings.isna().any().any():
        raise ValueError("Ratings dataset contains missing values.")

    if ratings.duplicated(subset=["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user-movie interactions detected.")

    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Invalid rating values detected.")

    ratings['timestamp'] = pd.to_datetime(
        ratings['timestamp'], unit='s', utc=True)
    #convert Unix timestamps (seconds since the Unix epoch) to UTC datetimes

    ratings['is_positive'] = (ratings['rating'] >= positive_threshold).astype(int)
    #perform vectorization, apply T/F according to the condition, then convert those T/F to 1's and 0's

    ratings = ratings.sort_values(
        by=['user_id', 'timestamp']
        ).reset_index(drop=True)
    #sort our dataframe by user_id and timestamp, then reset the index to be sequential again

    return ratings

def save_ratings(ratings: pd.DataFrame):
    """Save the preprocessed ratings DataFrame to a CSV file."""

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True)

    ratings.to_csv(PROCESSED_DATA_PATH, index=False)

loaded_ratings = load_ratings()
save_ratings(loaded_ratings)