import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "movie_data" / "ratings.dat")

PROCESSED_DATA_PATH = (PROJECT_ROOT / "data" / "processed")

def load_ratings():
    ratings = pd.read_csv(RAW_DATA_PATH,
                      sep='::',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'],
                      engine='python')
    #load the .dat file in as a dataframe object, separating by ::

    ratings['timestamp'] = pd.to_datetime(
        ratings['timestamp'], unit='s')
    #convert the timestamp column to a datetime object, seconds turns into actual days/months/we

