import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "movie_data" / "ratings.csv")

ratings = pd.read_csv(PROCESSED_DATA_PATH)
#read our processed csv

def split_data(ratings: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):

    train_group = []
    val_group = []
    test_group = []
    #create empty lists to hold our train, validation, and test dataframes

    for user_id, user_ratings in ratings.groupby('user_id'):
        #sort each user interaction from earliset to latest
        user_ratings = user_ratings.sort_values(
            ['timestamp', 'movie_id'] #two user ratings can have the same timestamp, so we also sort by movie id
        )

        n = len(user_ratings)
        #the number of ratings for eah user

        train_end = int(n * train_ratio) 
        val_end = int(n * (train_ratio + val_ratio))
        #multiply the number of ratings by the train and validation ratios to get the index where we want to split the data

        train_group.append(user_ratings.iloc[:train_end]) 
        val_group.append(user_ratings.iloc[train_end:val_end])
        test_group.append(user_ratings.iloc[val_end:])

    train = pd.concat(train_group, ignore_index=True)
    val = pd.concat(val_group, ignore_index=True)
    test = pd.concat(test_group, ignore_index=True)
    #we utilize the concat function to combine the small dataframes into one huge one

    return train, val, test

train_df, val_df, test_df = split_data(ratings)

print(train_df.head(), val_df.head(), test_df.head())