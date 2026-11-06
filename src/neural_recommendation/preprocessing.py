import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "movie_data" / "ratings.csv")

def split_data(ratings: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split the ratings DataFrame into train, validation, and test sets based on user interactions."""

    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("Train and validation ratios must be between 0 and 1.")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Train and validation ratios must sum to less than 1.0.")

    train_group = []
    val_group = []
    test_group = []
    #create empty lists to hold our train, validation, and test dataframes

    for user_id, user_ratings in ratings.groupby('user_id'):
        #the groupby function creates a tuple with user_id, and the dataframe with all that user's ratings

        #sort each user interaction from earliset to latest
        user_ratings = user_ratings.sort_values(
            ['timestamp', 'movie_id'] #two user ratings can have the same timestamp, so we also sort by movie id
        )

        n = len(user_ratings)
        #the number of ratings for eah user (number of rows in their unique dataframe)

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

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path = PROJECT_ROOT / "data" / "processed" / "movie_data"):
    """Save the train, validation, and test DataFrames to CSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / 'train_ratings.csv', index=False)
    val_df.to_csv(output_dir / 'val_ratings.csv', index=False)
    test_df.to_csv(output_dir / 'test_ratings.csv', index=False)

if __name__ == "__main__":
    ratings = pd.read_csv(PROCESSED_DATA_PATH)

    train_df, val_df, test_df = split_data(ratings)

    save_splits(train_df, val_df, test_df)

    print("Chronological splits saved successfully!")

#total = len(ratings)

#print(f"Train: {len(train_df):,} ({len(train_df)/total:.2%})")
#print(f"Validation: {len(val_df):,} ({len(val_df)/total:.2%})")
#print(f"Test: {len(test_df):,} ({len(test_df)/total:.2%})")