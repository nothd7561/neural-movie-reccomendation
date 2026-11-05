import pandas as pd
from pathlib import Path

file_dir = Path(__file__).resolve().parents[1]
#locate the project root relative to this script

db_path = file_dir / 'data' / 'raw' / 'movie_data'
#construct the directory containing the raw MovieLens files

ratings = pd.read_csv(db_path / 'ratings.dat',
                      sep='::',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'],
                      engine='python')
#load user-movie ratings, specifying the custom "::" delimiter
#and column names because the file has no heade


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
#convert the timestamp column to a datetime object, seconds turns into actual days/months/weeks

rating_counts = ratings.groupby('user_id').size().reset_index(name='rating_count')
#count the number of ratings each user had made and add that as a new column to the dataframe

movie_rating_counts = ratings.groupby('movie_id').size().reset_index(name='rating_count').sort_values(by='rating_count', ascending=False)
#count the number of ratings each movie received

#print(movie_rating_counts.head())
#print(movie_rating_counts.describe())

import seaborn as sns
import matplotlib.pyplot as plt

histogram = sns.histplot(data=rating_counts, x='rating_count', bins=30, kde=True)
#plt.show()

movie_histogram = sns.histplot(data=movie_rating_counts, x='rating_count', bins=30, kde=True)
plt.show()