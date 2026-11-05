import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

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

user_counts = ratings.groupby('user_id').size().reset_index(name='user_count')
#count the number of ratings each user had made and add that as a new column to the dataframe

movie_rating_counts = ratings.groupby('movie_id').size().reset_index(name='rating_count').sort_values(by='rating_count', ascending=False)
#count the number of ratings each movie received

rating_counts = ratings.value_counts('rating').sort_index()
#create a new dataframe with just the counts of each rating value, sorted by the rating value

positive_ratings = rating_counts[ratings['rating'] >= 4] #count the number of ratings that are 4 or higher, which are conidered positive ratings
positive_percentage = (positive_ratings.sum() / rating_counts.sum()) * 100  #find the percentage of positive ratings

movie_histogram = sns.histplot(data=movie_rating_counts, x='rating_count', bins=30, kde=True)
#create a histogram of the number of ratings per movie, with 30 bins and a kernel density estimate (kde) overlay
histogram = sns.histplot(data=user_counts, x='user_count', bins=30, kde=True)
#create a histogram of the number of ratings per user, with 30 bins and a kernel density estimate (kde) overlay
rating_bar_chart = sns.barplot(x=rating_counts.index, y=rating_counts.values)

#print(movie_rating_counts.head())
#print(movie_rating_counts.describe())

ratings['is_positive'] = (ratings['rating'] >= 4).astype(int)
#perform vectorization, apply T/F according to the condition, then convert those T/F to 1's and 0's

ratings = ratings.sort_values(
    by=['user_id', 'timestamp']
).reset_index(drop=True)
#sort our dataframe by user_id and timestamp, then reset the index to be sequential again

print(ratings.head())

training_data = ratings.iloc[0:71]
validation_data = ratings.iloc[71:86]
test_data = ratings.iloc[86:100]