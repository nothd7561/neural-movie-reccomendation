# November 3, 2026 — Starting the Neural Recommendation System

Today I started working on my Neural Movie Recommendation System. The goal is to build a model using PyTorch that recommends movies based on a user's historical preferences. I'm planning to compare a popularity-based baseline, matrix factorization, and neural collaborative filtering to see whether using a neural network actually improves recommendations.

I set up my GitHub repository, installed Codex in VS Code, and created the initial project folder structure. I also decided to approach this project a little differently from my previous ML projects. Instead of following instructions or relying on AI to write everything, I want to think through the decisions myself first, get feedback on my reasoning, and understand why certain approaches work better than others.

Most of today was spent figuring out how recommendation systems actually work and planning our approach.

Initially, I thought we'd need movie information like genres, actors, directors, language, and descriptions alongside user ratings. I learned that while those features could be useful, they aren't necessary for collaborative filtering. The model can learn patterns from user–movie interactions alone. For example, if two users consistently rate the same movies similarly, a movie that one user enjoys could be a good recommendation for the other.

One thing I had to understand was the difference between predicting exact ratings and ranking movies. We're focusing on Top-10 recommendations, meaning the model's job isn't necessarily to predict that someone will give a movie exactly five stars. Instead, it needs to identify and rank movies they're likely to enjoy.

We also discussed how to define positive interactions. I originally suggested treating anything above 3.5 stars as positive, since three stars feels neutral. MovieLens 1M only uses whole-number ratings, so we settled on four and five stars being positive.

Another important topic was missing interactions. I figured that movies a user hasn't rated shouldn't automatically be treated as negative examples, since they might simply never have watched them. I initially thought we could represent those interactions with missing values, but learned that we don't need to create entries for every unrated movie. The dataset is naturally sparse. We'll eventually need to explore negative sampling when training the models.

For splitting our dataset, I suggested grouping interactions by user and timestamp to avoid data leakage. We decided to use chronological splits within each user's history, rather than separating users entirely. I initially suggested 60/20/20 for training, validation, and testing, but we're tentatively going with 70/15/15 to give the models more training data.

Finally, we went over evaluation metrics. I learned that more correct recommendations don't necessarily mean a better-ranked list. Precision@10 measures how many of the ten recommendations are relevant, Recall@10 measures how many relevant movies were retrieved, and NDCG@10 considers where relevant movies appear in the ranking. We're planning to use NDCG@10 as our main metric.

I haven't started working with the actual dataset yet. Today was mostly about understanding the problem and establishing a methodology.

**Next session:** Download MovieLens 1M, load the ratings into pandas, and begin exploring the user–movie interactions before preprocessing.

# November 4, 2026 — Exploring the MovieLens Dataset

Today I started working with the actual MovieLens 1M dataset after spending the previous session planning out the recommendation system. Most of today's work involved loading the data using pandas and getting a better understanding of how users interact with movies.

I initially thought we should load the users dataset first, since we'd eventually need to connect all our information together. However, I realized that starting with the ratings dataset made more sense because it already contains both the user and movie IDs, along with their respective ratings and timestamps. Since we're using collaborative filtering, these interactions are what we're primarily interested in.

For loading the data, I used `pathlib` to construct the file paths and `pd.read_csv()` to read the `.dat` file. I wasn't familiar with `.dat` files initially, but learned that the extension itself doesn't necessarily determine how to read the data. MovieLens uses `::` as its delimiter, so I specified that along with the Python parsing engine. I also learned that the timestamps are Unix timestamps, which represent the number of seconds since January 1, 1970. They'll be useful later when we split each user's interactions chronologically to avoid using future ratings to predict earlier preferences.

After loading the dataset, I started investigating how many movies each user rated. My first approach used `groupby()` with `transform('count')`, which worked, but repeated each user's rating count across all their interactions. I learned that using `groupby().size()` and creating a separate summary DataFrame was more appropriate for EDA because it gives one row per user.

I then created a histogram to examine the distribution of ratings per user. Initially, I thought the distribution was left-skewed because most observations were concentrated on the left. I learned that skewness is determined by the direction of the tail, meaning our distribution was actually right-skewed.

The median user had rated 96 movies, while the mean was approximately 166. The least active user had 20 ratings, and the most active had 2,314. This showed that some users contribute significantly more interactions than others, which could affect how well our models learn different users' preferences.

Next, I investigated the number of ratings each movie received. This distribution was also heavily right-skewed. Across 3,706 rated movies, the median was 123.5 ratings, while the mean was approximately 270. The most-rated movie received 3,428 ratings, while some movies had only one.

This introduced another important concept: item sparsity. Just as users with fewer interactions provide less information about their preferences, movies with very few ratings give our model less information to learn useful movie embeddings. I also learned that popular movies could dominate our recommendation results, which is something we'll want to investigate when comparing our popularity baseline against the neural models.

One visualization mistake I made was overlaying the user and movie rating-count distributions on the same histogram. Although this showed their similar shapes, the two distributions represent different quantities and contain different numbers of observations. Separate plots would make them easier to interpret.

Finally, I started examining the actual ratings from 1 to 5. I decided a bar chart would be appropriate because we're comparing the frequencies of five distinct rating values. I used `value_counts()` with `sort_index()` to count the ratings and keep them in numerical order, then plotted the results with Seaborn.

Overall, today was mostly about EDA and understanding the characteristics of the dataset before training anything. The biggest takeaway was that recommendation data has two important sources of imbalance: how active users are and how popular movies are. Both could affect the quality of our recommendations, especially for users and movies with limited interaction histories.

Next session, I'll finish analyzing the rating distribution, calculate the percentage of positive interactions (ratings of 4 or 5), and start preparing the data for our recommendation models.
