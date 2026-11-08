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

# November 5, 2026 — Preprocessing and Chronological Splitting

Today I continued working on the MovieLens recommendation system, mainly transitioning from EDA into the actual preprocessing pipeline. I wanted to make the project reproducible from the start instead of having to go back later and reorganize everything, which was something I ran into with my soccer predictor.

I started by calculating the percentage of positive ratings in the dataset. I wasn't sure how to do this in pandas, but learned that Boolean expressions can be averaged because `True` and `False` correspond to 1 and 0. Around 61.4% of the observed ratings were positive, using our threshold of 4 stars or higher. This doesn't mean users would enjoy 61.4% of all movies, since people choose which movies they watch and rate.

I then worked on creating the `is_positive` column. My initial approach used a for loop with an if statement, but I accidentally assigned values to the entire column on every iteration instead of individual rows. I learned how to fix this using `.loc[]`, as well as how to build a list of labels with a loop. Ultimately, vectorization with `.astype(int)` was simpler and more efficient.

After that, I started preparing the dataset for chronological splitting. I initially thought we'd need to extract the hours and seconds from timestamps to sort them correctly, but realized Unix timestamps already represent the complete time as seconds since January 1, 1970. We can sort them directly using `sort_values()`.

We also went over `.iloc[]`. I made an off-by-one mistake when splitting 100 interactions into training, validation, and testing sets. The important thing I learned was that Python includes the starting position but excludes the ending position when slicing.

I suggested saving the processed dataset before continuing, since I didn't want our preprocessing work to depend on manually rerunning notebook cells. We checked for duplicate user–movie interactions, missing values, and ratings outside the valid 1–5 range. All of those checks passed, so there wasn't any cleaning needed based on those issues.

I then created `data_loader.py` with reusable functions for loading and saving the ratings. The loader validates the dataset, converts the timestamps, creates binary labels, and sorts the interactions. I also made the positive-rating threshold configurable instead of hardcoding it. One mistake I made was initially pointing the CSV save path to a folder rather than a filename, which I corrected.

We briefly discussed reproducibility as well. I initially figured that because MovieLens 1M is a fixed dataset, running the same code later should produce identical results. I learned that software dependencies and random operations during model training can also affect reproducibility. We'll need to record our library versions and control random seeds once we start training models.

The second half of the session focused on chronological splitting. We're planning a 70/15/15 split within each user's rating history. I initially suggested taking 15% of the leftover interactions after allocating 70% to training, but realized that validation should receive 15% of the original total, not 15% of the remaining 30%.

I also asked why we weren't using scikit-learn's `train_test_split()` or cross-validation like in my previous ML projects. The main reason is that we're trying to predict future preferences using earlier interactions. Random splitting could allow future ratings into the training history. GroupKFold would create a different problem, since holding out entire users means our model wouldn't have learned embeddings for them.

One thing I spent extra time understanding was the actual data structure of `groupby()`. I was confused about what the loop variables represented, so I worked through how pandas returns a group identifier and a DataFrame containing that group's rows. I now understand that `train_groups` is a list of smaller DataFrames, one per user, while `pd.concat()` combines them into one large training DataFrame.

We finished by discussing an evaluation issue: some users might have no positive interactions in their held-out sets. That would make certain recommendation metrics undefined for those users. We'll need to decide how to handle this consistently rather than ignoring the problem.

Overall, today's biggest progress was moving from exploratory analysis toward a reusable ML pipeline. I also got a much better understanding of pandas indexing, group iteration, and why chronological evaluation matters.

**Next session:** Finish implementing and testing the chronological split, decide how to handle users with no positive held-out interactions, save the splits, and begin building the popularity baseline.

# November 7, 2026 — Building the Popularity Baseline

Today I continued working on the Neural Recommendation System, mainly finishing the preprocessing stage and starting our first actual recommendation model.

I started by checking that our chronological splitting function worked correctly. I had already written the function to split each user's interactions into training, validation, and testing sets using a 70/15/15 ratio, and confirmed that the outputs looked right. I also implemented `save_splits()` to save the three datasets as separate CSV files, which means we won't need to rerun our entire preprocessing pipeline every time we train a model.

One thing I was still confused about was the purpose of the `if __name__ == "__main__":` statement. I learned that when Python imports a module, it executes the top-level code but doesn't automatically execute the bodies of functions that are only being defined. The `__main__` guard allows us to run our preprocessing pipeline directly without triggering it every time another script imports one of its functions. This also helped me understand that our Python modules are mainly collections of reusable functions, with optional execution code at the bottom.

We also finalized how we're going to handle users who don't have positive interactions in their held-out datasets. Rather than deleting those users from training, we'll keep them and only exclude them from ranking metrics that require positive test or validation interactions. I learned that this could introduce selection bias, since the users we evaluate might have different rating habits from the users we exclude. We'll need to report evaluation coverage alongside our model results.

After that, I moved on to building the popularity baseline. Initially, I thought popularity should simply mean the total number of ratings a movie receives, since a movie with more ratings is generally more popular. However, for our project, we decided to define popularity using the number of positive ratings instead. This aligns better with our goal of recommending movies people are likely to enjoy.

I wrote a function to group ratings by `movie_id`, sum the `is_positive` column, and rank movies in descending order. My first implementation used a dictionary and a for loop, but I later simplified it using pandas `groupby()`, `sum()`, and `sort_values()`. I also caught an important data leakage issue: I had initially used the complete processed ratings dataset rather than the training split. We corrected that so our popularity scores only use training data.

The next challenge was making recommendations for individual users. I learned that we need to exclude movies someone has already rated, regardless of whether they liked or disliked them. I initially created a function called `rid_duplicates()` that built a dictionary mapping each user ID to a set of movies they'd rated. I then separated this into a `get_user_history()` function because identifying previously rated movies and filtering recommendations are two different responsibilities. This should also make the code easier to reuse when we build the more advanced models.

I then worked on `get_recommendations()`, which takes a user's rating history, goes through the popularity ranking, skips movies they've already rated, and collects the top 10 unseen movies. My loop logic was correct, but I initially tried to call the ranking function again inside the recommendation function, even though the ranking was already being passed as an argument. I also learned that iterating directly over a pandas Series gives its values, not its index. Since our Series uses movie IDs as the index and positive-rating counts as the values, we need to iterate over `.index` to access the actual movie IDs.

Overall, I feel like I'm starting to understand how the different parts of a machine learning project fit together. We're keeping the data loading, preprocessing, popularity ranking, and recommendation filtering in separate functions rather than putting everything into one large script. The popularity model itself is pretty simple, but it's important because it'll give us a baseline to compare against matrix factorization and neural collaborative filtering later.

We now have the core popularity-ranking and recommendation functions written, although we haven't tested the complete recommendation pipeline on actual users yet.

**Next session:** Run the popularity baseline, verify that it returns 10 unseen movies per user, connect movie IDs to actual titles, and begin implementing our ranking evaluation metrics.
