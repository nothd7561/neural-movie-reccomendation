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

# November 12, 2026 — Neural Recommendation System

Today I continued working on the Neural Movie Recommendation project, focusing mainly on understanding how matrix factorization works and beginning its implementation in PyTorch.

### Popularity Baseline Results

Before moving into matrix factorization, I finished running the popularity baseline evaluation across our validation users.

The results were:

- **Users evaluated:** 5,933
- **Precision@10:** 0.0641
- **Recall@10:** 0.0501
- **NDCG@10:** 0.0759

The popularity baseline works by counting the number of positive training ratings each movie receives, ranking movies based on those counts, and recommending the top unseen movies to each user.

The biggest limitation is that it isn't really personalized. Two users will generally receive the same recommendations except for movies they've already rated.

These results now serve as our benchmark. The goal is to see whether matrix factorization and neural collaborative filtering can improve recommendation quality by learning individual preferences.

### Understanding Matrix Factorization

I started learning how matrix factorization works. The main idea is that instead of ranking movies globally, we assign each user and movie an embedding.

An **embedding is a vector of learned numbers representing a user or movie**. For example, a user embedding might be `[0.8, 0.2, 0.5]`.

These numbers aren't manually assigned, and they don't necessarily represent obvious characteristics like action or romance. They're learned through training.

An **embedding table** is just a matrix containing all the embeddings. Each row represents a user or movie, while each column represents an embedding dimension.

For example, 6,040 users with 16 embedding dimensions would give us a table of shape `6040 × 16`.

We use the dot product between a user's embedding and a movie's embedding to estimate their compatibility. This means multiplying the corresponding elements of both vectors and summing the products.

During training, the embeddings are updated to produce predictions that more closely match actual ratings.

### Why We Need Embedding Indices

One thing I initially struggled with was understanding the difference between original IDs, embedding indices, and embeddings.

- **Original ID:** Identifies a user or movie in MovieLens.
- **Embedding index:** Identifies the row in an embedding table.
- **Embedding:** The actual vector of learnable numbers stored in that row.

Because MovieLens IDs aren't necessarily consecutive, we created dictionaries such as `user_to_idx` and `movie_to_idx`.

For example, if User 20 maps to index 1, PyTorch retrieves row 1 of the user embedding table.

The indices don't change during training. The numbers inside the embeddings do.

### Preparing the Data for PyTorch

I worked on `prepare_data()`, which converts our MovieLens data into a format PyTorch can use.

First, we extract unique user and movie IDs and map them to consecutive indices.

Then we use pandas `.map()` to convert the original IDs into embedding indices.

We also extract the actual rating column, which is our supervised learning target.

We convert these into three tensors:

- `user_tensor`: Contains user embedding indices.
- `movie_tensor`: Contains movie embedding indices.
- `rating_tensor`: Contains actual movie ratings.

An important distinction is that **these tensors don't contain the learned embeddings**. They contain the information needed to retrieve embeddings and train the model.

We use `torch.long` for indices because embedding lookups require integer indices, and `torch.float32` for ratings because they're used in numerical predictions and loss calculations.

I also learned the difference between `TensorDataset` and `DataLoader`.

`TensorDataset` groups corresponding user indices, movie indices, and ratings into training observations.

`DataLoader` provides those observations in batches and can shuffle them during training.

This makes it possible to process the dataset in smaller batches instead of handling every observation simultaneously.

### Building the MatrixFactorization Class

I started implementing the `MatrixFactorization` class, which inherits from PyTorch's `nn.Module`.

`nn.Module` provides the infrastructure for building models and tracking learnable parameters.

Within `__init__()`, we created two embedding layers:

- `nn.Embedding(num_users, embedding_dim)`
- `nn.Embedding(num_movies, embedding_dim)`

I initially had trouble understanding what these layers actually contained. They're essentially learnable tables of numbers, stored in the layer's weight matrix.

When we pass user indices into `self.user_embeddings(user_indices)`, PyTorch retrieves the corresponding rows from the user embedding table.

The `forward()` method then retrieves the user and movie vectors and calculates their dot products.

I also learned that we don't need to manually loop through every observation. PyTorch supports vectorized operations.

For example:

`(user_vectors * movie_vectors).sum(dim=1)`

This multiplies corresponding embedding values and sums them across each row, producing one prediction per user–movie observation.

### User Biases, Movie Biases, and Global Mean

We also added user and movie biases to improve the prediction formula.

A **user bias** accounts for users who generally rate movies higher or lower than average.

A **movie bias** accounts for movies that generally receive higher or lower ratings.

Each user and movie gets one learnable bias value, so their bias embedding tables have one column.

We use `.squeeze(-1)` when retrieving biases because the output initially has shape `(batch_size, 1)`, while our predicted ratings have shape `(batch_size,)`.

Squeezing removes that unnecessary dimension without removing any values. It also prevents incorrect broadcasting when adding biases to predictions.

We also included a global mean rating, calculated from training data, to give the model a reference point for predictions.

The resulting prediction formula is:

**Predicted rating = global mean + user/movie embedding dot product + user bias + movie bias.**

The embeddings and biases are learned during training. We don't manually calculate their final values.

### Beginning train_model() — MSE and Adam

Toward the end of the session, I started working on the `train_model()` function.

Its purpose is to take our model and training data, calculate prediction errors, and update the learnable parameters to reduce those errors.

We established four parameters:

- `model`: The matrix factorization object containing the learnable parameters.
- `train_loader`: Supplies batches of training data.
- `epochs`: Number of complete passes through the training dataset.
- `learning_rate`: Controls the overall scale of parameter updates.

We decided to use MSE because we're training our matrix factorization model to predict numerical star ratings.

We initialized the loss function using `nn.MSELoss()`.

I also asked what an **Adam optimizer** actually is because I hadn't learned it before.

Adam stands for **Adaptive Moment Estimation**. It's an optimization algorithm that updates a model's parameters to reduce its loss.

Ordinary gradient descent uses the current gradient to determine a parameter update. Adam also maintains moving estimates of previous gradients and squared gradients, allowing it to adapt updates for individual parameters.

In simpler terms, Adam is an adaptive gradient-based optimizer.

For our model, Adam will update the learnable numbers inside the user embedding table, movie embedding table, and bias tables.

The syntax is:

`optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)`

`model.parameters()` supplies the learnable parameters, while `lr` specifies the learning rate.

One important distinction I learned is that **backpropagation calculates gradients, while the optimizer uses those gradients to update parameters**.

PyTorch handles the mathematics of Adam, so we don't need to implement its formulas ourselves.

### What I Want to Remember

The main concepts I want to be able to explain without looking them up are:

- Why matrix factorization can generate personalized recommendations while a popularity baseline cannot.
- What an embedding is and why embeddings are stored in tables.
- Why we map original IDs to embedding indices.
- How PyTorch retrieves embeddings and uses their dot product to make predictions.
- How tensors, TensorDataset, and DataLoader work together.
- Why biases and a global rating mean are useful.
- What `nn.Module`, `nn.Embedding`, and `forward()` do.
- What Adam is, how it differs from basic gradient descent, and what parameters it updates.

### Next Steps

I've begun setting up `train_model()`, but the training loop isn't finished yet.

Next, I need to implement the epoch and batch loops, generate predictions, calculate MSE, call backpropagation, and use Adam to update the model parameters.

After training works, I'll implement `recommend_movies()`, generate personalized recommendations, and evaluate matrix factorization using the same Precision@10, Recall@10, and NDCG@10 metrics as our popularity baseline.

The main goal is still to determine whether learning personalized user–movie embeddings actually improves recommendation quality.

# November 20, 2026 — Neural Recommender Progress

I spent the past few sessions trying to figure out why my neural collaborative filtering model was doing so poorly compared to the popularity baseline. I went back through data preparation, evaluation, matrix factorization, and NCF. I feel a lot more comfortable with how user and movie IDs become embedding indices, how tensors and DataLoaders fit together, and how the forward pass uses learned vectors to make predictions.

### What I Tried

- **Popularity filtering.** NCF was recommending a lot of movies with hardly any training ratings. I tried minimum-rating thresholds of 0, 10, 25, and 50, and filtering improved NDCG substantially.
- **Validation and checkpoints.** I added validation BCE and started restoring the epoch with the lowest validation loss.
- **Embedding size.** I compared 8, 16, 32, and 64 dimensions. Larger embeddings generally ranked movies better, although 64 dimensions showed much more overfitting.
- **Training objective.** I realized BCE was learning whether a movie someone had already rated got a positive rating. What I actually want at recommendation time is a ranking of movies the user hasn't rated.
- **BPR.** I implemented negative sampling and a pairwise ranking loss so the model learns to score a known positive movie above a sampled unobserved movie.

### What Happened With BPR

The initial BPR model reached **NDCG@10 = 0.0682**, compared with about **0.0201** for the 32-dimensional BCE model. After adding validation-based checkpoint selection, full validation NDCG@10 reached **0.0726** across 5,933 users. The popularity baseline is still a little ahead at **0.0759**.

Epoch 6 was the best checkpoint. Training loss kept going down after that, but validation NDCG dropped. The biggest thing I want to remember is how much the training objective matters. Switching from classification to pairwise ranking made a bigger difference than increasing the embedding size.

### Next Steps

I want to try drawing fresh negatives every epoch, then run multiple random seeds and standardize the model comparisons. After that, I want to build a Tableau dashboard and a small recommendation app. I'm also thinking about how to present this as a complete ML workflow with a clear story connecting data preparation, training, and evaluation.
