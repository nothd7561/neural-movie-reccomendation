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
