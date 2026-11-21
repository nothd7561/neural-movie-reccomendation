import argparse
from functools import partial
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import torch.optim as optim
from neural_recommendation.popularity import get_user_history
from neural_recommendation.evaluation import evaluate_model
from neural_recommendation.popularity_diagnostic import analyze_recommendation_popularity
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH =(PROJECT_ROOT / 'data' / 'processed' / 'movie_data' /
    'train_ratings.csv')

VAL_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "movie_data" / "val_ratings.csv")

def prepare_ncf_data(train_df : pd.DataFrame):
    user_dict = {}
    movie_dict = {}

    user_to_idx = train_df['user_id'].unique()
    movie_to_idx = train_df['movie_id'].unique()
    #grab all the unique user/movie ids

    for index, id in enumerate(user_to_idx):
        user_dict[id] = index
    for index, id in enumerate(movie_to_idx):
        movie_dict[id] = index
    #fill a dictionary with the id numbers as keys, index position as values
    #this is so we can extract the ids later on in the embedding tables

    user_indices = train_df['user_id'].map(user_dict)
    #embedding index for users

    movie_indices = train_df['movie_id'].map(movie_dict)
    #embedding index for movies

    labels = train_df['is_positive']
    #grab the movie ratings from the training dataframe

    user_tensor = torch.tensor(user_indices.to_numpy(), dtype=torch.long)
    movie_tensor = torch.tensor(movie_indices.to_numpy(), dtype=torch.long)
    labels_tensor = torch.tensor(labels.to_numpy(), dtype=torch.float32)

    dataset = TensorDataset(user_tensor, movie_tensor, labels_tensor)
    #create a tensor dataset from our user/movie/rating tensors

    train_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True)
    #utilize dataloader to organize our tensor dataset into batches with 256 observations per batch

    return train_loader, user_dict, movie_dict

def prepare_ncf_validation(val_df, user_dict, movie_dict):
    initial_count = len(val_df)

    val_df = val_df[
        (val_df['user_id'].isin(user_dict)) &
        (val_df['movie_id'].isin(movie_dict))].copy()
    #create a copy of the validation df of only rows that have movie and user ids common with the training df

    filtered_count = len(val_df)
    print(f'Filtered rows: {initial_count - filtered_count}')

    user_indices = val_df['user_id'].map(user_dict)
    movie_indices = val_df['movie_id'].map(movie_dict)
    positives = val_df['is_positive']

    user_tensor = torch.tensor(user_indices.to_numpy(), dtype=torch.long)
    movie_tensor = torch.tensor(movie_indices.to_numpy(), dtype=torch.long)
    label_tensor = torch.tensor(positives.to_numpy(), dtype=torch.float32)

    dataset = TensorDataset(user_tensor, movie_tensor, label_tensor)
    #create a tensor dataset from our user/movie/labels tensors

    val_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False)
    return val_loader

def prepare_bpr_data(train_df: pd.DataFrame,user_dict: dict,movie_dict: dict,batch_size: int = 256,seed: int = 42):
    rng = np.random.default_rng(seed)

    #all movies with embeddings learned from training
    all_movies = set(movie_dict.keys())

    #map each user to every movie they rated in training
    user_histories = get_user_history(train_df)

    #keep only positively rated interactions
    positive_interactions = train_df[
        train_df["is_positive"] == 1
    ][["user_id", "movie_id"]]

    #precompute unseen movies
    unseen_by_user = {}

    for user_id, rated_movies in user_histories.items():
        unseen_by_user[user_id] = np.array(
            sorted(all_movies - rated_movies),
            dtype=np.int64)

    #store the indices for the tuple
    user_indices = []
    positive_indices = []
    negative_indices = []

    for user_id, positive_movie_id in positive_interactions.itertuples(
        index=False,
        name=None):
        unseen_movies = unseen_by_user[user_id]

        if len(unseen_movies) == 0:
            continue

        negative_movie_id = int(rng.choice(unseen_movies))

        user_indices.append(user_dict[user_id])
        positive_indices.append(movie_dict[positive_movie_id])
        negative_indices.append(movie_dict[negative_movie_id])

    #convert lists into PyTorch tensors
    user_tensor = torch.tensor(user_indices, dtype=torch.long)
    positive_tensor = torch.tensor(positive_indices, dtype=torch.long)
    negative_tensor = torch.tensor(negative_indices, dtype=torch.long)

    dataset = TensorDataset(
        user_tensor,
        positive_tensor,
        negative_tensor)

    bpr_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)

    return bpr_loader

def prepare_bpr_sampling_data(train_df, user_dict, movie_dict):
    """Prepare reusable positive interactions and negative candidate pools."""

    all_movies = set(movie_dict.keys())
    user_histories = get_user_history(train_df)

    positive_df = train_df.loc[
        train_df["is_positive"] == 1,
        ["user_id", "movie_id"]
    ]

    positive_by_user = {}
    unseen_by_user = {}

    for user_id, user_ratings in positive_df.groupby("user_id"):

        positive_by_user[user_id] = np.array(
            [movie_dict[m] for m in user_ratings["movie_id"]],
            dtype=np.int64
        )

        unseen_by_user[user_id] = np.array(
            [movie_dict[m] for m in sorted(
                all_movies - user_histories[user_id]
            )],
            dtype=np.int64
        )

    return positive_by_user, unseen_by_user

def sample_bpr_epoch(
    positive_by_user,
    unseen_by_user,
    user_dict,
    rng,
    batch_size=256
):
    """Draw fresh negative examples and construct one epoch's DataLoader."""

    users = []
    positives = []
    negatives = []

    for user_id, positive_movies in positive_by_user.items():

        unseen_movies = unseen_by_user[user_id]

        if len(unseen_movies) == 0:
            continue

        negative_movies = rng.choice(
            unseen_movies,
            size=len(positive_movies),
            replace=True
        )

        users.extend([user_dict[user_id]] * len(positive_movies))
        positives.extend(positive_movies)
        negatives.extend(negative_movies)

    if not users:
        raise ValueError("No positive training interactions have an unobserved negative candidate.")

    dataset = TensorDataset(
        torch.tensor(users, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=16):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        self.fc1 = nn.Linear((embedding_dim * 2),64)
        #our input number is the concatenation of the user and movie embeddings
        #takes our 32 concatenated features and passes them through 64 hidden neurons
        self.fc2 = nn.Linear(64, 32)
        #hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(32, 1)
        #hidden layer 2 to output layer


    def forward(self, user_indices, movie_indices): #forward propagation portion
        user_vectors = self.user_embeddings(user_indices)
        movie_vectors = self.movie_embeddings(movie_indices)

        vectors_concat = torch.cat((user_vectors, movie_vectors), dim=1)

        fc1 = self.fc1(vectors_concat)
        hidden1 = torch.relu(fc1)

        fc2 = self.fc2(hidden1)
        hidden2 = torch.relu(fc2)

        fc3 = self.fc3(hidden2).squeeze(-1)

        return fc3

def train_ncf(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    loss = nn.BCEWithLogitsLoss()
    adam = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for user_batch, movie_batch, label_batch in train_loader:
            predictions = model(user_batch, movie_batch)

            batch_loss = loss(predictions, label_batch)

            adam.zero_grad()
            #clean the gradients from the previous batches

            batch_loss.backward()
            #use backwards propagation to calculate gradients for the current epoch
            adam.step()
            #call Adam to use the historical gradients to update model parameters

            total_loss += batch_loss.item() * label_batch.size(0)
        average_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_total_loss = 0.0

        with torch.no_grad():
            for user_batch, movie_batch, label_batch in val_loader:
                predictions = model(user_batch, movie_batch)
                batch_loss = loss(predictions, label_batch)

                val_total_loss += batch_loss.item() * label_batch.size(0)
        average_val_loss = val_total_loss / len(val_loader.dataset)

        if average_val_loss < best_val_loss:
            #detect when the model begins to overfit
            best_val_loss = average_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            #make a copy of the model at the epoch where the model starts overfitting


        print(f'Epoch: {epoch+1} / {epochs}. Average Train loss: {average_loss}')
        print(f'Epoch: {epoch+1} / {epochs}. Average Val loss: {average_val_loss}')
    model.load_state_dict(best_model_state)
    #return the model using the parameters of the best epoch
    return model

import copy

def train_bpr(
    model, epoch_loader, user_dict, val_df, user_histories, movie_dict, movie_counts,
    epochs=10, learning_rate=0.001, min_ratings=50
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_ndcg = -1.0
    best_epoch = 0
    best_model_state = None

    eligible_users = sorted(
        set(val_df.loc[val_df["is_positive"] == 1, "user_id"])
        & set(user_histories))

    val_rng = np.random.default_rng(42)
    selected_users = val_rng.choice(eligible_users, size=min(200, len(eligible_users)), replace=False)

    sampled_histories = {
        user_id: user_histories[user_id]
        for user_id in selected_users}

    for epoch in range(epochs):
        bpr_loader = epoch_loader()

        model.train()
        total_loss = 0.0

        for user_batch, positive_batch, negative_batch in bpr_loader:

            positive_scores = model(user_batch, positive_batch)
            negative_scores = model(user_batch, negative_batch)

            batch_loss = torch.nn.functional.softplus(
                negative_scores - positive_scores).mean()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item() * user_batch.size(0)

        average_loss = total_loss / len(bpr_loader.dataset)

        def bpr_recommend(user_id, user_history, k):
            return recommend_ncf(model=model,user_id=user_id,user_dict=user_dict,movie_dict=movie_dict,
                user_history=user_history,
                movie_counts=movie_counts,
                k=k,
                min_ratings=min_ratings)

        results = evaluate_model(bpr_recommend, sampled_histories, val_df, k=10)

        val_ndcg = results["ndcg_at_k"]

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train BPR: {average_loss:.4f} | "
            f"Val NDCG@10: {val_ndcg:.4f}")

    model.load_state_dict(best_model_state)

    print(f"Best checkpoint: Epoch {best_epoch} | NDCG@10: {best_ndcg:.4f}")

    return model

def recommend_ncf(model: NeuralCollaborativeFiltering,
    user_id: int,
    user_dict: dict,
    movie_dict: dict,
    user_history: set,
    movie_counts : pd.Series,
    k: int = 10, min_ratings: int = 0):
    user_idx = user_dict[user_id]

    candidate_movies = []
    for movie_id in movie_dict:
        if movie_id not in user_history and movie_counts[movie_id] >= min_ratings:
            candidate_movies.append(movie_id)

    candidate_indices = []
    for movie in candidate_movies:
        candidate_indices.append(movie_dict[movie])

    user_indices = [user_idx] * len(candidate_indices)

    movie_tensor = torch.tensor(candidate_indices, dtype=torch.long)
    user_tensor = torch.tensor(user_indices, dtype=torch.long)

    top_movies = []

    model.eval()
    with torch.no_grad():
        predictions =  model(user_tensor, movie_tensor)
        positions = torch.argsort(predictions, descending=True)

        for idx in positions[:k]:
            top_movies.append(int(candidate_movies[idx.item()]))

    return top_movies


if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    movie_counts = train_df.groupby("movie_id").size()

    train_loader, user_dict, movie_dict = prepare_ncf_data(train_df)

    user_histories = get_user_history(train_df)

    positive_by_user, unseen_by_user = prepare_bpr_sampling_data(
        train_df,
        user_dict,
        movie_dict)
    rng = np.random.default_rng(42)

    def epoch_loader():
        return sample_bpr_epoch(
            positive_by_user=positive_by_user,
            unseen_by_user=unseen_by_user,
            user_dict=user_dict,
            rng=rng,
            batch_size=256
        )

    torch.manual_seed(42)

    bpr_model = NeuralCollaborativeFiltering(
        num_users=len(user_dict),
        num_movies=len(movie_dict),
        embedding_dim=32)
    
    trained_bpr_model = train_bpr(
        model=bpr_model,
        epoch_loader=epoch_loader,
        user_dict=user_dict,
        val_df=val_df,
        user_histories=user_histories,
        movie_dict=movie_dict,
        movie_counts=movie_counts,
        epochs=10,
        learning_rate=0.001,
        min_ratings=50
)

    def recommend(user_id, user_history, k):
        return recommend_ncf(
            model=trained_bpr_model,
            user_id=user_id,
            user_dict=user_dict,
            movie_dict=movie_dict,
            user_history=user_history,
            movie_counts=movie_counts,
            k=k,
            min_ratings=50)

    results = evaluate_model(
        recommend,
        user_histories,
        val_df,
        k=10)

    print("Dynamic BPR validation results:", results)

    eligible_users = sorted(
        set(val_df.loc[val_df["is_positive"] == 1, "user_id"])
        & set(user_histories))

    diagnostic_results = analyze_recommendation_popularity(
        recommend,
        user_histories,
        train_df,
        eligible_users,
        k=10)

    print("Dynamic BPR recommendation popularity:", diagnostic_results)

#$env:PYTHONPATH = "src"
#python -m neural_recommendation.neural_cf

