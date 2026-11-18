import pandas as pd
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from neural_recommendation.popularity_diagnostic import analyze_recommendation_popularity


### GOAL : PREDICT MOVIES A USER MIGHT LIKE BASED ON THEIR HISTORICAL RATINGS USING MATRIX FACTORIZATION
# This method uses other observations from the user to estimate their preferences for unobserved movies


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "movie_data" / "train_ratings.csv")

VAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "movie_data" / "val_ratings.csv")

#establish the data paths


def prepare_data(train_df:pd.DataFrame):

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
    #membedding index for movies

    ratings = train_df['rating']
    #grab the movie ratings from the training dataframe

    user_tensor = torch.tensor(user_indices.to_numpy(), dtype=torch.long)
    movie_tensor = torch.tensor(movie_indices.to_numpy(), dtype=torch.long)
    rating_tensor = torch.tensor(ratings.to_numpy(), dtype=torch.float32)
    #convert each pandas series into a tensor

    dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
    #create a tensor dataset from our user/movie/rating tensors

    train_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True)
    #utilize dataloader to organize our tensor dataset into batches with 256 observations per batch

    return train_loader, user_dict, movie_dict

def prepare_validation_data(val_df, user_dict, movie_dict):
    initial_count = len(val_df)

    val_df = val_df[
    (val_df['user_id'].isin(user_dict)) &
    (val_df['movie_id'].isin(movie_dict))].copy()
    #create a copy of the validation df of only rows that have movie and user ids common with the training df

    filtered_count = len(val_df)
    print(f'Filtered rows: {initial_count - filtered_count}')

    user_indices = val_df['user_id'].map(user_dict)
    movie_indices = val_df['movie_id'].map(movie_dict)
    ratings = val_df['rating']

    user_tensor = torch.tensor(user_indices.to_numpy(), dtype=torch.long)
    movie_tensor = torch.tensor(movie_indices.to_numpy(), dtype=torch.long)
    rating_tensor = torch.tensor(ratings.to_numpy(), dtype=torch.float32)

    dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
        #create a tensor dataset from our user/movie/rating tensors
    
    val_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False)
    return val_loader



class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=16, global_mean=0.0):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.movie_embeddings.weight, mean=0.0, std=0.05)

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

        self.global_mean = global_mean


    def forward(self, user_indices, movie_indices):
        user_vectors = self.user_embeddings(user_indices)
        movie_vectors = self.movie_embeddings(movie_indices)
        #these create embedding layers, which is basically just a table 

        user_biases = self.user_bias(user_indices).squeeze(-1)
        movie_biases = self.movie_bias(movie_indices).squeeze(-1)
        #remove the 2nd dimension from the embedding table, so that our bias matches our dot product dimension


        predictions = ((user_vectors * movie_vectors).sum(dim=1) + user_biases + movie_biases + self.global_mean)
        #dot product for each user-movie pair

        return predictions
        

def train_model( model: MatrixFactorization,train_loader: DataLoader, val_loader: DataLoader,epochs: int = 10,
                learning_rate: float = 0.001):
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for user_batch, movie_batch, rating_batch in train_loader:
            predictions = model(user_batch, movie_batch)

            loss = mse_loss(rating_batch, predictions)
            #get the meansquared error for said batch

            optimizer.zero_grad()
            #clean the gradients from the previous batches

            loss.backward()
            #use backwards propagation to calculate gradients for the current epoch
            optimizer.step()
            #call Adam to use the historical gradients to update model parameters

            total_loss += loss.item() * rating_batch.size(0)

        train_average_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_total_loss = 0.0
        
        with torch.no_grad():
            for user_batch, movie_batch, rating_batch in val_loader:
                predictions = model(user_batch, movie_batch)
                    
                loss = mse_loss(rating_batch, predictions)
                val_total_loss += loss.item() * rating_batch.size(0)
            val_average_loss = val_total_loss / len(val_loader.dataset)

        print(f'{epoch + 1}/{epochs}: TRAIN MSE = {train_average_loss} | VAL MSE = {val_average_loss}')

    return model

def recommend_movies(model: MatrixFactorization,
    user_id: int,
    user_to_idx: dict,
    movie_to_idx: dict,
    user_history: set,k: int = 10):


    user_idx = user_to_idx[user_id]

    candidate_movies = []
    candidate_indices = []

    for movie in movie_to_idx:
        if movie not in user_history:
            candidate_movies.append(movie)

    for movie in candidate_movies:
            candidate_indices.append(movie_to_idx[movie])

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

from neural_recommendation.popularity import get_user_history
from neural_recommendation.evaluation import evaluate_model

if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    train_loader, user_dict, movie_dict = prepare_data(train_df)
    val_loader = prepare_validation_data(val_df, user_dict, movie_dict)

    model = MatrixFactorization(
    len(user_dict),
    len(movie_dict),
    embedding_dim=16,
    global_mean=train_df['rating'].mean())

    trained_model = train_model(model, train_loader, val_loader)

    user_histories = get_user_history(train_df)

    recommendations = recommend_movies(model, 1, user_dict, movie_dict, user_histories[1])

    def mf_recommend(user_id, user_history, k):
    #call recommend_movies using the trained model
    #and our existing movie/user dictionaries
        recommendations1 = recommend_movies(trained_model, user_id, user_dict, movie_dict, user_history, k)
        return recommendations1

    results = evaluate_model(mf_recommend, user_histories, val_df, k=10)
    print(results)
        # Get the same validation users used for evaluation
    eligible_user_ids = (
        val_df.loc[val_df["is_positive"] == 1, "user_id"]
        .unique()
    )

    # Analyze how popular the recommended movies are
    diagnostic_results = analyze_recommendation_popularity(
        recommend_fn=mf_recommend,
        user_histories=user_histories,
        train_df=train_df,
        user_ids=eligible_user_ids,
        k=10
    )

    print("Matrix Factorization Recommendation Popularity:")
    print(diagnostic_results)

