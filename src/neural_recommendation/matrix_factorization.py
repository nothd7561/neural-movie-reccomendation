import pandas as pd
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


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

    user_indices = train_df['user_id'].map(user_to_idx)
    #embedding index for users

    movie_indices = train_df['movie_id'].map(movie_to_idx)
    #membedding index for movies

    ratings = train_df['ratings']
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
    #utilize dataloader to organize our tensor dataset into 256 batches

    return train_loader, user_to_idx, movie_to_idx


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=16, global_mean=0.0):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        self.global_mean = global_mean


    def forward(self, user_indices, movie_indices):
        user_vectors = self.user_embeddings(user_indices)
        movie_vectors = self.movie_embeddings(movie_indices)
        #these create embedding layers, which is basically just a table 

        user_biases = self.user_bias(user_indices).squeeze(-1)
        movie_biases = self.movie_bias(user_indices).squeeze(-1)


        predictions = ((user_vectors * movie_vectors).sum(dim=1) + user_biases + movie_biases + self.global_mean)
        #dot product for each user-movie pair

        return predictions
        

def train_model():


def recommend_movies():