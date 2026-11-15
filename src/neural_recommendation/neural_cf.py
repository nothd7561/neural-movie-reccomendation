import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

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
    #membedding index for movies
    
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

class NeuralCollaborativeFiltering(nn.Module):
    