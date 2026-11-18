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

def train_ncf(model, train_loader, epochs=10, learning_rate=0.001):
    loss = nn.BCEWithLogitsLoss()
    adam = optim.Adam(model.parameters(), lr=learning_rate)

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
        print(f'Epoch: {epoch+1} / {epochs}. Average loss: {average_loss}')
    return model

def recommend_ncf(model: NeuralCollaborativeFiltering,
    user_id: int,
    user_dict: dict,
    movie_dict: dict,
    user_history: set,
    k: int = 10):
    user_idx = user_dict[user_id]

    candidate_movies = []
    for id in movie_dict:
        if id not in user_history:
            candidate_movies.append(id)

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
     
    train_loader, user_dict, movie_dict = prepare_ncf_data(train_df)
    val_loader = prepare_ncf_validation(val_df, user_dict, movie_dict)

    model = NeuralCollaborativeFiltering(num_users=len(user_dict),
    num_movies=len(movie_dict), embedding_dim=16)

    user_histories = get_user_history(train_df)

    trained_model = train_ncf(model,
    train_loader,
    epochs=10,
    learning_rate=0.001)

    def ncf_recommend(user_id, user_history, k):
        recommendations = recommend_ncf(model=trained_model, user_id=user_id,user_dict=user_dict,
                      movie_dict=movie_dict, user_history=user_history, k=k)

        return recommendations

    results = evaluate_model(ncf_recommend, user_histories, val_df, k=10)

    print(results)

    eligible_user_ids = (
    val_df.loc[val_df["is_positive"] == 1, "user_id"]
    .unique())

    diagnostic_results = analyze_recommendation_popularity(
        recommend_fn=ncf_recommend,
        user_histories=user_histories,
        train_df=train_df,
        user_ids=eligible_user_ids,
        k=10)

    print("NCF recommendation popularity:")
    print(diagnostic_results)
        


    