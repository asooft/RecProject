import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model
from keras.utils.vis_utils import plot_model
from scipy.sparse import vstack
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
import pickle

def get_similar_movies(ratings,movies,movie_name, top_k):
  with open(r"model_knn.pkl", 'rb') as file:
    model_knn = pickle.load(file)
  utility_matrix = ratings.pivot(index="movieId",columns="userId",values="rating")
  utility_matrix = utility_matrix.fillna(0)

  movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
  query_index_movie_ratings = utility_matrix.loc[movie_id, :].values.reshape(1, -1)
  distances, indices = model_knn.kneighbors(query_index_movie_ratings, n_neighbors=top_k+1)

  similar_movies = []
  for i in range(0, len(distances.flatten())):
      if i == 0:
          continue  # Skip the first item, as it is the query movie itself
      indices_flat = indices.flatten()[i]
      similar_movie_id = utility_matrix.iloc[indices_flat, :].name
      similar_movie_row = movies[movies['movieId'] == similar_movie_id]
      similar_movie_name = similar_movie_row['title'].values[0]
      similar_movie_genre = similar_movie_row['genres'].values[0]
      similar_movies.append({'Movie': similar_movie_name, 'Genre': similar_movie_genre})

  similar_movies_df = pd.DataFrame(similar_movies)
  return similar_movies_df