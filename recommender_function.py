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


def get_recommendations(movies_data,df,user_id, k):
  options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
  model = tf.keras.models.load_model('Col', options=options)  
  #model=keras.models.load_model('Col')

  user_ids = df["userId"].unique().tolist()
  user2user_encoded = {x: i for i, x in enumerate(user_ids)}
  userencoded2user = {i: x for i, x in enumerate(user_ids)}


  movie_ids = df["movieId"].unique().tolist()
  movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
  movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}


  movies_watched_by_user = df[df.userId == user_id]
  movies_not_watched = movies_data[
      ~movies_data["movieId"].isin(movies_watched_by_user.movieId.values)
  ]["movieId"]
  movies_not_watched = list(
      set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
  )
  movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
  user_encoder = user2user_encoded.get(user_id)
  user_movie_array = np.hstack(
      ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
  )
  ratings = model.predict(user_movie_array).flatten()
  top_ratings_indices = ratings.argsort()[-k:][::-1]
  recommended_movie_ids = [
      movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
  ]

  recommended_movies = movies_data[movies_data["movieId"].isin(recommended_movie_ids)]
  recommended_movies = recommended_movies.merge(
      pd.DataFrame({"movieId": recommended_movie_ids, "rating": ratings[top_ratings_indices]}),
      on="movieId",
  )
  recommended_movies = recommended_movies.sort_values("rating", ascending=False).reset_index(drop=True)
  
  return recommended_movies[["title", "genres"]].head(k)