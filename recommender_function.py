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
import streamlit as st
import os

EMBEDDING_SIZE = 50
class RecommenderNet(keras.Model):
    def _init_(self, num_users=610, num_movies=9724, embedding_size=50, **kwargs):
        super()._init_(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


model = RecommenderNet(610, 9724, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
)

def get_recommendations(movies_data,df,user_id, k):

      #current_path = os.getcwd()
      #print("Current path:", current_path)
      #st.write("Current path:", current_path)
      
    #folder_path = 'Col'  # Specify the folder path

    # Iterate over each file in the folder
    #for file_name in os.listdir(folder_path):
    #    # Check if the path is a file (not a subdirectory)
    #    if os.path.isfile(os.path.join(folder_path, file_name)):
    #        st.write(file_name)
      
    #new_model = RecommenderNet(keras.Model)
    options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    #new_model = tf.keras.models.load_model('Col', options=options)  
    model=keras.models.load_model('Col/saved_model.pb')
    #imported = tf.saved_model.load(path)
    model = model.signatures["keras_metadata.pb"]



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