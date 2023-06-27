import streamlit as st
import pandas as pd
import plost
import tempfile
import os
import subprocess
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model
from keras.utils.vis_utils import plot_model
from scipy.sparse import vstack
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
import recommender_function
import recommender_function2

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Here you can add code to fetch and display the recommended movies based on the selected genre and user preferences.

# Additional features
# You can add more interactivity and components to your dashboard, such as sliders, checkboxes, buttons, etc.


import streamlit as st

ratings_data = pd.read_csv('Dataset/ratings.csv')
movies_data = pd.read_csv('Dataset/movies.csv')
tags_data=pd.read_csv('Dataset/tags.csv')
links_data=pd.read_csv('Dataset/links.csv')

df=pd.read_csv('data.csv')
raw_data=pd.read_csv("raw_data.csv")


# Get a list of unique users
users = ratings_data['userId'].unique()
movies=movies_data['title'].unique()

# Title
st.title("Recommender System Dashboard")
# Sidebar
st.sidebar.title("User Preferences")
selected_page = st.sidebar.radio("Navigation", ["Movies", "Users"])

# Items page
if selected_page == "Movies":
  st.header("Movies Page")
  # Add your code and components specific to the items page here
  movie_selected_view = st.sidebar.radio("View", ["Movie Details", "Movie similarities"])

  if movie_selected_view == "Movie Details":
    st.subheader("Movie details")
    selected_movie = st.selectbox("Select a movie", movies)
    st.write(f"Selected moveis: {selected_movie}")

    selected_movie_data = movies_data[movies_data['title'] == selected_movie]

    # Filter links_data based on the selected movie
    selected_link_data = links_data[links_data['movieId'] == selected_movie_data['movieId'].values[0]]
#mean
    mean_ratings = ratings_data.groupby('movieId')['rating'].mean()
    selected_movie_id = movies_data[movies_data['title'] == selected_movie]['movieId'].values[0]
    selected_movie_ratings = ratings_data[ratings_data['movieId'] == selected_movie_id]


    # Display the complete information about the selected movie
    if not selected_movie_data.empty and not selected_link_data.empty:
        st.write(f"**Movie: {selected_movie}**")
        st.write(f"**Movie ID:** {selected_movie_data['movieId'].values[0]}")
        st.write(f"**Genres:** {selected_movie_data['genres'].values[0]}")
        st.write(f"**IMDb ID:** {selected_link_data['imdbId'].values[0]}")
        if selected_movie_id in mean_ratings:
          mean_rating = mean_ratings[selected_movie_id]
          st.write(f"**Average Rating:** {mean_rating:.2f}")
        else:
          st.write("No ratings found for the selected movie.")

    else:
        st.write("No movie selected.")








  elif movie_selected_view == "Movie similarities":
    st.subheader("Movie similarities")
    selected_movie = st.selectbox("Select a movie", movies)
    st.write(f"Selected moveis: {selected_movie}")

    movie_rec = movies_data[movies_data['title'] == selected_movie]

    
    #   # Check if user selection has changed
    if "prev_movie" not in st.session_state or selected_movie != st.session_state.prev_movie:
        # Reset page and items_per_page
        st.session_state.page = 1
        st.session_state.items_per_page = 10


    max_items = st.sidebar.number_input("Top similar movies", min_value=1, value=5, step=1, key="Top_similar_movies_input")
    # total_items = min(len(movie_rec), max_items)
    total_items=max_items

    # Input for items per page
    items_per_page = st.sidebar.number_input("Movies per Page", min_value=1, max_value=total_items, value=5, step=1, key="movies_per_page_input")

    num_pages = int((total_items - 1) / items_per_page) + 1
  
    similar_df=recommender_function2.get_similar_movies(ratings_data,movies_data,selected_movie, max_items)

    page = st.sidebar.number_input("Page", min_value=1, max_value=num_pages, value=st.session_state.page, step=1)
    start_index = (page - 1) * items_per_page
    end_index = min(start_index + items_per_page, total_items)
    sliced_data = similar_df.iloc[start_index:end_index]
    st.sidebar.write(f"Page: {page}/{num_pages}")


    st.write(f"Displaying similarities {start_index + 1} to {end_index} out of {total_items}")

    similar_movies = []
    for index, row in sliced_data.iterrows():
        movie_name = row['Movie']
        movie_genre = row['Genre']
        adjusted_index = index + 1  # Calculate adjusted index
        similar_movies.append([adjusted_index, movie_name, movie_genre])

    df = pd.DataFrame(similar_movies, columns=['Index','Movie Name', 'Movie Genre'])
    df.index += 1  # Modify index to start from 1
    df = df.set_index('Index')
    st.table(df)

# Users page
elif selected_page == "Users":
    st.header("Users Page")
     # Example: Display a list of users
    user_selected_view = st.sidebar.radio("View", ["User History", "User Recommendation"])

#radio #1 history 

    if user_selected_view == "User History":
      st.subheader("User History")
      selected_user = st.selectbox("Select a user", users, key="user_select")
      st.write(f"Selected user: {selected_user}")

      user_data = ratings_data[ratings_data['userId'] == selected_user]

      if not user_data.empty:
          st.subheader("User Interactions")
          st.write(f"User {selected_user} interacted with the following items:")

          # Check if user selection has changed
          if "prev_user" not in st.session_state or selected_user != st.session_state.prev_user:
              # Reset page and items_per_page
              st.session_state.page = 1
              st.session_state.items_per_page = 10

          # Pagination
          items_per_page = st.sidebar.number_input("Items per Page", min_value=1, value=st.session_state.items_per_page, step=1)
          num_items = len(user_data)
          num_pages = int((num_items - 1) / items_per_page) + 1

          page = st.sidebar.number_input("Page", min_value=1, max_value=num_pages, value=st.session_state.page, step=1)
          start_index = (page - 1) * items_per_page
          end_index = min(start_index + items_per_page, num_items)
          sliced_data = user_data.iloc[start_index:end_index]

          st.write(f"Displaying interactions {start_index + 1} to {end_index} out of {num_items}")

          # Display interactions for the selected page
          table_data = []
          for index, row in sliced_data.iterrows():
              movie_id = row['movieId']
              rating = row['rating']
              movie_name = movies_data[movies_data['movieId'] == movie_id]['title'].values[0]
              movie_genre = movies_data[movies_data['movieId'] == movie_id]['genres'].values[0]
              table_data.append({
                  "Movie ID": int(movie_id),
                  "Movie Name": movie_name,
                  "Movie Genre": movie_genre,
                  "Rating": rating
              })

          # Create a DataFrame from the list of dictionaries
          table_df = pd.DataFrame(table_data)
          table_df.index += 1  # Modify index to start from 1
          st.table(table_df)

          st.sidebar.write(f"Page: {page}/{num_pages}")

          
      else:
          st.write("No interactions found for the selected user.")




#radio #2 recommendation 
    elif user_selected_view == "User Recommendation":
            st.subheader("User Recommendation")
            selected_user = st.selectbox("Select a user", users)
            st.write(f"Selected user: {selected_user}")


            user_rec = ratings_data[ratings_data['userId'] == selected_user]#from model
              # Check if user selection has changed
            if "prev_user" not in st.session_state or selected_user != st.session_state.prev_user:
                # Reset page and items_per_page
                st.session_state.page = 1
                st.session_state.items_per_page = 10


            max_items = st.sidebar.number_input("Top Recommended Items", min_value=1, value=5, step=1, key="Top_recommended_items_input")
            total_items = min(len(user_rec), max_items)

            # Input for items per page
            items_per_page = st.sidebar.number_input("Items per Page", min_value=1, max_value=total_items, value=5, step=1, key="items_per_page_input")

            num_pages = int((total_items - 1) / items_per_page) + 1
            
            recommended_df=recommender_function.get_recommendations(movies_data,df,selected_user, max_items)

            page = st.sidebar.number_input("Page", min_value=1, max_value=num_pages, value=st.session_state.page, step=1)
            start_index = (page - 1) * items_per_page
            end_index = min(start_index + items_per_page, total_items)
            sliced_data = recommended_df.iloc[start_index:end_index]
            st.sidebar.write(f"Page: {page}/{num_pages}")


            st.write(f"Displaying recommendations {start_index + 1} to {end_index} out of {total_items}")

            recommended_movies = []
            for index, row in sliced_data.iterrows():
                movie_name = row['title']
                movie_genre = row['genres']
                adjusted_index = index + 1  # Calculate adjusted index
                recommended_movies.append([adjusted_index, movie_name, movie_genre])



            df = pd.DataFrame(recommended_movies, columns=['Index','Movie Name', 'Movie Genre'])
            df.index += 1  # Modify index to start from 1
            df = df.set_index('Index')
            st.table(df)
              

genre = st.sidebar.selectbox("Select Genre", ["Action", "Comedy", "Drama"])

# Main content
st.write(f"You selected {genre} genre.")

   



