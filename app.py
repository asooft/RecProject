import streamlit as st
import os
import pandas as pd


st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='Recommender Movies Dashboard')

        
st.title('Recommender Movies Dashboard')


st.cache_data()
ratings=pd.read_csv("Dataset/ratings.csv")
movies=pd.read_csv("Dataset/movies.csv")
df = pd.merge(ratings, movies, on='movieId', how='left')
first_100_users = df['userId'].unique()[:100]
df = df[df['userId'].isin(first_100_users)]
genres_unique = pd.DataFrame(df.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genre']) # Format into DataFrame to store later
df = df.join(df.genres.str.get_dummies().astype(int))
df.drop('genres', inplace=True, axis=1)
genres_unique = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX', 'Western', 'Film-Noir', '(no genres listed)']
cats_ohe = pd.get_dummies(df[genres_unique])
null_counts = ratings.isnull().sum()
null_percentages = (null_counts / len(ratings)) * 100
null_summary = pd.DataFrame({'Null Count': null_counts, 'Null Percentage': null_percentages})
null_counts = movies.isnull().sum()
null_percentages = (null_counts / len(ratings)) * 100
null_summary = pd.DataFrame({'Null Count': null_counts, 'Null Percentage': null_percentages})
unique_items = ratings['movieId'].nunique()
unique_users = ratings['userId'].nunique()
unique_items = movies['movieId'].nunique()
user_mappings = df['userId'].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'new_id'}).set_index('userId')

# Assuming you have the user_mappings DataFrame available

# Get the unique values from the userId column
user_ids = user_mappings.index.unique()

# Create a dictionary to map userId to new_id
mapping_dict = user_mappings['new_id'].to_dict()

# Create the dropdown menu
selected_user_id = st.selectbox('Select a userId:', user_ids)

# Get the corresponding new_id using the mapping_dict
selected_new_id = mapping_dict[selected_user_id]

# Create a DataFrame with the selected mapping
mapping_df = pd.DataFrame({'userId': [selected_user_id], 'new_id': [selected_new_id]})

# Display the mapping DataFrame as a table without the index column
st.write(mapping_df)

# Display a name for the input field and get the numeric input
value = st.number_input("Enter a Top Number of movies you want")

# Use the entered number
st.write("Top Number of movies you want:", value)
