import streamlit as st
import os
import pandas as pd

# Set the theme to dark by default
st.set_theme('dark')

# Create a button in the sidebar
theme_switcher = st.sidebar.button("Toggle Theme")

# Check if the button is clicked
if theme_switcher:
    # Toggle between dark and default (white) theme
    if st.get_theme() == 'dark':
        st.set_theme('default')
    else:
        st.set_theme('dark')
        
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.title('Recommender Movies Dashboard')
st.set_page_config(page_title='Recommender Movies Dashboard')

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

# Get the unique values from the new_id column
new_ids = user_mappings['new_id'].unique()

# Create the dropdown menu
selected_new_id = st.selectbox('Select Used Id:', new_ids)

# Filter the DataFrame based on the selected new_id
filtered_df = user_mappings[user_mappings['new_id'] == selected_new_id]

# Display the filtered DataFrame
st.write(filtered_df)
