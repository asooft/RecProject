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
value = st.number_input("Enter a Top Number of movies you want", step=1.0, format="%d")

# Convert the input value to an integer
integer_value = int(value)

# Use the entered integer
st.write("Top Number of movies you want:", integer_value)


import pickle
file_path = "fm_model.pkl"  # Change the path as per your preference
with open(file_path, 'rb') as f:
    model = pickle.load(f)
    
 # Save selected_new_id in a variable
new_id_variable = selected_new_id
    
new=user_mappings.loc[new_id_variable].new_id
features,ratings = dataset_test[[new]]


model.eval()
with torch.no_grad():
  st.write(f'Predicted rating for User of interest: {model(features).item()}') # Get the model output on the user of interest after running the previous cell to now their new_id
  st.write(f'Actual Rating: {ratings.values[0]}') # Extract the actual rating for the user of interest from dataset_test Dataset object

# Replace None with the new_id of the user
items_our_user_rated = (train[train.userId==new].movieId).unique().tolist()
items_our_user_rated.extend((test[test.userId==new].movieId).unique().tolist())

items_our_user_can_rate = movie_mappings[~movie_mappings.new_id.isin(items_our_user_rated)].new_id.tolist()

st.write(f'Number of unique items user of interest rated is {len(items_our_user_rated)}')
st.write(f'Number of unique items that can be recommended to user of interest is {len(items_our_user_can_rate)}')
st.write(f'Preview of the item list:\n\t{items_our_user_can_rate[:integer_value]}')

N = 5  # Number of recommendations

recommendations = []

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for item_id in items_our_user_can_rate:
        features = dataset_test[[item_id]][0]  # Create a dataset for the item
        #print(features)
        # Check if the dataset for the item is empty
        if features.nelement() == 0:
            continue

        predicted_rating = model(features).item()  # Get the predicted rating for the user

        recommendations.append((item_id, predicted_rating))

# Sort the recommendations based on predicted ratings in descending order
recommendations.sort(key=lambda x: x[1], reverse=True)

# Select the top N recommendations
top_recommendations = recommendations[:N]

# Print the top recommendations
st.write("Top Recommendations:")
for item_id, predicted_rating in top_recommendations:
    movie_title = movie_mappings[movie_mappings.new_id == item_id].index[0]
    st.write(f"Movie: {movie_title}, Predicted Rating: {predicted_rating}")
    
    # Print the top recommendations with movie names
print("Top Recommendations:")
for item_id, predicted_rating in top_recommendations:
    movie_title = movies[movies['movieId'] == item_id]['title'].values[0]
    st.write(f"Movie: {movie_title}, Predicted Rating: {predicted_rating}")

