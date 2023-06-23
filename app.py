import streamlit as st
import os
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
import pickle
import joblib
from PIL import Image
from surprise import SVD
from surprise import Dataset as SurpriseDataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import KNNBasic,  KNNWithMeans, KNNBaseline
from surprise.model_selection import KFold
from surprise import Reader
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from itertools import product
from IPython.display import display, clear_output
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler


st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='Recommender Movies Dashboard')

        
st.title('Recommender Movies Dashboard')


class DPMovieDataset(Dataset):
      def __init__(self, user_ids, data, agg_hist, active_matrix, recommendation=False):
        self.user_ids = user_ids
        self.data = data
        self.agg_hist = agg_hist
        self.active_matrix = active_matrix
        self.recommendation = recommendation

      def __len__(self):
        return self.user_ids.shape[0]

      def __getitem__(self, idx):
        batch_data = self.data[self.data['userId'].isin(idx)] # Select the rows corresponding to the list of user indices `idx` from self.data dataframe
        cat_cols = batch_data[['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX', 'Western', 'Film-Noir', '(no genres listed)']] # From batch_data extract only the one-hot encoded categorical columns
        agg_history = batch_data[['userId']].merge(self.agg_hist, left_on='userId', right_index=True) # Get the aggregated history for each selected transaction using merge
        active_groups = self.active_matrix[self.active_matrix.index.isin(batch_data.index)] # Select the rows corresponding to the indices of the transactions selected in batch_data

        features = torch.from_numpy(np.hstack((active_groups.values, agg_history.values, cat_cols.values))) # Concatenate the processed columns together horizontally

        if not self.recommendation:
          targets = batch_data['rating']
          return features, targets
        else:
          return features
          
class FactorizationMachine(torch.nn.Module):
      def __init__(self, n, k, bias=False):
        super(FactorizationMachine, self).__init__()
        self.n = n
        self.k = k
        self.linear = torch.nn.Linear(self.n, 1, bias)
        self.V = torch.nn.Parameter(torch.randn(n,k)) # Creating the latent matrix V of size (n X k) and initializing it with random values

      def forward(self, x_batch):
        x_batch = x_batch.float()
        part_1 = torch.matmul(x_batch, self.V).pow(2).sum(1, keepdim=True)  # perform the first part of the interaction term: row-wise-sum((XV)^2)
        part_2 = torch.matmul(x_batch.pow(2), self.V.pow(2)).sum(1, keepdim=True) # perform the second part of the interaction term: row-wise-sum((X)^2 * (V)^2))
        inter_term = (part_1 - part_2)/2 # Put the interaction term parts together (refer to the equations above)
        var_strength = self.linear(x_batch) # Perform the linear part of the model equation (refer to the demo notebook on how to use layers in pytorch models)
        return var_strength + inter_term
        
ratings=pd.read_csv("Dataset/ratings.csv")
movies=pd.read_csv("Dataset/movies.csv")
          
def run_user_based():
        
    # Apply custom CSS styles
    st.markdown(
        """
        <style>
        .custom-text5 {
            font-size: 36px; /* Change the font size as desired */
            color: #8D3CC1; /* Change the color as desired */
            font-weight: bold;
        }
        </style>
        """
        , unsafe_allow_html=True
    )

    # Display the text with custom styling
    st.write(
        '<div class="custom-text5">User Based Recommendation</div>',
        unsafe_allow_html=True
    )

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
    # Select a userId using st.selectbox
    selected_user_id = st.selectbox('Select a user Id:', user_ids)

    # Get the corresponding new_id using the mapping_dict
    selected_new_id = mapping_dict[selected_user_id]

    # Create a DataFrame with the selected mapping
    mapping_df = pd.DataFrame({'userId': [selected_user_id], 'new_id': [selected_new_id]})

    # Display the mapping DataFrame as a table without the index column
    #st.write(mapping_df)

    # Display a name for the input field and get the numeric input
    valueS = st.number_input("Enter top number of similar movies you want:", value=5, step=1, format="%d")

    # Convert the input value to an integer
    integer_value = int(valueS)
    
        # Create a container with a specified width and height
    with st.container():
        # Set the container's style to display as a rectangle with a border and padding
        st.markdown(
            """
            <style>
            .custom-box2 {
                background-color: #f8f8f8;
                border: 2px solid #4e6bff;
                border-radius: 10px;
                padding: 15px;
                width: 40%;
            }

            .custom-title2 {
                font-size: 24px;
                font-weight: bold;
                color: #4e6bff;
                margin-bottom: 10px;
            }
            </style>
            """
        , unsafe_allow_html=True)

        # Display the "Top Recommendations" title
        st.write(
            '<div class="custom-box2">'
            '<div class="custom-title2">Top Recommendations Movies</div>'
            '</div>'
            , unsafe_allow_html=True
        )

    # Use the entered integer
    #st.write("Top Number of movies you want:", integer_value)

    #integer_value = 5
    movie_mappings = df['movieId'].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'new_id'}).set_index('movieId')

    df_copy = df.copy() # To avoid changing the original DataFrame
    df_copy['ones'] = 1

    columns_to_replace = ["userId", "movieId"]  # Specify the columns you want to replace

    df_mapped = df_copy.copy()  # Create a copy of the original dataframe

    # Replace values in the specified columns
    df_mapped[columns_to_replace] = df_mapped[columns_to_replace].replace({
        "userId": user_mappings.new_id.to_dict(),
        "movieId": movie_mappings.new_id.to_dict()
    })

    # Replace None with the appropriate values
    agg_history = pd.pivot_table(df_mapped,
                   values='ones', index='userId', columns='movieId', fill_value=0)


    # Replace None with correct values
    # We need to normalize the aggregated history by dividing each value by the sum of all the values in the same row
    agg_history_norm = agg_history / agg_history.values.sum(axis=1, keepdims=True)


    df['Train'] = (ratings.groupby("userId").cumcount(ascending=False) != 0).replace({True:1, False:0})


    final = pd.concat([df_mapped[['userId','movieId']], cats_ohe,df[['Train','rating']]], axis=1)

    train = final[final.Train == 1]
    test = final[final.Train == 0]

    
    active_columns = pd.get_dummies(final[['userId','movieId']].astype(str))
    dataset_train = DPMovieDataset(user_mappings.values, train, agg_history_norm, active_columns)
    dataset_test = DPMovieDataset(user_mappings.values, test, agg_history_norm, active_columns)

    dataloader_train = DataLoader(dataset_train,
                                  sampler=BatchSampler(SequentialSampler(dataset_train), batch_size=10, drop_last=False),
                                  batch_size=None)

    dataloader_test = DataLoader(dataset_test,
                                  sampler=BatchSampler(SequentialSampler(dataset_test), batch_size=10, drop_last=False),
                                  batch_size=None)
                                  
    
        
    features,ratings=dataset_train[[1]]


    model = FactorizationMachine(n=11413, k=integer_value)

    def model_step(mode, x, y=None, optimizer=None, train=True):
      if train: # If we're in training phase, then zero the gradients and make sure the model is set to train
        model.train()
        optimizer.zero_grad()
      else: # If we're in evaluation phase, then make sure the model is set to eval
        model.eval()

      with torch.set_grad_enabled(train): # Either to perform the next lines with gradient tracing or not
        pred = model(x) # Get the model output from x
        pred = pred.reshape(pred.shape[0], ) # Flatten the prediction values

        y = torch.from_numpy(y.values.reshape(y.shape[0], )).float()

        criterion = torch.nn.MSELoss() # Define the criterion as MSELoss from torch
        loss = criterion(pred, y)

        if train:
          loss.backward()
          optimizer.step()

      return loss
      
    def train_loop(model, train_loader, eval_loader, lr, w_decay, epochs, eval_step):
      step = 0
      """ Defining our optimizer """
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
      epochs_l, steps, t_losses, v_losses = [], [], [], []

      epochs_tqdm = tqdm(range(epochs), desc='Training in Progress', leave=True)
      for epoch in epochs_tqdm:
        for x, y in train_loader:
          loss_batch = model_step(model, x, y, optimizer, train=True)
          step +=1
          if step % eval_step == 0:
            train_loss = loss_batch
            val_loss = 0
            for x, y in eval_loader:
              val_loss += model_step(model, x, y, train=False)
            epochs_l.append(epoch+1)
            steps.append(step)
            t_losses.append(train_loss.detach().numpy())
            v_losses.append(val_loss.detach().numpy())
            clear_output(wait=True)
            display(pd.DataFrame({'Epoch': epochs_l, 'Step': steps, 'Training Loss': t_losses, 'Validation Loss': v_losses}))
            

    file_path = "fm_model.pkl"  # Change the path as per your preference
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    selected_new_id = selected_user_id    
     # Save selected_new_id in a variable
    new_id_variable = selected_new_id
        
    new=user_mappings.loc[new_id_variable].new_id
    features,ratings = dataset_test[[new]]


    model.eval()
    #with torch.no_grad():
  
    # Replace None with the new_id of the user
    items_our_user_rated = (train[train.userId==new].movieId).unique().tolist()
    items_our_user_rated.extend((test[test.userId==new].movieId).unique().tolist())

    items_our_user_can_rate = movie_mappings[~movie_mappings.new_id.isin(items_our_user_rated)].new_id.tolist()


    N = integer_value  # Number of recommendations

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


    #st.write("Top Recommendations:")
    for item_id, predicted_rating in top_recommendations:
        movie_title = movie_mappings[movie_mappings.new_id == item_id].index[0]
        
    for item_id, predicted_rating in top_recommendations:
        movie_title = movies[movies['movieId'] == item_id]['title'].values[0]
        
        # Create a container with a specified width and height
        with st.container():
            # Set the container's style to display as a rectangle with a border and padding
            st.markdown(
                """
                <style>
                .custom-box4 {
                    background-color: #f5f5f5;
                    border: 2px solid #336699;
                    border-radius: 8px;
                    padding: 12px;
                    width: 40%;
                    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.2);
                }

                .custom-text4 {
                    font-size: 18px;
                    font-weight: bold;
                    color: #336699;
                    margin-bottom: 8px;
                }
                </style>
                """
            , unsafe_allow_html=True)

            # Display the movie title and predicted rating
            st.write(
                '<div class="custom-box4">'
                f'<div class="custom-text4">{movie_title}</div>'
                '</div>'
                , unsafe_allow_html=True
            )


def run_movie_based():
    df = pd.merge(ratings,movies,on='movieId')
    reader = Reader(rating_scale=(1, 5))
    data = SurpriseDataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    # Load the KNN model from the file
    model = joblib.load('knn_model.pkl')
    
    def get_similar_movies(movie_name, k):
        tsr_inner_id = model.trainset.to_inner_iid(movies[movies['title'] == movie_name]['movieId'].values[0])
        tsr_neighbors = model.get_neighbors(tsr_inner_id, k=k)
        similar_movie_ids = [model.trainset.to_raw_iid(inner_id) for inner_id in tsr_neighbors]

        similar_movies = movies[movies['movieId'].isin(similar_movie_ids)].copy()
        similar_movies['similarity'] = [model.sim[tsr_inner_id, model.trainset.to_inner_iid(movie_id)] for movie_id in similar_movie_ids]
        similar_movies.sort_values(by='similarity', ascending=False, inplace=True)
        
        with st.container():
        # Set the container's style to display as a rectangle with a border and padding
            st.markdown(
                """
                <style>
                .custom-box2 {
                    background-color: #f8f8f8;
                    border: 2px solid #4e6bff;
                    border-radius: 10px;
                    padding: 15px;
                    width: 40%;
                }

                .custom-title2 {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4e6bff;
                    margin-bottom: 10px;
                }
                </style>
                """
            , unsafe_allow_html=True)
            
            html_string = f'<div class="custom-box2">' \
                          f'<div class="custom-title2">Top {k} Similar Movies to \'{movie_name}\':</div>' \
                          f'</div>'

            # Display the HTML string in Streamlit
            st.write(html_string, unsafe_allow_html=True)
        

        #st.write(f"Top {k} Similar Movies to '{movie_name}':")
        for index, row in similar_movies.iterrows():
            # Create a container with a specified width and height
            with st.container():
                # Set the container's style to display as a rectangle with a border and padding
                st.markdown(
                    """
                    <style>
                    .custom-box4 {
                        background-color: #f5f5f5;
                        border: 2px solid #336699;
                        border-radius: 8px;
                        padding: 12px;
                        width: 40%;
                        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.2);
                    }

                    .custom-text4 {
                        font-size: 18px;
                        font-weight: bold;
                        color: #336699;
                        margin-bottom: 8px;
                    }
                    </style>
                    """
                , unsafe_allow_html=True)

                html_string = f'<div class="custom-box4">' \
                              f'<div class="custom-text4">{row["title"]}</div>' \
                              f'</div>'

                # Display the HTML string in Streamlit
                st.write(html_string, unsafe_allow_html=True)
                #st.write(row['title'])
        
    # Create a selectbox to choose a movie
    
    # Get the selected movie from the selectbox
    selected_movie = st.sidebar.selectbox("Select a movie", movies['title'].unique(), key='movie_select')

    # Store the selected movie in the app state
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = selected_movie
    else:
        if st.session_state.selected_movie != selected_movie:
            st.experimental_set_query_params(selected_movie=selected_movie)
            st.session_state.selected_movie = selected_movie

    # Retrieve the selected movie from the app state
    selected_movie = st.session_state.selected_movie

    #selected_movie = st.sidebar.selectbox("Choose a movie", movies['title'].unique(), index=0, key='movie_select')


    # Filter the DataFrame based on the selected movie
    #filtered_movies = movies[movies['title'] == selected_movie]

    # Display the filtered movies
    #st.write(filtered_movies)
    
    # Display a name for the input field and get the numeric input
    valueS = st.number_input("Enter top number of similar movies you want:", value=5, step=1, format="%d")

    # Convert the input value to an integer
    integer_value = int(valueS)
    
    get_similar_movies(selected_movie, integer_value)


# Streamlit app code
def main():
    # Define the path to the image
    image_path = "images/r1.png"

    # Open and display the image
    image = Image.open(image_path)
    resized_image = image.resize((800, 200))
    st.image(resized_image, use_column_width=True)


    # Add buttons with images
    col1, col2 = st.columns(2)

    if col1.button("User Based Recommendation", key="movie_button"):
        run_user_based()

    #col1.image(image1, use_column_width=True)

    if col2.button("Movie Based Recommendation"):
        run_movie_based()

    #col2.image(image2, use_column_width=True)

if __name__ == "__main__":
    main()

    

