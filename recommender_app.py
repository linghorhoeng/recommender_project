import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import streamlit as st

# Load MovieLens dataset
@st.cache_data
def load_data():
    movies = pd.read_csv('./movies.csv')
    ratings = pd.read_csv('./ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# Prepare data for the recommender
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train the model
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
algo = SVD()
algo.fit(trainset)

# Create a mapping of movieId to title
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

# Recommender function
def get_recommendations(selected_movie, top_n=10):
    selected_movie_id = movies[movies['title'] == selected_movie]['movieId'].values[0]
    predictions = [
        (movie_id, algo.predict(uid=0, iid=movie_id).est)
        for movie_id in movies['movieId'].unique()
        if movie_id != selected_movie_id
    ]
    # Sort by predicted rating
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return [(movie_id_to_title[movie_id], score) for movie_id, score in recommendations]

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Select a movie you liked, and we'll recommend similar movies!")

# Movie selection
selected_movie = st.selectbox("Choose a movie", movies['title'].sort_values())

if selected_movie:
    st.write(f"Movies recommended based on your choice: **{selected_movie}**")
    recommendations = get_recommendations(selected_movie)
    for idx, (title, score) in enumerate(recommendations, start=1):
        st.write(f"{idx}. {title} (Predicted rating: {score:.2f})")
