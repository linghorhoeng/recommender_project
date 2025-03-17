import streamlit as st
import requests
import random

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from math import ceil
import pandas as pd
import numpy as np

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

mode = st.radio("Select User Mode", ["Manual User ID", "🎲 Random User"], horizontal=True)
if mode == "Manual User ID":
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
else:
    user_id = random.randint(1, 500)  # or however many users you have

top_n = st.slider("Number of Recommendations", 1, 20, 5)

if st.button("Get Recommendations"):
    with st.spinner("Fetching..."):
        response = requests.post("http://127.0.0.1:5000/recommend", json={"user_id": user_id, "top_n": top_n})
        if response.ok:
            data = response.json()
            recs = data.get("recommendations", [])
            if not recs:
                st.info("No recommendations. Possibly user ID not in training data.")
            else:
                st.write(f"**Recommendations for User {user_id}**")
                for idx, rec in enumerate(recs, start=1):
                    st.write(f"{idx}. {rec['title']} (Predicted Rating: {rec['predicted_rating']:.2f})")
        else:
            st.error("API call failed.")


# TMDb API Key
api_key = "a38d83603bc41f6246679b70b735bccb"

# Fetch genres
@st.cache_data
def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        genres = response.json()['genres']
        return {genre['name']: genre['id'] for genre in genres}
    return {}

# Fetch movies
def fetch_movies(endpoint, params=None):
    url = f"https://api.themoviedb.org/3/{endpoint}?api_key={api_key}&language=en-US"
    if params:
        url += "&" + "&".join(f"{k}={v}" for k, v in params.items())
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}

# Get movie poster URL
def get_poster_url(poster_path):
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Get trailer video URL
def get_trailer_url(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        videos = response.json().get("results", [])
        if videos:
            trailer = next((video for video in videos if video['type'] == 'Trailer'), None)
            if trailer:
                return f"https://www.youtube.com/embed/{trailer['key']}"
    return None

# Machine Learning: SVD Model for Collaborative Filtering
@st.cache_data
def build_svd_model(ratings_df):
    # Using Surprise library's SVD model for collaborative filtering
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    return svd

# Predict ratings for a user-item pair using SVD
def predict_rating(svd_model, user_id, movie_id):
    return svd_model.predict(user_id, movie_id).est

# Display movie card with poster and trailer
def display_movie_card(movie, col, show_trailer=False):
    poster_url = get_poster_url(movie.get('poster_path'))
    title = movie['title']
    movie_id = movie['id']
    
    # Display movie poster and title
    col.markdown(
        f"""
        <div class="movie-card">
            <img src="{poster_url}" alt="{title}" class="movie-poster">
            <div class="movie-title">{title}</div>
    """,
        unsafe_allow_html=True,
    )

    # Display Trailer below poster if show_trailer is True
    if show_trailer:
        trailer_url = get_trailer_url(movie_id)
        if trailer_url:
            col.markdown(f'<iframe width="100%" height="315" src="{trailer_url}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', unsafe_allow_html=True)
    
    # Close the div for movie-card
    col.markdown("</div>", unsafe_allow_html=True)


# Custom CSS Styling
st.markdown(
    """
    <style>
    body {
        background-color: #1c1c1e;
        color: #f5f5f5;
    }
    .main {
        background-color: #28282b;
        border-radius: 10px;
        padding: 20px;
    }
    .movie-card {
        text-align: center;
        margin-bottom: 20px;
    }
    .movie-card img {
        border-radius: 10px;
        width: 100%;
        max-height: 300px;
        object-fit: cover;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.4);
    }
    .movie-title {
        margin-top: 10px;
        font-weight: bold;
        font-size: 14px;
        color: #ff6f61;
    }
    .movie-poster:hover {
        transform: scale(1.05);
        transition: transform 0.2s;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        padding: 10px 0;
        color: #aaaaaa;
        font-size: 12px;
    }
    footer a {
        color: #ff6f61;
        text-decoration: none;
        margin: 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Sidebar
st.write("Discover movies by genres and explore similar recommendations!")

genres = fetch_genres()

if genres:
    with st.sidebar.expander("🎭 Explore by Genre", expanded=True):
        selected_genre = st.selectbox("Select a Genre", list(genres.keys()))
        genre_page = st.number_input("Page for Genre Recommendations", min_value=1, value=1, step=1)

    # Fetch Popular Movies
    popular_movies = fetch_movies("movie/popular", {"page": 1}).get("results", [])
    movie_titles = {movie['title']: movie['id'] for movie in popular_movies}

    # Main Content
    selected_movie = st.selectbox("Choose a movie", list(movie_titles.keys()))

    if selected_movie:
        selected_movie_id = movie_titles[selected_movie]
        movie_details = fetch_movies(f"movie/{selected_movie_id}")
        if movie_details:
            st.subheader(f"**{movie_details['title']}** ({movie_details['release_date'][:4]})")
            cols = st.columns([1, 2])

            with cols[0]:
                poster_url = get_poster_url(movie_details.get('poster_path'))
                if poster_url:
                    st.image(poster_url, width=200)

            with cols[1]:
                st.write(f"**Rating**: {movie_details['vote_average']}/10")
                st.write(f"**Votes**: {movie_details['vote_count']}")
                st.write(f"**Release Date**: {movie_details['release_date']}")
                st.write(f"**Runtime**: {movie_details['runtime']} minutes")
                st.write(f"**Overview**: {movie_details['overview']}")

            # Display Trailer below the poster only when the movie is selected
            trailer_url = get_trailer_url(selected_movie_id)
            if trailer_url:
                st.markdown(f'<iframe width="100%" height="315" src="{trailer_url}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', unsafe_allow_html=True)

            # Similar Movies
            st.subheader("Similar Movies")
            similar_movies = fetch_movies(f"movie/{selected_movie_id}/similar", {"page": 1})
            cols = st.columns(5)
            for idx, movie in enumerate(similar_movies.get("results", [])[:10]):
                with cols[idx % 5]:
                    display_movie_card(movie, st)

    # Genre Movies
    st.subheader(f"Movies in {selected_genre}")
    genre_id = genres[selected_genre]
    genre_movies = fetch_movies("discover/movie", {"with_genres": genre_id, "page": genre_page})
    cols = st.columns(5)
    for idx, movie in enumerate(genre_movies.get("results", [])[:10]):
        with cols[idx % 5]:
            # Add the `show_trailer` argument to control whether to show the trailer below the poster.
            display_movie_card(movie, st, show_trailer=False)

# Footer - Centered
st.markdown(
    """
    <footer style="text-align: center; margin-top: 50px;">
        <div>© 2025 Movie Recommender. All Rights Reserved.</div>
        <div class="footer-links" style="text-align: center;">
            <a href="#">Terms of Use</a> |
            <a href="#">Privacy</a> |
            <a href="#">Cookie Preferences</a> |
            <a href="#">Help Center</a> |
            <a href="#">Jobs</a> |
            <a href="#">Legal Notices</a> |
            <a href="#">Contact Us</a>
        </div>
    </footer>
    """,
    unsafe_allow_html=True,
)