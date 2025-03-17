import requests
import streamlit as st

# TMDb API Key
api_key = "a38d83603bc41f6246679b70b735bccb"

@st.cache_data(show_spinner=False)
def get_poster_url(movie_title):
    """Fetch movie poster from TMDB API."""
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": movie_title}
        response = requests.get(search_url, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None  # No poster found
    except requests.RequestException as e:
        print(f"Error fetching poster for {movie_title}: {e}")
        return None
    
def get_movie_details(title):
    """Fetch movie details from TMDB API."""
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": title}
        response = requests.get(search_url, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        if not results:
            return None  # No results found

        movie_id = results[0].get("id")
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        details_response = requests.get(details_url, params={"api_key": api_key}).json()

        # Add poster URL safely
        poster_path = details_response.get("poster_path")
        details_response["poster_url"] = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

        return details_response
    except requests.RequestException as e:
        print(f"Error fetching details for {title}: {e}")
        return None

def get_movie_trailer(title):
    """Fetch movie trailer from TMDB API."""
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": api_key, "query": title}
        response = requests.get(search_url, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        if not results:
            return None  # No movie found

        movie_id = results[0].get("id")
        videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        videos_response = requests.get(videos_url, params={"api_key": api_key}).json()

        trailers = [video for video in videos_response.get("results", []) if video.get("type", "").lower() == "trailer"]
        if trailers:
            return f"https://www.youtube.com/watch?v={trailers[0]['key']}"
        return None
    except requests.RequestException as e:
        print(f"Error fetching trailer for {title}: {e}")
        return None
