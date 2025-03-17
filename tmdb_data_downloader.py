import requests
import pandas as pd

# Replace with your TMDb API key
api_key = "a38d83603bc41f6246679b70b735bccb"

def fetch_popular_movies(api_key, num_pages=5):
    movies = []
    for page in range(1, num_pages + 1):
        url = "https://api.themoviedb.org/3/movie/popular"
        params = {"api_key": api_key, "language": "en-US", "page": page}
        response = requests.get(url, params=params)
        data = response.json()
        movies.extend(data.get("results", []))
    return movies

def simulate_ratings(movies, num_users=500):
    import numpy as np
    ratings = []
    # Use movie IDs from fetched movies
    movie_ids = [movie["id"] for movie in movies if "id" in movie]
    for user_id in range(1, num_users + 1):
        # Each user rates 10 random movies
        rated = np.random.choice(movie_ids, size=10, replace=False)
        for mid in rated:
            rating = np.random.uniform(1, 5)
            ratings.append({"userId": user_id, "movieId": mid, "rating": round(rating, 1)})
    return pd.DataFrame(ratings)

def main():
    # Fetch popular movies
    movies = fetch_popular_movies(api_key, num_pages=5)
    # Save movies data
    movies_df = pd.DataFrame(movies)
    movies_df.to_csv("data/tmdb_movies.csv", index=False)
    
    # Generate simulated ratings
    ratings_df = simulate_ratings(movies, num_users=500)
    ratings_df.to_csv("data/tmdb_ratings.csv", index=False)
    
    print("✅ Datasets saved: tmdb_movies.csv and tmdb_ratings.csv")

if __name__ == "__main__":
    main()
