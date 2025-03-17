import pandas as pd

class DataLoader:
    def __init__(self, ratings_path, movies_path):
        self.ratings_path = ratings_path
        self.movies_path = movies_path

    def load_data(self):
        """Load the TMDb-based ratings and movies datasets."""
        ratings_df = pd.read_csv(self.ratings_path)
        movies_df = pd.read_csv(self.movies_path)

        # Ensure correct dtypes
        # For example, 'id' is your movieId in tmdb_movies.csv
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')

        return ratings_df, movies_df
