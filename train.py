from utils.data_loader import DataLoader
from utils.recommender import MovieRecommender

def main():
    ratings_path = "data/tmdb_ratings.csv"
    movies_path = "data/tmdb_movies.csv"

    # Load data
    data_loader = DataLoader(ratings_path, movies_path)
    ratings_df, movies_df = data_loader.load_data()

    # Initialize Recommender
    recommender = MovieRecommender(ratings_df, movies_df)

    # Train Model
    recommender.train_model()
    # Optionally do grid search
    recommender.grid_search()

if __name__ == "__main__":
    main()