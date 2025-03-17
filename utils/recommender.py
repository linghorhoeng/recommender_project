import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

class MovieRecommender:
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.model = None
        self.trainset = None

    def train_model(self, n_factors=50, n_epochs=20):
        """Train and save the SVD model."""
        reader = Reader(rating_scale=(1, 5))  # or (0.5, 5.0)
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']],
            reader
        )
        self.trainset = data.build_full_trainset()

        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs)
        self.model.fit(self.trainset)

        os.makedirs("models", exist_ok=True)
        with open("models/svd_model.pkl", "wb") as f:
            pickle.dump((self.model, self.trainset), f)

        print("✅ Model and trainset saved.")

    def grid_search(self):
        """Perform Grid Search to find best hyperparameters."""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']],
            reader
        )
        param_grid = {
            'n_factors': [50, 100],
            'n_epochs': [10, 20]
        }
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
        gs.fit(data)

        print(f"✅ Best RMSE: {gs.best_score['rmse']}")
        print(f"✅ Best Params: {gs.best_params['rmse']}")

        self.model = gs.best_estimator['rmse']
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

        with open("models/svd_model.pkl", "wb") as f:
            pickle.dump((self.model, self.trainset), f)

        print("✅ Best model saved after grid search.")

    def load_model(self):
        """Load the trained model and trainset."""
        model_path = "models/svd_model.pkl"
        if not os.path.exists(model_path):
            print("❌ Model file not found. Please train first.")
            return
        with open(model_path, "rb") as f:
            self.model, self.trainset = pickle.load(f)
        print("✅ Model and trainset loaded.")

    def recommend(self, user_id, top_n=5):
        """Recommend top N movies for a given user from the TMDb dataset."""
        if not self.model or not self.trainset:
            print("❌ Model not loaded. Call `load_model()` first.")
            return pd.DataFrame()

        # Check if user exists in the trainset
        if not self.trainset.knows_user(user_id):
            print(f"❌ Unknown user {user_id}.")
            return pd.DataFrame()

        # Get all movie IDs from the dataset
        all_movie_ids = self.movies_df['id'].dropna().unique()

        # Find movies user has already rated
        seen_movies = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].unique()

        # Predict ratings for unseen movies
        unseen = list(set(all_movie_ids) - set(seen_movies))
        predictions = []
        for mid in unseen:
            pred = self.model.predict(user_id, mid).est
            predictions.append((mid, pred))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]

        # Build DataFrame
        rec_df = pd.DataFrame(top_predictions, columns=['movieId', 'predicted_rating'])
        # Merge with movie details
        rec_df = rec_df.merge(self.movies_df, left_on='movieId', right_on='id', how='left')

        return rec_df[['movieId', 'title', 'predicted_rating']]
