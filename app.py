from flask import Flask, request, jsonify
import pandas as pd
from utils.data_loader import DataLoader
from utils.recommender import MovieRecommender

app = Flask(__name__)

# Load Data
ratings_path = "data/tmdb_ratings.csv"
movies_path = "data/tmdb_movies.csv"
data_loader = DataLoader(ratings_path, movies_path)
ratings_df, movies_df = data_loader.load_data()

# Initialize Recommender and load model
recommender = MovieRecommender(ratings_df, movies_df)
recommender.load_model()

@app.route("/")
def home():
    return jsonify({"message": "TMDb-based SVD Recommender is running!"})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_id = data.get("user_id", 1)
    top_n = data.get("top_n", 5)

    recs = recommender.recommend(user_id, top_n)
    if recs.empty:
        return jsonify({"recommendations": []})

    return jsonify({"recommendations": recs.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
