"""
Movie Recommendation System using User Preference and Collaborative Filtering Techniques
-------------------------------------------------------------------------------------
Implements:
  - Data loading & cleaning (duplicates, missing values)
  - Exploratory data analysis helpers
  - User-based collaborative filtering (cosine similarity)
  - Item-based collaborative filtering (cosine similarity)
  - Top-N personalised recommendations
  - Simple accuracy evaluation (RMSE on a held-out split)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class MovieRecommender:
    def __init__(self, csv_path="data/ratings.csv"):
        self.csv_path = csv_path
        self.df = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None

    # ---------- Data collection & preparation ----------
    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def clean_data(self):
        """Handle missing values and duplicates."""
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(subset=["user_id", "movie_id", "rating"], inplace=True)
        self.df["rating"] = self.df["rating"].astype(float)
        after = len(self.df)
        print(f"Cleaned data: removed {before - after} invalid/duplicate rows.")
        return self.df

    # ---------- Exploratory / user preference analysis ----------
    def genre_preferences(self, user_id):
        user_df = self.df[self.df["user_id"] == user_id]
        return user_df.groupby("genre")["rating"].mean().sort_values(ascending=False)

    # ---------- Feature engineering ----------
    def build_user_item_matrix(self):
        self.user_item_matrix = self.df.pivot_table(
            index="user_id", columns="movie_title", values="rating"
        ).fillna(0)
        return self.user_item_matrix

    # ---------- Similarity measurement ----------
    def compute_user_similarity(self):
        sim = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            sim, index=self.user_item_matrix.index, columns=self.user_item_matrix.index
        )
        return self.user_similarity

    def compute_item_similarity(self):
        sim = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            sim, index=self.user_item_matrix.columns, columns=self.user_item_matrix.columns
        )
        return self.item_similarity

    # ---------- User-based collaborative filtering ----------
    def recommend_user_based(self, user_id, top_n=5):
        if self.user_similarity is None:
            self.compute_user_similarity()

        sim_scores = self.user_similarity[user_id].drop(user_id)
        other_ratings = self.user_item_matrix.T[sim_scores.index]
        weighted_ratings = other_ratings.dot(sim_scores) / (sim_scores.sum() + 1e-9)

        already_rated = self.user_item_matrix.loc[user_id]
        recommendations = weighted_ratings[already_rated == 0].sort_values(ascending=False)
        return recommendations.head(top_n)

    # ---------- Item-based collaborative filtering ----------
    def recommend_item_based(self, user_id, top_n=5):
        if self.item_similarity is None:
            self.compute_item_similarity()

        user_ratings = self.user_item_matrix.loc[user_id]
        scores = self.item_similarity.dot(user_ratings) / (
            self.item_similarity.abs().sum(axis=1) + 1e-9
        )
        already_rated = user_ratings[user_ratings > 0].index
        scores = scores.drop(already_rated)
        return scores.sort_values(ascending=False).head(top_n)

    # ---------- Model evaluation ----------
    def evaluate_rmse(self, test_size=0.2, random_state=42):
        """A simple accuracy check using a train/test split and
        predicting held-out ratings from user-based similarity."""
        train_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state
        )

        train_matrix = train_df.pivot_table(
            index="user_id", columns="movie_title", values="rating"
        ).fillna(0)
        sim = cosine_similarity(train_matrix)
        sim_df = pd.DataFrame(sim, index=train_matrix.index, columns=train_matrix.index)

        preds, actuals = [], []
        for _, row in test_df.iterrows():
            u, m, r = row["user_id"], row["movie_title"], row["rating"]
            if u not in train_matrix.index or m not in train_matrix.columns:
                continue
            sims = sim_df[u].drop(u, errors="ignore")
            ratings = train_matrix[m].drop(u, errors="ignore")
            denom = sims.sum()
            pred = (sims * ratings).sum() / denom if denom != 0 else train_matrix[m].mean()
            preds.append(pred)
            actuals.append(r)

        if not preds:
            return None
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        return rmse


def demo():
    rec = MovieRecommender("data/ratings.csv")
    rec.load_data()
    rec.clean_data()
    rec.build_user_item_matrix()

    print("\nUser-Item Matrix:\n", rec.user_item_matrix)

    uid = 1
    print(f"\nGenre preference for user {uid}:\n", rec.genre_preferences(uid))

    print(f"\nTop recommendations (user-based) for user {uid}:\n", rec.recommend_user_based(uid))
    print(f"\nTop recommendations (item-based) for user {uid}:\n", rec.recommend_item_based(uid))

    rmse = rec.evaluate_rmse()
    print(f"\nModel RMSE on held-out ratings: {rmse:.3f}" if rmse else "\nNot enough data to evaluate.")


if __name__ == "__main__":
    demo()
