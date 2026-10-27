# Movie Recommendation System using User Preference and Collaborative Filtering Techniques

A lightweight, ready-to-run movie recommender built in Python with pandas and scikit-learn. It demonstrates both **user-based** and **item-based collaborative filtering** using cosine similarity, plus basic data cleaning, EDA, and accuracy evaluation.

**Author:** Deepa Darshan S · 24UAI0013 · 2024–2027

## Features

- Data collection & preparation from a CSV ratings dataset
- Data cleaning (duplicates, missing values)
- Exploratory data analysis & user genre-preference analysis
- Feature engineering: user-item rating matrix
- Similarity measurement (cosine similarity)
- User-based collaborative filtering
- Item-based collaborative filtering
- Top-N personalised recommendations
- Model evaluation via RMSE on a held-out split

## Project Structure

```
movie-recommender/
├── data/
│   └── ratings.csv        # sample user-movie ratings dataset
├── recommender.py          # core recommendation engine
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd movie-recommender
pip install -r requirements.txt
```

## Usage

Run the built-in demo:

```bash
python recommender.py
```

Or use it in your own script:

```python
from recommender import MovieRecommender

rec = MovieRecommender("data/ratings.csv")
rec.load_data()
rec.clean_data()
rec.build_user_item_matrix()

# See what a user tends to like
print(rec.genre_preferences(user_id=1))

# Get recommendations
print(rec.recommend_user_based(user_id=1, top_n=5))
print(rec.recommend_item_based(user_id=1, top_n=5))

# Evaluate accuracy
print(rec.evaluate_rmse())
```

Swap in your own dataset (e.g. MovieLens) by replacing `data/ratings.csv` — it just needs `user_id, movie_id, movie_title, genre, rating` columns.

## How It Works

1. **User-based CF** finds users with similar taste (cosine similarity over rating vectors) and recommends movies those similar users rated highly.
2. **Item-based CF** finds movies similar to ones a user already liked, based on how all users rated them.
3. **Evaluation** holds out a portion of ratings, predicts them from the training data, and reports RMSE (lower is better).

## Project Development Log

This system was built following a structured timeline — from requirement analysis and dataset understanding through data cleaning, EDA, feature engineering, similarity measurement, model building/testing, UI design, and final evaluation:

| Date | Milestone |
|------|-----------|
| 13/05/26 | Project introduction |
| 14/05/26 | Understanding recommendation systems |
| 15/05/26 | Survey on recommendation systems |
| 16/05/26 | Requirement analysis |
| 18/05/26 | Understanding the movie dataset |
| 19/05/26 | Data collection and preparation |
| 20/05/26 | Data cleaning and pre-processing |
| 21/05/26 | Handling missing values, duplicates |
| 22/05/26 | Exploring data analysis |
| 23/05/26 | User preference analysis |
| 25/05/26 | Feature engineering for systems |
| 26/05/26 | Introduction to collaborative filtering |
| 27/05/26 | User-based collaborative filtering |
| 28/05/26 | Similarity measurement techniques |
| 29/05/26 | Cosine similarity implementation |
| 30/05/26 | Building the model |
| 31/05/26 | Model training and testing |
| 01/06/26 | Generate personalised recommendations |
| 02/06/26 | Performance evaluation for model |
| 03/06/26 | Item-based collaborative filtering |
| 04/06/26 | Recommendation accuracy |
| 05/06/26 | Recommendation model with application |
| 06/06/26 | User interface design |
| 08/06/26 | System testing and debugging |
| 09/06/26 | Final result analysis |
| 10/06/26 | Project documentation |
| 11/06/26 | Project review and presentation |

## License

MIT — free to use and modify.
