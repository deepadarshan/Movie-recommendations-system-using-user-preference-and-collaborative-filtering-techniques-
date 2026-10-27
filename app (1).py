import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.caption("User preference & collaborative filtering techniques")


@st.cache_resource
def load_recommender():
    rec = MovieRecommender("data/ratings.csv")
    rec.load_data()
    rec.clean_data()
    rec.build_user_item_matrix()
    return rec


rec = load_recommender()
users = sorted(rec.user_item_matrix.index.tolist())

user_id = st.selectbox("Select a user", users)
method = st.radio("Recommendation method", ["User-based", "Item-based"], horizontal=True)
top_n = st.slider("Number of recommendations", 1, 10, 5)

st.subheader(f"Genre preferences for user {user_id}")
st.bar_chart(rec.genre_preferences(user_id))

st.subheader("Recommended movies")
if method == "User-based":
    results = rec.recommend_user_based(user_id, top_n=top_n)
else:
    results = rec.recommend_item_based(user_id, top_n=top_n)

if results.empty:
    st.info("No new recommendations — this user has rated everything in the dataset.")
else:
    for title, score in results.items():
        st.write(f"**{title}** — predicted score: {score:.2f}")

with st.expander("Model accuracy (RMSE)"):
    rmse = rec.evaluate_rmse()
    st.write(f"RMSE on held-out ratings: **{rmse:.3f}**" if rmse else "Not enough data to evaluate.")
