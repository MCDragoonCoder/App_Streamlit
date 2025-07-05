import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import praw
import os
from dotenv import load_dotenv
import nltk
nltk.download('vader_lexicon')

# Load .env variables
load_dotenv()

# Initialize Reddit API using PRAW
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Analyze sentiment using VADER
def get_emotion(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Fetch and analyze Reddit posts
@st.cache_data(show_spinner=False)
def fetch_and_analyze(subreddit_name, keyword, limit):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for post in subreddit.hot(limit=limit):
        if keyword.lower() in post.title.lower():
            emotion = get_emotion(post.title)
            posts_data.append({
                "title": post.title,
                "score": post.score,
                "emotion": emotion,
                "url": post.url
            })

    return pd.DataFrame(posts_data)

# Streamlit layout with 2 columns
st.title("🧠 Reddit EmotionAnalyzer")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("🔎 Input")
    subreddit_name = st.text_input("Subreddit", "AskReddit")
    keyword = st.text_input("Keyword", "how")
    post_limit = st.slider("Number of Posts to Analyze", 5, 100, 10)
    analyze_button = st.button("Analyze")

with right_col:
    if analyze_button:
        with st.spinner("Analyzing posts..."):
            df = fetch_and_analyze(subreddit_name, keyword, post_limit)

        if not df.empty:
            st.success(f"Found {len(df)} matching posts.")
            st.dataframe(df)

            st.subheader("📊 Emotion Distribution")
            fig, ax = plt.subplots()
            df["emotion"].value_counts().plot(kind="bar", ax=ax, color='skyblue')
            ax.set_ylabel("Count")
            ax.set_xlabel("Emotion")
            st.pyplot(fig)
        else:
            st.warning("No posts found with that keyword.")
