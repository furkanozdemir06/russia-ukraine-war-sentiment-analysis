import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Russia-Ukraine War Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# Load trained model (l) and vectorizer (cv)
@st.cache_resource
def load_assets():
    l = joblib.load('best_model.pkl')
    cv = joblib.load('vectorizer.pkl')
    return l, cv

l, cv = load_assets()

# UI Header
st.title("🐦 Russia-Ukraine War Tweet Sentiment Analyzer")
st.markdown("Powered by **Logistic Regression (94.36% Accuracy)**")
st.divider()

# User input text box
user_tweet = st.text_area(
    "Enter a tweet to analyze:", 
    value="Diplomatic talks must continue to ensure peace and stability.",
    height=120
)

# Predict button
if st.button("Analyze Sentiment", type="primary"):
    if user_tweet.strip() != "":
        # Transform text using cv and predict using l
        tweet_vec = cv.transform([user_tweet])
        prediction = l.predict(tweet_vec)[0]
        
        st.subheader("Analysis Result:")
        
        # Display output
        if prediction == 1 or str(prediction).lower() == 'positive':
            st.success("Sentiment: **Positive** 😊")
        elif prediction == -1 or str(prediction).lower() == 'negative':
            st.error("Sentiment: **Negative** 😡")
        else:
            st.info("Sentiment: **Neutral** 😐")
    else:
        st.warning("Please enter a valid tweet before analyzing.")