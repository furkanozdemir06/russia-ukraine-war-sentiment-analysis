# Ukraine-Russia War Twitter Sentiment Analysis

An NLP project that analyzes tweets about the Russia-Ukraine war. Tweets are cleaned and vectorized, labeled with VADER sentiment scores, explored with word clouds and named entity recognition, and used to train and compare several classical machine learning classifiers.

## Overview

The goal is to build a sentiment classification model for war-related tweets. The pipeline covers text preprocessing, lexicon-based sentiment labeling, exploratory text analysis, and a benchmark of eight classifiers. The best model and its vectorizer are saved for reuse.

## Workflow

1. **Data loading and EDA**
   - Kept only the `username`, `tweet`, and `language` columns.
   - Checked missing values and the language distribution.
2. **Text preprocessing**
   - Lowercasing, URL removal, punctuation and digit removal, newline cleanup, and whitespace normalization.
   - Tokenization (NLTK), English stop-word removal, and lemmatization (WordNet).
3. **Sentiment labeling with VADER**
   - Computed positive, negative, and neutral scores for each tweet.
   - Assigned each tweet the label with the highest score.
4. **Exploratory text analysis**
   - Word clouds for all tweets, positive tweets, and negative tweets.
   - Named entity recognition with spaCy (`en_core_web_sm`) on a 5,000-tweet sample to find the most mentioned organizations.
5. **Modeling**
   - Vectorized tweets with `CountVectorizer`.
   - Trained and compared 8 classifiers on an 80/20 train/test split, ranked by F1-score.
6. **Model export**
   - Saved the Logistic Regression model and the vectorizer with `joblib`.

## Models Compared

- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors

Each model is evaluated with accuracy, precision, recall, and F1-score (micro-averaged), plus a confusion matrix.

## Results

| Metric | Best model (Logistic Regression) |
|--------|----------------------------------|
| Accuracy | ~94.36% |
| Precision | ~94.36% |
| Recall | ~94.36% |
| F1-score | ~94.36% |

Note that the labels come from VADER rather than human annotation, so these scores measure how well the classifiers reproduce VADER's sentiment labels, not agreement with human judgment.

## Tech Stack

- Python
- pandas, NumPy
- NLTK (tokenization, stop words, lemmatization, VADER)
- spaCy (named entity recognition)
- scikit-learn
- WordCloud
- matplotlib, seaborn
- joblib
