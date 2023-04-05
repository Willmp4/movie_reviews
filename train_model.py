import pandas as pd 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib



def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(word for word in text if not word.isdigit())
    tokens= nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = SnowballStemmer("english")
    stemmed_tokens = [stemmer.stem(word) if not word.endswith('ion') else word for word in filtered_tokens]
    preprocess_text = ' '.join(stemmed_tokens)
    return preprocess_text

import os
import joblib

def main(train=False):
    # Define the file names for the model and vectorizer
    tfidf_vectorizer_file = 'tfidf_vectorizer.pkl'
    model_file = 'sentiment_analysis_model.pkl'

    # Check if both files exist
    if os.path.exists(tfidf_vectorizer_file) and os.path.exists(model_file):
        # Load the vectorizer and model from the files
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_file)
        model = joblib.load(model_file)
        
        # Load the test data
        df = pd.read_csv('IMDB Dataset.csv')
        X_test = tfidf_vectorizer.transform(df['review'])
        y_test = df['sentiment']
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)[:, 1]
        
    else:
        # Load the dataset into a DataFrame
        df = pd.read_csv('IMDB Dataset.csv')

        # Apply the preprocess_text function to the 'review' column
        df['review'] = df['review'].apply(preprocess_text)

        # Create and save the TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_features = tfidf_vectorizer.fit_transform(df['review'])
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_file)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(tfidf_features, df['sentiment'], test_size=0.2, random_state=42)

        # Train and save the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, model_file)

        # Make predictions on the test data
        y_pred = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)[:, 1]

    return X_test, y_test, y_pred, y_pred_probs, df, model, tfidf_vectorizer

if __name__ == '__main__':
    main()