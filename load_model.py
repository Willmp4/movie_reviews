# load_model.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from train_model import preprocess_text

def main():
    # Load the saved model and vectorizer
    model = joblib.load('sentiment_analysis_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Load a new dataset or individual data points for making predictions or evaluations
    # Replace this with your own data loading method
    new_data = ['This movie was amazing!', 'I did not enjoy this film at all.']

    # Preprocess the new data
    preprocessed_data = [preprocess_text(text) for text in new_data]

    # Transform the preprocessed data using the loaded vectorizer
    tfidf_features = tfidf_vectorizer.transform(preprocessed_data)

    # Make predictions using the loaded model
    predictions = model.predict(tfidf_features)
    print('Predictions:', predictions)

if __name__ == '__main__':
    main()
