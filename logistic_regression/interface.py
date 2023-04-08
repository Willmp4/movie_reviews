import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from train_model import preprocess_text

# Define the file names for the model and vectorizer
tfidf_vectorizer_file = 'tfidf_vectorizer.pkl'
model_file = 'sentiment_analysis_model.pkl'

# Load the TfidfVectorizer and model
tfidf_vectorizer = joblib.load(tfidf_vectorizer_file)
model = joblib.load(model_file)

# Define a function to get user input and make predictions
def predict_sentiment():
    # Get input from user
    input_text = input('Enter some text: ')

    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)

    # Vectorize the preprocessed text
    input_vector = tfidf_vectorizer.transform([preprocessed_text])

    # Make prediction using the model
    prediction = model.predict(input_vector)[0]

    # Print the prediction
    if prediction == 'positive':
        print('The text has a positive sentiment.')
    else:
        print('The text has a negative sentiment.')

if __name__ == '__main__':
    predict_sentiment()
