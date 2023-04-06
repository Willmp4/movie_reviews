import numpy as np
from keras.utils import pad_sequences
from keras.models import load_model
from train_model import preprocess_text
import pickle
# Load the trained sentiment analysis model
model = load_model('sentiment_analysis_model.h5')

# Maximum length of each sequence
max_sequence_length = 250

# Function to convert user input to a padded sequence of fixed length
def preprocess_input(input_text, tokenizer):
    input_text = preprocess_text(input_text)
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)
    return input_padded

# Function to predict the sentiment of the user input
def predict_sentiment(input_text, tokenizer, model):
    input_padded = preprocess_input(input_text, tokenizer)
    prediction = model.predict(input_padded)[0][0]
    if prediction >= 0.5:
        return 'positive'
    else:
        return 'negative'

# Load the Tokenizer used during training
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Get user input
user_input = input('Enter a statement: ')
    
# Predict the sentiment of the user input
prediction = predict_sentiment(user_input, tokenizer, model)

print(f'The sentiment of the statement "{user_input}" is {prediction}.')
