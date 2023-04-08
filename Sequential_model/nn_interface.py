import numpy as np
from keras.utils import pad_sequences
from keras.models import load_model
import pickle
# Load the trained sentiment analysis model
model = load_model('sentiment_analysis_model.h5')

# Maximum length of each sequence
max_sequence_length = 250
negation_words = ['not', 'no', 'never']  # Words to be considered as negation words

# Words to be considered as negation words
def preprocess_text_with_negation(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Replace any negation words with "not_" to preserve their meaning in the model
    for negation_word in negation_words:
        text = text.replace(negation_word, "not_" + negation_word)
    
    # Remove any non-alphanumeric characters
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    
    return text

# Function to convert user input to a padded sequence of fixed length
def preprocess_input(input_text, tokenizer):
    input_text = preprocess_text_with_negation(input_text)
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