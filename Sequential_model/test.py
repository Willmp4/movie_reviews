## Sequential model
#Trained on IMDB dataset 
#https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense
from keras.regularizers import l1
import pickle
import tensorflow as tf
import re
from nltk.tokenize import word_tokenize

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



max_words = 10000  # Maximum number of unique words to consider
max_sequence_length = 250  # Maximum length of each
embedding_dim = 300  # Dimension of the GloVe word embeddings
negation_words = ['not', 'no', 'never']  # Words to be considered as negation words


def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs
    return embeddings_index


def create_embedding_matrix(embeddings_index, word_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def sentiment_to_binary(sentiment):
    if sentiment == 'positive':
        return 1
    else:
        return 0



def preprocess_text_with_negation(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Preserve negation words by appending an underscore to the following word
    negation_words = ['not', 'no', 'never']
    tokens = [tokens[i] + "_" + tokens[i+1] if tokens[i] in negation_words else tokens[i] for i in range(len(tokens) - 1)] + [tokens[-1]]
    
    # Remove non-alphanumeric characters (keeping the underscore for negation)
    tokens = [re.sub(r'[^a-z0-9_]', '', token) for token in tokens]
    
    # Remove any empty strings that may result from the previous step
    tokens = [token for token in tokens if token]
    
    # Combine the tokens back into a single string
    text = ' '.join(tokens)
    
    return text

df = pd.read_csv('IMDB Dataset.csv')

# Apply the preprocess_text_with_negation function to the 'review' column
df['review'] = df['review'].apply(preprocess_text_with_negation)

df['sentiment'] = df['sentiment'].apply(sentiment_to_binary)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Convert the text to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)


embeddings_index = load_glove_embeddings('glove.42B.300d.txt')
embedding_matrix = create_embedding_matrix(embeddings_index, word_index, embedding_dim)



from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, TimeDistributed

model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(SpatialDropout1D(0.2))

# Adding a 1D convolutional layer
model.add(Conv1D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Bidirectional(LSTM(100, dropout=0.0, recurrent_dropout=0.0, kernel_regularizer=l1(0.001), return_sequences=True)))

# Adding a second LSTM layer
model.add(Bidirectional(LSTM(50, dropout=0.0, recurrent_dropout=0.0, kernel_regularizer=l1(0.001), return_sequences=True)))

# Adding a TimeDistributed layer with a dense output
model.add(TimeDistributed(Dense(25, activation='relu')))

# Adding a GlobalMaxPooling1D layer to reduce the output to a single vector
model.add(GlobalMaxPooling1D())

# Adding a dense layer before the final output layer
model.add(Dense(25, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Display the model summary
model.summary()

# model = load_model('sentiment_analysis_model.h5')

# Create a checkpoint callback
# checkpoint = ModelCheckpoint('sentiment_analysis_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
print('Training the model...')
with open('new_tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_data=(X_test_padded, y_test))

# model = load_model('sentiment_analysis_model.h5')


scores = model.evaluate(X_test_padded, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1] * 100))
# Save the trained model
model.save('sentiment_analysis_model.h5')