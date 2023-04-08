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
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

max_words = 10000  # Maximum number of unique words to consider
max_sequence_length = 250  # Maximum length of each
embedding_dim = 300  # Dimension of the GloVe word embeddings
negation_words = ['not', 'no', 'never']  # Words to be considered as negation words


# Load GloVe embeddings
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
    
    # Replace any negation words with "not_" to preserve their meaning in the model
    for negation_word in negation_words:
        text = text.replace(negation_word, "not_" + negation_word)
    
    # Remove any non-alphanumeric characters
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    
    return text


# Load the dataset into a DataFrame
df = pd.read_csv('IMDB Dataset.csv')

# Apply the preprocess_text_with_negation function to the 'review' column
df['review'] = df['review'].apply(preprocess_text_with_negation)

df['sentiment'] = df['sentiment'].apply(sentiment_to_binary)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the text
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=250)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=250)

# Convert the tokenized data into a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(4)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(4)

# Fine-tune the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print('Training the model...')
model.fit(train_dataset, epochs=3, batch_size=4, validation_data=test_dataset)

# Evaluate the model
y_pred_logits = model.predict(test_dataset, batch_size=4)
y_pred = np.argmax(y_pred_logits.logits, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))

# Save the fine-tuned model
model.save_pretrained('sentiment_analysis_bert/')