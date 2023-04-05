import pandas as pd 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string 

df = pd.read_csv('IMDB Dataset.csv')
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




df['review'] = df['review'].apply(preprocess_text)
