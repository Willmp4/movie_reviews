import pandas as pd 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string 
import pandas as pd
import nltk

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


def main():
    # Load the dataset into a DataFrame
    df = pd.read_csv('IMDB Dataset.csv')

    # Apply the preprocess_text function to the 'review' column
    df['review'] = df['review'].apply(preprocess_text)

if __name__ == '__main__':
    main()
