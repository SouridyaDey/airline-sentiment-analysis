from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import nltk
import emoji
import re
import string
from bs4 import BeautifulSoup
import html
from contractions import fix as fix_contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Flask App
app = Flask(__name__)

# Load files
model = pickle.load(open('stacked_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Chatwords mapping
chat_words_map = {
    "u": "you", "ur": "your", "luv": "love", "gr8": "great",
    "b4": "before", "omg": "oh my god", "idk": "i do not know",
    "im": "i am", "thx": "thanks", "wanna": "want to",
    "gonna": "going to", "cuz": "because", 'fyi': 'for your information'
}

def expand_chatwords(text):
    return " ".join([chat_words_map.get(w, w) for w in text.split()])

tokenizer = RegexpTokenizer(r'\w+')

def clean_text(text):
 # Lowercase
    text = text.lower()
    
    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    words = tokenizer.tokenize(text)

    # Remove stopwords but keep negations like "not", "no"
    meaningful_words = [stemmer.stem(word) for word in words if word not in stop_words or word in ["not", "no"]]

    return ' '.join(meaningful_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['tweet']

    # Create a dummy input row
    input_df = pd.DataFrame({
        'text': [input_text],
        'retweet_count': [0],  # or ask the user
        'tweet_hour': [12],    # or use datetime.now().hour
        'negativereason': ['Not available'],
        'airline': ['United'],  # default airline
        'user_timezone': ['Not available']
    })

    # Create derived features
    input_df['num_characters'] = input_df['text'].apply(len)
    input_df['num_words'] = input_df['text'].apply(lambda x: len(tokenizer.tokenize(x)))
    input_df['num_sentences'] = input_df['text'].apply(lambda x: len(re.findall(r'[.!?]+', x)))
    input_df['transformed_text'] = input_df['text'].apply(clean_text)
    input_df.drop(columns=['text'], inplace=True)

    # Transform
    input_processed = preprocessor.transform(input_df).toarray()

    # Predict
    prediction = model.predict(input_processed)
    sentiment = encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
