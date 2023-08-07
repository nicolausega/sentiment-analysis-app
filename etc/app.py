# Import Library
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import snscrape.modules.twitter as sntwitter
import pymongo
from pymongo import MongoClient
from PIL import Image
from datetime import date
import json
import os
import subprocess
import base64
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import base64
import re
import emoji
import unicodedata
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Define session state
# Define session state
class SessionState:
    def __init__(self):
        self._state = {}

    def __getitem__(self, key):
        return self._state.get(key, None)

    def __setitem__(self, key, value):
        self._state[key] = value

# Create an instance of the session state
session_state = SessionState()

# main function
def main():

    # Sidebar menu options
    st.sidebar.image("images/logo.png", use_column_width=True)

    # Menus Jendela
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Twitter Scraper", "Sentiment Analysis", "History"],
            icons=["house", "cloud-upload", "list-task", "gear"],
            menu_icon="cast",
            default_index=0,
        )

    # Display content based on selected menu
    if selected == "Home":
        display_home()
    elif selected == "Twitter Scraper":
        display_scraper()
    elif selected == "Sentiment Analysis":
        display_sentiment_analysis()
    elif selected == "History":
        display_history()


# Functions to display content for each menu option
def display_home():
    st.title("Twitter Sentiment Analysis App")
    st.subheader("This app is dedicated for scraping and sentiment analysis for Loket.com")
    st.write("Please head to the Twitter Scraper page to begin.")
    st.image("images/loket.gif", use_column_width=True)
    # Add your home page content here


# Define df at the module level
df = pd.DataFrame()

def display_scraper():
    st.title("Twitter Scraper")
    st.subheader("Welcome to the Twitter Scraper page!")
    st.write("Click the scrape button to open Scraper App")

    colab_link = "https://colab.research.google.com/drive/1wP62GYVaWw6OGRlrYI91d35KpNFO6P_B#scrollTo=1U_V0LNSmGIy"

    if st.button("Open Colab Link"):
        # Open the link in a new tab
        st.markdown(f'<a href="{colab_link}" target="_blank">Click here to open the Colab link</a>', unsafe_allow_html=True)

    st.write("Upload the data file scraped successfully")

    # Add your scraper page content here
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.write(df)

        # Preprocessing steps
        # 1. Remove duplicates based on 'id_str' column and keep the first occurrence
        df.drop_duplicates(subset='id_str', keep='first', inplace=True)

        # 2. Remove unused columns
        df.drop(['id_str', 'quote_count', 'reply_count', 'retweet_count',
                 'favorite_count', 'lang', 'user_id_str', 'conversation_id_str',
                 'tweet_url'], axis=1, inplace=True)

        # 3. Remove mentions, hashtags, and links
        mention_hashtag_pattern = r'^@\w+|#\w+'
        link_pattern = r'http\S+|www.\S+|pic.twitter.com/\S+'
        combined_pattern = rf'{mention_hashtag_pattern}|{link_pattern}'

        df['converted_tweet'] = df['full_text'].astype(str).apply(lambda x: re.sub(combined_pattern, '', x, flags=re.MULTILINE).strip())
        
        # 4. Define a function to convert emojis into English textual descriptions
        # Read the CSV file for mapping English to Bahasa Indonesia
        df_map = pd.read_csv("kamus/emoji dictionary indo.csv")
        
        # Create a mapping dictionary for English words to Bahasa Indonesia phrases
        mapping_dict = dict(zip(df_map['Name'], df_map['Name indo']))

        def convert_emojis_to_words(text):
            demojized_text = emoji.demojize(text, delimiters=(" ", ""))  # Add spaces between emojis
            return demojized_text.replace(":", "").replace("_", " ").strip()

        # Define a function to replace English words/phrases with Bahasa Indonesia phrases using the mapping dictionary
        def replace_with_bahasa(text):
            # Implementation of replace_with_bahasa function
            words = text.split()
            translated_words = []
            i = 0
            while i < len(words):
                found = False
                for j in range(len(words), i, -1):
                    phrase = " ".join(words[i:j])
                    if phrase in mapping_dict:
                        translated_words.append(mapping_dict[phrase])
                        i = j
                        found = True
                        break
                if not found:
                    translated_words.append(words[i])
                    i += 1

            translated_text = " ".join(translated_words)
            return translated_text

        # Apply emoji conversion and English to Bahasa Indonesia translation
        df['converted_tweet'] = df['converted_tweet'].apply(lambda x: replace_with_bahasa(convert_emojis_to_words(x)))

        # 5. Define a function to remove punctuation
        def remove_punctuation(text):
            # Remove punctuation using string.punctuation
            text = ''.join([char for char in text if char not in string.punctuation])
            # Remove additional punctuation using regex
            text = re.sub(r'[^\w\s]', '', text)
            return text

        # Apply punctuation removal
        df['converted_tweet'] = df['converted_tweet'].apply(remove_punctuation)

        # 6. Define a function to remove numbers or digits
        def remove_numbers(text):
            text = re.sub(r'\d+', '', text)
            return text

        # Apply number removal
        df['converted_tweet'] = df['converted_tweet'].apply(remove_numbers)

        # 7. Normalization words
        # Define a function to normalize words
        def normalize_text(text):
            # Lowercase the text
            text = text.lower()
            # Remove diacritics using the 'NFKD' normalization form
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

            # Normalize repeated characters
            normalized_text = ""
            prev_char = None
            for char in text:
                if char != prev_char:
                    normalized_text += char
                prev_char = char

            return normalized_text

        # Apply word normalization to the 'converted_tweet' column
        df['converted_tweet'] = df['converted_tweet'].apply(normalize_text)

        # 8. Load the dataset for slang word conversion
        df_slang = pd.read_csv("kamus/slang words.csv")

        # Replace missing values in 'formal' column with an empty string
        df_slang['formal'].fillna('', inplace=True)

        # Create a mapping dictionary from the dataset
        slang_mapping = dict(zip(df_slang['slang'], df_slang['formal']))

        # Define the convert_slang_words function
        def convert_slang_words(text):
            words = text.split()
            converted_words = []
            for word in words:
                if isinstance(word, str):  # Check if it's a string
                    if word in slang_mapping:
                        converted_words.append(slang_mapping[word])
                    else:
                        converted_words.append(word)
                else:
                    converted_words.append(str(word))  # Convert to string if it's not already
            converted_text = ' '.join(converted_words)
            return converted_text

        # Apply slang word conversion
        df['converted_tweet'] = df['converted_tweet'].apply(convert_slang_words)

        # 9. Apply stemming using the Indonesian stemmer from Sastrawi
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['converted_tweet'] = df['converted_tweet'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

        # 10. Read the stopwords from the "stopwords.txt" file
        with open("kamus/stopwords.txt", 'r') as file:
            stopwords_indonesia = file.read().splitlines()

        # Define a function to remove stopwords
        def remove_stopwords(text):
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in stopwords_indonesia]
            filtered_text = ' '.join(filtered_words)
            return filtered_text

        # Apply stopword removal
        df['converted_tweet'] = df['converted_tweet'].apply(remove_stopwords)

        # 11. Lowercasing and stripping
        df['converted_tweet'] = df['converted_tweet'].str.strip()
        
        # 12. Remove rows with empty or NaN data in the 'converted_tweet' column
        df.dropna(subset=['converted_tweet'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        st.subheader("Preprocessed Data")
        st.write(df)

        # Provide a link to download the preprocessed CSV file
        csv_file = df.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">Click here to download the preprocessed CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)

        session_state["df"] = df

###### End of scrape page
def display_sentiment_analysis():
    st.title("Sentiment Analysis")
    st.write("Welcome to the Sentiment Analysis page!")
    # Add your sentiment analysis page content here

    # Check if df is empty (i.e., data has been uploaded and preprocessed)
    df = session_state["df"]
    if df is None or df.empty:
        st.warning("Please upload and preprocess data first on the Twitter Scraper page.")
    else:
        if st.button("Predict Data"):
            # Load the SVM model using joblib
            model = joblib.load('svm_model.joblib')

            # Load the TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=15000)

            # Get the texts for prediction from the DataFrame
            texts = df['converted_tweet'].values

            # Transform the input texts into features using the loaded vectorizer
            input_features = vectorizer.transform(texts)

            # Make predictions using the loaded SVM model
            predictions = model.predict(input_features)

            # Define the label mapping
            label_mapping = {'positif': 1, 'negatif': 0, 'netral': 2}
            
            # Map the numerical labels back to the original labels
            predicted_labels = [k for k, v in label_mapping.items() if v == predictions]

            # Add a new column 'Predicted Label' to the DataFrame with the predicted labels
            df['Predicted Label'] = predicted_labels

            # Display the DataFrame with the predicted labels
            st.dataframe(df)

def display_history():
    st.title("History")
    st.write("Welcome to the History page!")
    # Add your history page content here


# Call the main function
if __name__ == "__main__":
    main()