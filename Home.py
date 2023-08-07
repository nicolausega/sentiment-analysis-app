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

    st.title("Twitter Sentiment Analysis App")
    st.subheader("This app is dedicated for scraping and sentiment analysis for Loket.com")
    st.write("Please head to the Twitter Scraper page to begin.")
    st.image("images/loket.gif", use_column_width=True)

# Call the main function
if __name__ == "__main__":
    main()