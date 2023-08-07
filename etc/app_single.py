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


def display_scraper():
    st.title("Twitter Scraper")
    st.subheader("Welcome to the Twitter Scraper page!")
    st.write("Specify the search parameters")

    # Add your scraper page content here
    # User-configurable parameters
    filename = st.text_input("File name", "tweets")
    search_keyword = st.text_input("Search keyword", "loketcom")
    limit = st.number_input("Limit", value=100, min_value=1)
    since_date = st.date_input("Since Date", value=date.today())
    until_date = st.date_input("Until Date", value=date.today())
    auth_token = st.text_input("Auth Token", "", type="password")

    # Button to start scraping
    if st.button("Scrape"):
        # Create a new 'data' folder if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")

        # Generate the command for tweet-harvest
        keyword_query = search_keyword + " lang:id since:" + str(since_date) + " until:" + str(until_date)
        command = f"npx --yes tweet-harvest@latest -o data/{filename}_{search_keyword.replace(' ', '_')} -s '{keyword_query}' -l {limit} --token {auth_token}"

        # Run the tweet-harvest command using subprocess
        try:
            subprocess.run(command, shell=True, check=True)
            st.success(f"Tweet scraping for keyword '{search_keyword}' complete!")

            # Save the file to Google Drive folder
            credentials = Credentials.from_authorized_user_file('credentials.json')
            drive_service = build('drive', 'v3', credentials=credentials)

            csv_file_path = os.path.join("tweets-data", "data", f"{filename}_{search_keyword.replace(' ', '_')}.csv")
            csv_file_name = f"{filename}_{search_keyword.replace(' ', '_')}.csv"
            folder_id = "1TXug91mqFvE7e3lZcWT59Ke9svcDnpK7"  # Replace with the specific folder ID
            file_metadata = {
                'name': csv_file_name,
                'parents': [folder_id],
            }
            media = MediaFileUpload(csv_file_path, mimetype='text/csv')
            file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            st.success(f"File uploaded to Google Drive with ID: {file.get('id')}")

            # Read the CSV file from Google Drive (Optional, if you want to display the DataFrame)
            df = pd.read_csv(csv_file_path)

            # Provide a link to download the CSV file
            download_url = get_csv_download_url(csv_file_path, csv_file_name)
            st.markdown(download_url, unsafe_allow_html=True)

        except subprocess.CalledProcessError as e:
            st.error(f"An error occurred while running tweet-harvest for keyword '{search_keyword}': {e}")

def get_csv_download_url(csv_file_path, csv_file_name):
    with open(csv_file_path, 'rb') as f:
        csv_file_content = f.read()
    csv_base64 = base64.b64encode(csv_file_content).decode()
    csv_href = f'<a href="data:text/csv;base64,{csv_base64}" download="{csv_file_name}">Click here to download the CSV file</a>'
    return csv_href


###### End of scrape page
def display_sentiment_analysis():
    st.title("Sentiment Analysis")
    st.write("Welcome to the Sentiment Analysis page!")
    # Add your sentiment analysis page content here


def display_history():
    st.title("History")
    st.write("Welcome to the History page!")
    # Add your history page content here


# Call the main function
if __name__ == "__main__":
    main()