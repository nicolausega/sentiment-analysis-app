#Import Library
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

#MangoDB client connection
#client = MongoClient("localhost", 27017)

#db = client["twitter_scraper"]
#collection = db["tweets"]

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://loketcom:loketdotcom@cluster.3k2noof.mongodb.net/")
db = client["twitter_scraper"]
collection = db["tweets"]

# main function
def main():

    # Sidebar menu options
    st.sidebar.image("logo.png", use_column_width=True)

    #Menus Jendela
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", 'Twitter Scraper', "Sentiment Analysis", "History"], 
            icons=['house', 'cloud-upload', 'list-task', 'gear'], menu_icon="cast", default_index=0)

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
    # Add your home page content here

def display_scraper():
    st.title("Twitter Scraper")
    st.subheader("Welcome to the Twitter Scraper page!")
    st.write("Specify the search parameters")

    # Add your scraper page content here
    # User-configurable parameters
    filename = st.text_input("File name", "tweets")
    search_keywords = st.text_input("Search keywords (separated by commas)", "loketcom,loket.com,loket dot com")
    limit = st.number_input("Limit", value=100, min_value=1)
    since_date = st.date_input("Since Date", value=date.today())
    until_date = st.date_input("Until Date", value=date.today())
    auth_token = st.text_input("Auth Token", "", type="password")

    # Button to start scraping
    if st.button("Scrape"):
        # Create a new 'data' folder if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")

        # Split search keywords into a list
        keywords_list = [keyword.strip() for keyword in search_keywords.split(",")]

        # Initialize an empty list to store the scraped data
        data_list = []

        # Scrape tweets for each keyword
        progress_bar = st.progress(0)
        for index, keyword in enumerate(keywords_list):
            # Generate the command for tweet-harvest
            keyword_query = keyword + " lang:id since:" + str(since_date) + " until:" + str(until_date)
            command = f"npx --yes tweet-harvest@latest -o data/{filename}_{keyword.replace(' ', '_')} -s '{keyword_query}' -l {limit} --token {auth_token}"

            # Run the tweet-harvest command using subprocess
            try:
                subprocess.run(command, shell=True, check=True)
                st.success(f"Tweet scraping for keyword '{keyword}' complete!")

                # Read the CSV file into a pandas DataFrame
                # Check if the file exists
                csv_file_path = os.path.join("tweets-data", "data", f"{filename}_{keyword.replace(' ', '_')}.csv")
                if not os.path.isfile(csv_file_path):
                    excel_file_path = os.path.join("tweets-data", "data", f"{filename}_{keyword.replace(' ', '_')}.xlsx")
                    if not os.path.isfile(excel_file_path):
                        st.warning("File not found.")
                        continue
                    df = pd.read_excel(excel_file_path)
                    df.to_csv(csv_file_path, index=False)

                df = pd.read_csv(file_path)

                # Convert the DataFrame to a list of dictionaries
                data = df.to_dict("records")

                # Insert the data into MongoDB
                collection.insert_many(data)

                # Append the DataFrame to the data list
                data_list.append(df)

            except subprocess.CalledProcessError as e:
                st.error(f"An error occurred while running tweet-harvest for keyword '{keyword}': {e}")

            # Update progress bar
            progress = (index + 1) / len(keywords_list)
            progress_bar.progress(progress)

        # Concatenate the scraped data into a single DataFrame
        df_combined = pd.concat(data_list, ignore_index=True)

        # Save the combined DataFrame to a CSV file
        save_to_csv(df_combined, filename)

        # Convert DataFrame to JSON
        json_data = df_combined.to_json(orient="records")

        # Download the JSON file
        download_json(json_data, filename)

        # Display the combined DataFrame
        st.subheader("Scraped Tweets (Combined)")
        st.dataframe(df_combined)

def read_from_mongodb(filename, keyword):
    # Connect to MongoDB
    client = MongoClient("mongodb+srv://loketcom:loketdotcom@cluster.3k2noof.mongodb.net/")
    db = client["twitter_scraper"]
    collection = db[filename]

    # Query data from MongoDB
    query = {"keyword": keyword}
    saved_data = list(collection.find(query))

    return data

def download_json(json_data, filename):
    b64 = base64.b64encode(json_data.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download JSON</a>'
    st.markdown(href, unsafe_allow_html=True)

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
if __name__ == '__main__':
    main()

