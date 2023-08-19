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
from Home import session_state
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.dates as mdates
import matplotlib.backends.backend_pdf as pdf_backend
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import xlsxwriter

# Sidebar menu options
st.sidebar.image("images/logo.png", use_column_width=True)

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
        vectorizer = joblib.load('vectorizer.joblib')

        # Get the texts for prediction from the DataFrame
        texts = df['converted_tweet'].values

        # Transform the input texts into features using the loaded vectorizer
        input_features = vectorizer.transform(texts)

        # Make predictions using the loaded SVM model
        predictions = model.predict(input_features)

        # Define the label mapping
        label_mapping = {'positif': 1, 'negatif': 0, 'netral': 2}
            
        # Map the numerical labels back to the original labels
        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        predicted_labels = np.array([reverse_label_mapping[label] for label in predictions])

        # Add a new column 'Predicted Label' to the DataFrame with the predicted labels
        df['predicted_label'] = predicted_labels

        # Define the columns you want to display
        selected_columns = ['username', 'full_text', 'predicted_label']

        # Display only the selected columns using st.write()
        #st.write(df[selected_columns])

        # Function to display the DataFrame table
        def display_dataframe_table(df):
            st.subheader("Predicted Data Table:")
            st.dataframe(df[selected_columns])
            
            # Exploratory Data Analysis
            st.title("Exploratory Data Analysis")

        # Function to display sentiment conclusion
        def display_sentiment_conclusion(df):
            sentiment_counts = df['predicted_label'].value_counts()
            st.subheader("Sentiment Conclusion:")
            st.write("Total Tweets:", df.shape[0])
            st.write("Positive Tweets:", sentiment_counts.get('positif', 0))
            st.write("Negative Tweets:", sentiment_counts.get('negatif', 0))

        # Function to display pie chart of predicted label
        def display_pie_chart(df):
            sentiment_counts = df['predicted_label'].value_counts()
            sentiment_counts_filtered = sentiment_counts[sentiment_counts.index.isin(['positif', 'negatif'])]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(sentiment_counts_filtered, labels=sentiment_counts_filtered.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title('Predicted Label Distribution')
            st.subheader("Pie Chart of Predicted Label:")
            st.pyplot(fig)

        # Function to display top words from converted_tweet for each label (negatif & positif)
        def display_top_words(df):
            vectorizer = CountVectorizer(max_features=10)
            for label in ['negatif', 'positif']:
                subset_df = df[df['predicted_label'] == label]
                converted_tweets = subset_df['converted_tweet'].tolist()
                if converted_tweets:
                    word_counts = vectorizer.fit_transform(converted_tweets).sum(axis=0)
                    words = vectorizer.get_feature_names_out()
                    word_counts = word_counts.tolist()[0]
                    word_freq_df = pd.DataFrame({'Word': words, 'Frequency': word_counts})
                    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
                    st.subheader(f"Top Words for {label.capitalize()} Tweets:")
                    st.dataframe(word_freq_df)

        # Function to display n-grams for each label (negatif & positif)
        def display_ngrams(df):
            vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
            for label in ['negatif', 'positif']:
                subset_df = df[df['predicted_label'] == label]
                converted_tweets = subset_df['converted_tweet'].tolist()
                if converted_tweets:
                    ngram_counts = vectorizer.fit_transform(converted_tweets).sum(axis=0)
                    ngrams = vectorizer.get_feature_names_out()
                    ngram_counts = ngram_counts.tolist()[0]
                    ngram_freq_df = pd.DataFrame({'N-gram': ngrams, 'Frequency': ngram_counts})
                    ngram_freq_df = ngram_freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
                    st.subheader(f"N-grams for {label.capitalize()} Tweets:")
                    st.dataframe(ngram_freq_df)
        
        # Function to display sentiment graph over time
        def display_sentiment_graph(df):
            # Make a copy of the DataFrame before modifying it
            df_copy = df.copy()

            # Convert 'created_at' to datetime if needed
            df_copy['created_at'] = pd.to_datetime(df_copy['created_at'], format='%a %b %d %H:%M:%S %z %Y')

            # Set 'created_at' as the DataFrame index
            df_copy.set_index('created_at', inplace=True)

            # Calculate the daily count of predicted labels
            sentiment_counts = df_copy['predicted_label'].resample('D').count()

            # Calculate the positive and negative counts for the sentiment graph
            sentiment_graph_data = pd.DataFrame(sentiment_counts, columns=['count'])
            positive_counts = df_copy[df_copy['predicted_label'] == 'positif']['predicted_label'].resample('D').count()
            negative_counts = df_copy[df_copy['predicted_label'] == 'negatif']['predicted_label'].resample('D').count()

            # Fill NaN values with 0 and reindex to align with the index of sentiment_counts
            sentiment_graph_data['positive'] = positive_counts.reindex(sentiment_counts.index, fill_value=0)
            sentiment_graph_data['negative'] = negative_counts.reindex(sentiment_counts.index, fill_value=0)

            # Plot the sentiment graph
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sentiment_graph_data.index, sentiment_graph_data['positive'], label='Positive', color='green')
            ax.plot(sentiment_graph_data.index, sentiment_graph_data['negative'], label='Negative', color='red')
            ax.bar(sentiment_graph_data.index, sentiment_graph_data['count'], alpha=0.3, label='Total', color='blue')
            ax.legend()
            ax.set_xlabel('Date')
            ax.set_ylabel('Count')
            ax.set_title('Sentiment Graph over Time')

            # Format the x-axis to display just the date without the hour and rotate the labels
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.xticks(rotation=45)

            st.subheader("Sentiment Graph:")
            st.pyplot(fig)


        # Function to display word clouds for overall data, positif tweets, and negatif tweets
        def display_wordcloud(df):
            # Word cloud for positif tweets
            positif_tweets = ' '.join(df[df['predicted_label'] == 'positif']['converted_tweet'])
            if positif_tweets:
                wordcloud_positif = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(positif_tweets)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_positif, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud for Positif Tweets')
                st.subheader("Word Cloud for Positif Tweets:")
                st.pyplot(fig)
            else:
                st.warning("No positif tweets to generate word cloud.")

            # Word cloud for negatif tweets
            negatif_tweets = ' '.join(df[df['predicted_label'] == 'negatif']['converted_tweet'])
            if negatif_tweets:
                wordcloud_negatif = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(negatif_tweets)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_negatif, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud for Negatif Tweets')
                st.subheader("Word Cloud for Negatif Tweets:")
                st.pyplot(fig)
            else:
                st.warning("No negatif tweets to generate word cloud.")

            # Word cloud for overall data
            all_tweets = ' '.join(df['converted_tweet'])
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_tweets)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud for Overall Data')
            st.subheader("Word Cloud for Overall Data:")
            st.pyplot(fig)
        
        #Menampilkan EDA
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #display_sentiment_conclusion(df)
        #display_pie_chart(df)
        #display_sentiment_graph(df)
        #display_top_words(df)
        #display_ngrams(df)
        #display_wordcloud(df)

        # Assuming you have already loaded and processed your dataframe 'df'
        df = session_state["df"]

        # Display the DataFrame in Streamlit
        st.subheader("Predicted Data Table:")
        st.dataframe(df[selected_columns])

        # Offer the Excel file for download
        excel_file = BytesIO()
        df[selected_columns].to_excel(excel_file, index=False, engine='xlsxwriter')
        excel_file.seek(0)
        b64 = base64.b64encode(excel_file.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predicted_data.xlsx">Click here to download the predicted data as Excel file</a>'
        st.markdown(href, unsafe_allow_html=True)

        #
        # Function to save all the visualizations as a PDF
        def save_visualizations_as_pdf(df):

            # Save the visualizations as a PDF file
            pdf_bytes = BytesIO()
            with PdfPages(pdf_bytes) as pdf:
                display_sentiment_conclusion(df)
                display_pie_chart(df)
                display_sentiment_graph(df)
                display_top_words(df)
                display_ngrams(df)
                display_wordcloud(df)
        
            # Offer the PDF for download
            st.subheader("Download Visualizations as PDF:")
            st.download_button(label="Download PDF", data=pdf_bytes.getvalue(), file_name="sentiment_analysis_visualizations.pdf", mime="application/pdf")

        # Offer the option to download all visualizations as a PDF
        save_visualizations_as_pdf(df)
