import pandas as pd

file_path = 'tweets-data/tweets-data/tweets_loketcom.csv'

try:
    df = pd.read_csv(file_path, delimiter=';')
    print(df.head())  # Print the first few rows of the DataFrame
    print(df.info())  # Display information about the DataFrame
except Exception as e:
    print(f"An error occurred while reading the file: {e}")