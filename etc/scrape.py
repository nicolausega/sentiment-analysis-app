import subprocess
import pandas as pd
from datetime import date
from datetime import datetime
import os

# User-configurable parameters
filename = "tweets"
search_keywords = "loketcom,loket.com,loket dot com"
limit = 100
since_date = date.today()
until_date = date.today()

# Split search keywords into a list
keywords_list = [keyword.strip() for keyword in search_keywords.split(",")]

# Generate keyword queries
keyword_queries = [f"{keyword} lang:id since:{since_date} until:{until_date}" for keyword in keywords_list]

# Initialize an empty list to store the scraped data
data_list = []

# Scrape tweets for each keyword
for index, keyword_query in enumerate(keyword_queries):
    # Generate the command for tweet-harvest
    command = f"npx --yes tweet-harvest@latest -o tweets-data/{filename}_{keywords_list[index].replace(' ', '_')} -s '{keyword_query}' -l {limit} --token"

    # Run the tweet-harvest command using subprocess
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Tweet scraping for keyword '{keywords_list[index]}' complete!")

        # Read the CSV file into a pandas DataFrame
        file_path = os.path.join("tweets-data", "tweets-data", f"{filename}_{keywords_list[index].replace(' ', '_')}.csv")
        if not os.path.isfile(file_path):
            print("File not found.")
            continue

        df = pd.read_csv(file_path, delimiter=';')

        # Convert since_date and until_date to datetime objects
        since_date = datetime.combine(since_date, datetime.min.time())
        until_date = datetime.combine(until_date, datetime.max.time())

        # Filter the DataFrame by lang:id and date range
        df = df[df['lang'] == 'id']
        df['created_at'] = pd.to_datetime(df['created_at'], format="%a %b %d %H:%M:%S %z %Y")
        df = df[(df['created_at'] >= since_date) & (df['created_at'] <= until_date)]

        # Save the filtered DataFrame to a CSV file
        csv_file_path = os.path.join("tweets-data", "data", f"{filename}_{keywords_list[index].replace(' ', '_')}.csv")
        df.to_csv(csv_file_path, index=False)

        # Append the DataFrame to the data list
        data_list.append(df)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running tweet-harvest for keyword '{keywords_list[index]}': {e}")

# Concatenate the scraped data into a single DataFrame
df_combined = pd.concat(data_list, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_file_path = os.path.join("tweets-data", f"{filename}_combined.csv")
df_combined.to_csv(combined_file_path, index=False)

# Display the combined DataFrame
print("Combined Data:")
print(df_combined)
