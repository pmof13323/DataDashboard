import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob

# loading dataset and extracting columns of interest
df = pd.read_csv('spotify_songs.csv')  # make sure the file is in the same directory as your script

columns_to_extract = ['track_name', 'track_artist', 'track_popularity', 'playlist_genre']
extracted_df = df[columns_to_extract]

# creating pie chart for genre distribution
genre_counts = extracted_df['playlist_genre'].value_counts()

st.write("Here is a pie chart of the distribution of genres in the 2020 data:")

plt.figure(figsize=(10, 6))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Distribution of Playlist Genres')

st.pyplot(plt)

# sentiment analysis of whether song titles were positive or negative on average
def analysingSentiment(title):
    title = str(title)
    analysis = TextBlob(title)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

extracted_df_copy = extracted_df.copy()
extracted_df_copy["sentiment"] = extracted_df_copy["track_name"].apply(analysingSentiment)
sentimentCount = extracted_df_copy["sentiment"].value_counts()

st.write("Here is a pie chart of the distribution of sentiments in song titles:")

plt.figure(figsize=(10, 6))
plt.pie(sentimentCount, labels=sentimentCount.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal') 
plt.title('Distribution of Sentiments in Song Titles')
st.pyplot(plt)
