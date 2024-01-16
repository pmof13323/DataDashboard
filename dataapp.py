import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob

# loading dataset and extracting columns of interest
df = pd.read_csv('spotify_songs.csv')  # make sure the file is in the same directory as your script

columns_to_extract = ['track_name', 'track_artist', 'track_popularity', 'playlist_genre']
extracted_df = df[columns_to_extract]

# creating pie chart for genre distribution using plotly
genre_counts = extracted_df['playlist_genre'].value_counts()

fig_genre = px.pie(genre_counts, values=genre_counts, names=genre_counts.index, title='Distribution of Playlist Genres')
st.write("Here is a pie chart of the distribution of genres in the 2020 data:")
st.plotly_chart(fig_genre)

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

# creating pie chart for sentiment analysis using plotly
fig_sentiment = px.pie(sentimentCount, values=sentimentCount, names=sentimentCount.index, title='Distribution of Sentiments in Song Titles')
st.write("Here is a pie chart of the distribution of sentiments in song titles:")
st.plotly_chart(fig_sentiment)
