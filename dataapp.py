import pandas as pd
import streamlit as st
import gender_guesser.detector as gr
import plotly.express as px
from textblob import TextBlob

st.markdown('# Welcome to the 2020 Spotify music dashboard.')
data_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"

# create a button to take user to original data
st.markdown(f'<a href="{data_url}" target="_blank"><button style="color: white; background-color: #FF4B4B; border: None; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">Go to Original Data</button></a>', unsafe_allow_html=True)

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

# detecting gender of artist based on their name
detector = gr.Detector()

def genderAnalysis(name):
    if isinstance(name, str):
        firstName = name.split()[0]
        return detector.get_gender(firstName)
    else:
        return 'unknown'


extracted_df_copy["gender"] = extracted_df_copy["track_artist"].apply(genderAnalysis)
gendCount = extracted_df_copy["gender"].value_counts()

figGender = px.pie(gendCount, values = gendCount, names = gendCount.index, title='Distribution of genders across artists')
st.write("Here's the distribution of genders from artist names")
st.plotly_chart(figGender)