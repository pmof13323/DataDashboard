import pandas as pd
import streamlit as st
import gender_guesser.detector as gr
import plotly.express as px
import plotly.graph_objs as go
from textblob import TextBlob

st.markdown('# Welcome to the 2020 Spotify music dashboard.')
data_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"

# create a button to take user to original data
st.markdown(f'<a href="{data_url}" target="_blank"><button style="color: white; background-color: #FF4B4B; border: None; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">Go to Original Data</button></a>', unsafe_allow_html=True)

# loading dataset and extracting columns of interest
df = pd.read_csv('spotify_songs.csv')  # make sure the file is in the same directory as your script

columnsExtract = ['track_name', 'track_artist', 'track_popularity', 'playlist_genre']
extactedDF = df[columnsExtract]
genreCounts = extactedDF['playlist_genre'].value_counts()

# creating pie chart for genre distribution using plotly
figGenre = px.pie(genreCounts, values=genreCounts, names=genreCounts.index, title='Distribution of Playlist Genres')
st.plotly_chart(figGenre)

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

dfCopy = extactedDF.copy()
dfCopy["sentiment"] = dfCopy["track_name"].apply(analysingSentiment)
sentimentCount = dfCopy["sentiment"].value_counts()

# creating pie chart for sentiment analysis using plotly
figSentiment = px.pie(sentimentCount, values=sentimentCount, names=sentimentCount.index, title='Distribution of Sentiments in Song Titles')
st.plotly_chart(figSentiment)

# detecting gender of artist based on their name
detector = gr.Detector()

def genderAnalysis(name):
    if isinstance(name, str):
        firstName = name.split()[0]
        return detector.get_gender(firstName)
    else:
        return 'unknown'

dfCopy["gender"] = dfCopy["track_artist"].apply(genderAnalysis)
gendCount = dfCopy["gender"].value_counts()

# pie chart to display gender distribution across artists
figGender = px.pie(gendCount, values = gendCount, names = gendCount.index, title='Distribution of genders across artists')
st.plotly_chart(figGender)

# finding relationship between musical characteristics and popularity

# splitting into four data frames based on popularity
sevenFifth = df[(df['track_popularity'] >= 75) & (df['track_popularity'] <= 100)]
fiftieth = df[(df['track_popularity'] >= 50) & (df['track_popularity'] <= 74)]
twentyFifth = df[(df['track_popularity'] >= 25) & (df['track_popularity'] <= 49)]
zeroth = df[(df['track_popularity'] >= 0) & (df['track_popularity'] <= 24)]

# creating new dataframe to hold averages for characteristics
averageCharacteristics = pd.DataFrame({
    'popularity': ['75-100', '50-74', '25-49', '0-24'],
    'danceability': [sevenFifth['danceability'].mean(), fiftieth['danceability'].mean(), twentyFifth['danceability'].mean(), zeroth['danceability'].mean()],
    'energy': [sevenFifth['energy'].mean(), fiftieth['energy'].mean(), twentyFifth['energy'].mean(), zeroth['energy'].mean()],
    'key': [sevenFifth['key'].mean(), fiftieth['key'].mean(), twentyFifth['key'].mean(), zeroth['key'].mean()],
    'loudness': [sevenFifth['loudness'].mean(), fiftieth['loudness'].mean(), twentyFifth['loudness'].mean(), zeroth['loudness'].mean()],
    'mode': [sevenFifth['mode'].mean(), fiftieth['mode'].mean(), twentyFifth['mode'].mean(), zeroth['mode'].mean()],
    'speechiness': [sevenFifth['speechiness'].mean(), fiftieth['speechiness'].mean(), twentyFifth['speechiness'].mean(), zeroth['speechiness'].mean()],
    'acousticness': [sevenFifth['acousticness'].mean(), fiftieth['acousticness'].mean(), twentyFifth['acousticness'].mean(), zeroth['acousticness'].mean()],
    'instrumentalness': [sevenFifth['instrumentalness'].mean(), fiftieth['instrumentalness'].mean(), twentyFifth['instrumentalness'].mean(), zeroth['instrumentalness'].mean()],
    'liveness': [sevenFifth['liveness'].mean(), fiftieth['liveness'].mean(), twentyFifth['liveness'].mean(), zeroth['liveness'].mean()],
    'valence': [sevenFifth['valence'].mean(), fiftieth['valence'].mean(), twentyFifth['valence'].mean(), zeroth['valence'].mean()],
    'tempo': [(sevenFifth['tempo'].mean())/100, (fiftieth['tempo'].mean())/100, (twentyFifth['tempo'].mean())/100, (zeroth['tempo'].mean())/100],
    'duration_ms': [(sevenFifth['duration_ms'].mean())/60000, (fiftieth['duration_ms'].mean())/60000, (twentyFifth['duration_ms'].mean())/60000, (zeroth['duration_ms'].mean())/60000]
})

# normalising loudness and values that are outside of the desired range 0 < value < 1
def normaliseLoudness(loudness):
    return (loudness - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())

averageCharacteristics['loudness'] = averageCharacteristics['loudness'].apply(normaliseLoudness)

def normalise(series):
    maxValue = series.abs().max()
    if maxValue > 1:
        return series / maxValue
    return series

characteristics = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

for char in characteristics:
    if char != 'loudness': 
        averageCharacteristics[char] = normalise(averageCharacteristics[char])

# creating plot and setting graph characteristics
fig = go.Figure()
colours = px.colors.qualitative.Plotly 

for index, char in enumerate(characteristics):
    fig.add_trace(go.Scatter(
        x=averageCharacteristics['popularity'],
        y=averageCharacteristics[char],
        mode='lines+markers',
        name=char,
        line=dict(color=colours[index % len(colours)]) 
    ))

fig.update_layout(
    title='Relationship Between Popularity Percentiles and Musical Characteristics',
    xaxis_title='Popularity Percentile',
        yaxis=dict(
        title='Characteristic Value',
        range=[0, 1.1], 
        dtick=0.05 
    ),
    legend_title='Musical Characteristics',
    template='plotly_white', 
    height=1000 
)

st.plotly_chart(fig, use_container_width=True)