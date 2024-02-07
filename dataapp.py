import pandas as pd
import streamlit as st
import gender_guesser.detector as gr
import plotly.express as px
import plotly.graph_objs as go
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import CountVectorizer

st.markdown('# Welcome to the 2020 Spotify music dashboard.')
data_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"

# create a button to take user to original data
st.markdown(f'<a href="{data_url}" target="_blank"><button style="color: white; background-color: #FF4B4B; border: None; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">Go to Original Data</button></a>', unsafe_allow_html=True)

# loading dataset and extracting columns of interest
df = pd.read_csv('spotify_songs.csv')  # make sure the file is in the same directory as your script

columnsExtract = ['track_name', 'track_artist', 'track_popularity', 'playlist_genre']
extactedDF = df[columnsExtract]
genreCounts = extactedDF['playlist_genre'].value_counts()

st.markdown('## Distribution of genres, song title sentiments and artist genders')

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

# add numerical sentiment score to dfCopy for aggregation
dfCopy['sentimentScore'] = dfCopy['track_name'].apply(lambda title: TextBlob(str(title)).sentiment.polarity)

# define popularity brackets
popularity_bins = [0, 25, 50, 75, 100]
popularity_labels = ['0-25%', '26-50%', '51-75%', '76-100%']
dfCopy['popularityBracket'] = pd.cut(dfCopy['track_popularity'], bins=popularity_bins, labels=popularity_labels)

# calculate average sentiment score for each genre and popularity bracket
avgSentiment = dfCopy.groupby(['playlist_genre', 'popularityBracket'])['sentimentScore'].mean().unstack()

# plotting and showing heatmap
heatmap = px.imshow(
    avgSentiment,
    labels=dict(x="Popularity Bracket", y="Genre", color="Average Sentiment"),
    x=popularity_labels,
    y=avgSentiment.index,
    title="Heatmap of Average Sentiment by Genre and Popularity"
)
st.plotly_chart(heatmap)

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

# Association Rule Mining for Playlist Names

st.markdown('## Association rule mining for playlist names')

# tokenising and creating a list of lists
playlistNames = df['playlist_name'].str.lower().str.split().tolist()

# one-hot encode the transactions - conversion of categorical information into a format that may be fed into machine learning algorithms to improve prediction accuracy.
te = TransactionEncoder()
te_ary = te.fit_transform(playlistNames)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# using Apriori algorithm to find frequent itemsets
frequentItemSets = apriori(df_encoded, min_support=0.01, use_colnames=True, max_len=2)

# generate association rules
rules = association_rules(frequentItemSets, metric="lift", min_threshold=1)

# filter rules based on a threshold for better visualisation
ruleFiltered = rules[(rules['lift'] > 1) & (rules['confidence'] > 0.5)]

# convert frozensets in 'antecedents' and 'consequents' to string for Plotly compatibility
ruleFiltered['antecedents'] = ruleFiltered['antecedents'].apply(lambda x: ', '.join(list(x)))
ruleFiltered['consequents'] = ruleFiltered['consequents'].apply(lambda x: ', '.join(list(x)))

# visualisation with Plotly on scatter graph
fig = px.scatter(ruleFiltered,
                 x='support',
                 y='confidence',
                 color='lift',
                 hover_data=['antecedents', 'consequents'],
                 title='Association Rules of Playlist Names')
st.plotly_chart(fig)

st.markdown('## Plot description and findings')
st.markdown("""
- **Support**: On the x-axis, support indicates how frequently the items in the rule appear together in the dataset. In the context of playlists, a higher support value means that the combination of words from playlist names appears more frequently.
- **Confidence**: On the y-axis, confidence is a measure of the likelihood that the consequent is found in transactions that contain the antecedent. In simpler terms, if you have a rule like rock -> classic, a higher confidence value means that if a playlist name contains the word "rock," it is very likely to also contain the word "classic." 
- **Lift**: Represented by the color scale, lift measures how much more often the antecedent and consequent of the rule occur together than we would expect if they were statistically independent. A lift value greater than 1 indicates that the presence of the antecedent increases the likelihood of the consequent occurring in a playlist name.

**Insights**
- Most rules have a relatively low support, which suggests that most word combinations from playlist names don't appear together very often.
- The confidence levels vary, with many rules showing high confidence, which means that for those specific word combinations, there is a strong relationship between the words in playlist names.   
- Lift values vary significantly, with some rules having very high lift values. This implies that for certain rules, the words are much more likely to appear together in a playlist name than by chance alone.        
""")

# N-gram analysis

st.markdown("**Analysis using N-grams:**")

playlistNames = df['playlist_name'].dropna().tolist()

# initialising CountVectorizer with bi-gram configuration
vectorise = CountVectorizer(ngram_range=(2, 2), stop_words='english')

# fitting vectoriser and transform the playlist names into a bi-gram frequency matrix
X = vectorise.fit_transform(playlistNames)

# sum up the counts of each bi-gram and convert to a DataFrame
biGrams = pd.DataFrame(X.sum(axis=0), columns=vectorise.get_feature_names_out()).T
biGrams.columns = ['Count']
biGrams = biGrams.sort_values(by='Count', ascending=False).head(20)

# plot the top 20 most frequent bi-grams in a bar graph
fig = px.bar(biGrams, x=biGrams.index, y='Count', title='Top 20 most frequent bi-grams in playlist names')
fig.update_layout(xaxis_title='Bi-grams', yaxis_title='Frequency', xaxis={'categoryorder':'total descending'})
fig.update_traces(marker_color='blue')
st.plotly_chart(fig)

st.markdown("Results from association rule mining can be compared to the results from an n-gram analysis. showing consistent results, with hip-hop being the most common pairing, followed by hard-rock")

# finding relationship between musical characteristics and popularity
st.markdown('## Relationships between musical characteristics and popularity of songs')

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

st.markdown('## Key Insights')
st.markdown("""
- **Song Duration**: There is a trend where the most popular tracks tend to have shorter durations. 
- **Loudness and Danceability**: Tracks with higher loudness and danceability metrics demonstrate better performance in the top quartile of popularity. 
- **Acousticness**: A higher degree of acousticness in the most popular songs suggests a preference for organic, natural sounds.
- **Energy, Instrumentalness, and Liveness**: Interestingly, songs with lower energy, instrumentalness, and liveness appear to dominate the top popularity bracket. This could point towards a prevailing taste for studio-produced tracks that emphasize vocal performance over instrumental solos or live concert recordings.
""")

st.markdown('## Word clouds for common words across song titles and artist names:')

# extracting song titles and artist names
allSongTitles = ' '.join(df['track_name'].fillna('').astype(str))
allArtistNames = ' '.join(df['track_artist'].fillna('').astype(str))

# generating word clouds
wordTitles = WordCloud(width=800, height=400, background_color='white').generate(allSongTitles)
wordArtists = WordCloud(width=800, height=400, background_color='white').generate(allArtistNames)

# displaying word clouds
fig_title, ax_title = plt.subplots(figsize=(10, 5))  
wordTitles = WordCloud(width=800, height=400, background_color='white').generate(allSongTitles)
ax_title.imshow(wordTitles, interpolation='bilinear')
ax_title.axis('off')
ax_title.set_title('Word Cloud for Song Titles')
st.pyplot(fig_title)  

fig_artist, ax_artist = plt.subplots(figsize=(10, 5))  
wordArtists = WordCloud(width=800, height=400, background_color='white').generate(allArtistNames)
ax_artist.imshow(wordArtists, interpolation='bilinear')
ax_artist.axis('off')
ax_artist.set_title('Word Cloud for Artist Names')
st.pyplot(fig_artist)  