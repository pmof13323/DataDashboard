import pandas as pd
import streamlit as st

# Load the dataset
df = pd.read_csv('spotify_songs.csv')  # make sure the file is in the same directory as your script

# Extract the specific columns
columns_to_extract = ['track_name', 'track_artist', 'track_popularity', 'playlist_genre']
extracted_df = df[columns_to_extract]

# Display the DataFrame in the Streamlit app
st.write(extracted_df)
