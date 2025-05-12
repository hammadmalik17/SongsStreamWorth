Mission Objective:

Build a machine learning model that predicts the "Hit Potential" of a song using its metadata (danceability, energy, tempo, etc.

Step | Task | Notes
1 | Get a dataset | Find a song dataset with danceability, energy, tempo, popularity, etc.
2 | EDA (Exploratory Data Analysis) | Understand distributions, correlations, spot patterns.
3 | Feature Engineering | Maybe combine or transform features creatively.
4 | Model Building | Train ML models (Logistic Regression / Random Forest first).
5 | Evaluation | Accuracy, Precision, Recall, ROC Curve.
6 | Mini Deployment (Optional) | Maybe a Streamlit app where you enter song stats and get prediction.


Dataset in use --
<!-- import kagglehub

# Download latest version
path = kagglehub.dataset_download("nelgiriyewithana/top-spotify-songs-2023")

print("Path to dataset files:", path) -->

Description 

This dataset contains a comprehensive list of the most famous songs of 2023 as listed on Spotify. The dataset offers a wealth of features beyond what is typically available in similar datasets. It provides insights into each song's attributes, popularity, and presence on various music platforms. The dataset includes information such as track name, artist(s) name, release date, Spotify playlists and charts, streaming statistics, Apple Music presence, Deezer presence, Shazam charts, and various audio features.

Key Features:
    track_name: Name of the song
    artist(s)_name: Name of the artist(s) of the song
    artist_count: Number of artists contributing to the song
    released_year: Year when the song was released
    released_month: Month when the song was released
    released_day: Day of the month when the song was released
    
    
        in_spotify_playlists: Number of Spotify playlists the song is included in
        in_spotify_charts: Presence and rank of the song on Spotify charts

    Streams: Total number of streams on Spotify
        in_apple_playlists: Number of Apple Music playlists the song is included in
        in_apple_charts: Presence and rank of the song on Apple Music charts
        in_deezer_playlists: Number of Deezer playlists the song is included in
        in_deezer_charts: Presence and rank of the song on Deezer charts
        in_shazam_charts: Presence and rank of the song on Shazam charts

    Strong musical features:

        bpm: Beats per minute, a measure of song tempo
        key: Key of the song
        mode: Mode of the song (major or minor)
        danceability_%: Percentage indicating how suitable the song is for dancing
        valence_%: Positivity of the song's musical content
        energy_%: Perceived energy level of the song
        acousticness_%: Amount of acoustic sound in the song
        instrumentalness_%: Amount of instrumental content in the song
        liveness_%: Presence of live performance elements
        speechiness_%: Amount of spoken words in the song


| Variable Name      | Description                                 |
| ------------------ | ------------------------------------------- |
| `df`               | Original raw data                           |
| `new_df`           | Cleaned version used for analysis           |
| `num_cols`         | All numeric columns in `new_df`             |
| `skew_vals`        | Skewness values of numeric columns          |
| `high_skew_cols`   | Subset of `num_cols` that are highly skewed |
| `pt`               | PowerTransformer object (Yeo-Johnson)       |
| `transformed_data` | Numpy array of transformed values           |
| `col_yj`           | New transformed columns (added to `new_df`) |
