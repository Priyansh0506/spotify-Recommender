import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")

# basic styling - not too fancy
st.markdown("""
<style>
.stApp { background-color: #1a1a1a; color: white; }
.song-card {
    background-color: #2a2a2a;
    padding: 12px 18px;
    border-radius: 8px;
    margin: 8px 0;
    border-left: 3px solid #1DB954;
}
.stButton button {
    background-color: #1DB954 !important;
    color: black !important;
    border-radius: 5px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# title
st.title("🎵 Music Recommendation System")
st.write("This project recommends similar songs based on audio features using Cosine Similarity.")
st.write("---")

# load data
@st.cache_data
def load_data():
    df = pd.read_csv('SpotifyFeatures.csv')
    df = df.drop_duplicates(subset='track_name')
    df = df.dropna()
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    return df

@st.cache_data
def compute_similarity(_df):
    features = ['popularity', 'acousticness', 'danceability',
                'energy', 'instrumentalness', 'liveness',
                'loudness', 'speechiness', 'tempo', 'valence']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(_df[features])
    return cosine_similarity(X_scaled)

# loading
st.write("Loading dataset...")
df = load_data()
cosine_sim = compute_similarity(df)
st.success(f"Dataset loaded! Total songs: {len(df)}")

st.write("---")

# dataset info
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Songs", len(df))
col2.metric("Total Genres", df['genre'].nunique())
col3.metric("Total Artists", df['artist_name'].nunique())

st.write("---")

# search section
st.subheader("🔍 Search a Song")
song_input = st.text_input("Enter song name:")
n_rec = st.slider("How many recommendations?", 5, 15, 10)

if st.button("Get Recommendations"):
    if not song_input:
        st.warning("Please enter a song name!")
    else:
        matches = df[df['track_name'].str.contains(song_input, case=False, na=False)]

        if matches.empty:
            st.error(f"Song '{song_input}' not found in dataset. Try: Love, Happy, Baby, Dance")
        else:
            idx = matches.index[0]
            found_song = df.loc[idx, 'track_name']
            found_artist = df.loc[idx, 'artist_name']
            found_genre = df.loc[idx, 'genre']

            st.write(f"**Found:** {found_song} by {found_artist} ({found_genre})")
            st.write("---")
            st.subheader(f"Top {n_rec} Similar Songs:")

            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_rec+1]
            song_indices = [i[0] for i in sim_scores]

            result = df[['track_name', 'artist_name', 'genre']].iloc[song_indices].copy()
            result['similarity'] = [round(i[1]*100, 1) for i in sim_scores]
            result = result.reset_index(drop=True)

            for i, row in result.iterrows():
                st.markdown(f"""
                <div class="song-card">
                    <b>{i+1}. {row['track_name']}</b><br>
                    Artist: {row['artist_name']}<br>
                    Genre: {row['genre']} &nbsp;|&nbsp; 
                    Match: {row['similarity']}%
                </div>
                """, unsafe_allow_html=True)

st.write("---")
st.caption("Project by: Priyansh | Algorithm: Content-Based Filtering | Cosine Similarity")