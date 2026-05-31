import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Circular+Std&display=swap');

* { font-family: 'Segoe UI', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #0f0f0f 100%);
    color: white;
}

.main-header {
    text-align: center;
    padding: 40px 0 20px 0;
}

.main-header h1 {
    font-size: 3.5em;
    background: linear-gradient(90deg, #1DB954, #1ed760);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    letter-spacing: -1px;
}

.main-header p {
    color: #b3b3b3;
    font-size: 1.2em;
}

.search-container {
    background: #282828;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    border: 1px solid #333;
}

.song-card {
    background: linear-gradient(135deg, #282828, #1a1a1a);
    border-radius: 15px;
    padding: 20px 25px;
    margin: 10px 0;
    border-left: 5px solid #1DB954;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 20px;
}

.rank-badge {
    background: #1DB954;
    color: black;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 1.1em;
    flex-shrink: 0;
}

.song-info h3 {
    color: white;
    margin: 0;
    font-size: 1.1em;
}

.song-info p {
    color: #b3b3b3;
    margin: 2px 0;
    font-size: 0.9em;
}

.score-badge {
    margin-left: auto;
    background: #1DB954;
    color: black;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9em;
}

.genre-badge {
    background: #333;
    color: #1DB954;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.8em;
    display: inline-block;
    margin-top: 4px;
}

.stats-box {
    background: #282828;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    border: 1px solid #333;
}

.stats-box h2 {
    color: #1DB954;
    font-size: 2em;
    margin: 0;
}

.stats-box p {
    color: #b3b3b3;
    margin: 5px 0 0 0;
}

.found-song-banner {
    background: linear-gradient(90deg, #1DB954, #158a3e);
    border-radius: 15px;
    padding: 20px 30px;
    margin: 20px 0;
    color: black;
}

.found-song-banner h2 {
    margin: 0;
    font-size: 1.5em;
    font-weight: 900;
}

.found-song-banner p {
    margin: 5px 0 0 0;
    opacity: 0.8;
}

div[data-testid="stTextInput"] input {
    background: #333 !important;
    color: white !important;
    border: 2px solid #444 !important;
    border-radius: 30px !important;
    padding: 15px 25px !important;
    font-size: 1.1em !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #1DB954 !important;
}

.stButton button {
    background: linear-gradient(90deg, #1DB954, #1ed760) !important;
    color: black !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 15px 40px !important;
    font-size: 1.1em !important;
    font-weight: 700 !important;
    width: 100% !important;
    cursor: pointer !important;
}

.stSlider { padding: 10px 0; }

hr { border-color: #333; }
</style>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/priya/Desktop/spotify recommendation/SpotifyFeatures.csv')
    df = df.drop_duplicates(subset='track_name')
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    return df

@st.cache_data
def compute_similarity(_df):
    features = ['popularity', 'acousticness', 'danceability',
                'energy', 'instrumentalness', 'liveness',
                'loudness', 'speechiness', 'tempo', 'valence']
    X = _df[features]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return cosine_similarity(X_scaled)

# ===== HEADER =====
st.markdown("""
<div class="main-header">
    <h1>🎵 Spotify Recommender</h1>
    <p>Discover music you'll love </p>
</div>
""", unsafe_allow_html=True)

# ===== LOAD =====
with st.spinner("🎵 Loading music library..."):
    df = load_data()
    cosine_sim = compute_similarity(df)

# ===== STATS ROW =====
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stats-box">
        <h2>{len(df):,}</h2>
        <p>🎵 Songs Available</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stats-box">
        <h2>{df['genre'].nunique()}</h2>
        <p>🎸 Genres</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stats-box">
        <h2>{df['artist_name'].nunique():,}</h2>
        <p>🎤 Artists</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ===== SEARCH =====
col_search, col_num = st.columns([3, 1])
with col_search:
    song_input = st.text_input("", placeholder="🔍 Type a song name... e.g. Love, Happy, Baby")
with col_num:
    n_rec = st.slider("Results", 5, 20, 10)

search_btn = st.button("🎵 Find Similar Songs")

# ===== RESULTS =====
if search_btn:
    if not song_input:
        st.warning("⚠️ Please enter a song name!")
    else:
        matches = df[df['track_name'].str.contains(song_input, case=False, na=False)]

        if matches.empty:
            st.error(f"❌ '{song_input}' not found! Try: Love, Happy, Dance, Baby, Night")
        else:
            idx = matches.index[0]
            found_song = df.loc[idx, 'track_name']
            found_artist = df.loc[idx, 'artist_name']
            found_genre = df.loc[idx, 'genre']

            st.markdown(f"""
            <div class="found-song-banner">
                <h2>🎵 {found_song}</h2>
                <p>🎤 {found_artist} &nbsp;|&nbsp; 🎸 {found_genre}</p>
            </div>
            """, unsafe_allow_html=True)

            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_rec+1]
            song_indices = [i[0] for i in sim_scores]

            result = df[['track_name', 'artist_name', 'genre']].iloc[song_indices].copy()
            result['similarity'] = [round(i[1]*100, 1) for i in sim_scores]
            result = result.reset_index(drop=True)

            st.markdown("### 🎶 Recommended Songs")

            for i, row in result.iterrows():
                st.markdown(f"""
                <div class="song-card">
                    <div class="rank-badge">{i+1}</div>
                    <div class="song-info">
                        <h3>{row['track_name']}</h3>
                        <p>🎤 {row['artist_name']}</p>
                        <span class="genre-badge">🎸 {row['genre']}</span>
                    </div>
                    <div class="score-badge">{row['similarity']}% Match</div>
                </div>
                """, unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; padding:20px;'>
    🎵 Spotify Music Recommender | Built with Python & Machine Learning
</div>
""", unsafe_allow_html=True)