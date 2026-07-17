"""
Interactive demo for the Spotify content-based recommendation system.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.recommender import recommend, clean_artists

FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "speechiness", "valence",
]

CLUSTER_LABELS = {
    0: "Acoustic Instrumentals",
    1: "Upbeat Mainstream",
    2: "Mellow Vocal / Ballads",
    3: "Live Recordings",
    4: "High-Energy / Up-Tempo",
    5: "Spoken Word",
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "tracks_with_clusters.csv")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "models", "scaler.joblib")


# ----------------------------------------------------------------
# Data loading (cached so the ~170k-row dataset only loads once per session)
# ----------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["primary_artist"] = df["artists"].str.strip("[]'\"").str.split(",").str[0].str.strip()
    return df


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data
def get_scaled_features(_df, _scaler):
    return _scaler.transform(_df[[
        "acousticness", "danceability", "energy", "instrumentalness",
        "liveness", "loudness", "speechiness", "tempo", "valence",
    ]])


# ----------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------
st.set_page_config(page_title="Sound-Alike | Spotify Recommender", page_icon="\U0001F3B5", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #121212; }
    .song-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.7rem;
        border-left: 4px solid #1DB954;
    }
    .song-title { font-size: 1.05rem; font-weight: 600; color: #F5F5F5; }
    .song-meta { color: #A0A0A0; font-size: 0.85rem; }
    .similarity-badge {
        background-color: #1DB954;
        color: #121212;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        float: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()
scaler = load_scaler()
X_scaled = get_scaled_features(df, scaler)

st.title("\U0001F3B5 Sound-Alike")
st.caption(
    "A content-based music recommender built on ~170,000 real Spotify tracks (1921\u20132020). "
    "Pick a song, and get the 10 tracks whose *sound* — tempo, energy, mood, and more — is most similar."
)

# ----------------------------------------------------------------
# Sidebar: dataset info + honest scope notes
# ----------------------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        f"""
        **Dataset:** {len(df):,} tracks, 1921\u20132020
        **Method:** Cosine similarity over 9 scaled audio features
        **Clustering:** K-Means (k=6), used here to label each track's sound category

        ---
        **Worth knowing:**
        Recommendations are based purely on how a song *sounds*
        (tempo, energy, mood, etc.) — not on genre, lyrics, or what
        other listeners liked. Two songs can be recommended together
        because they share a similar tempo and energy, even if they're
        culturally or stylistically very different. See the project
        README for a fuller discussion of this trade-off.
        """
    )
    st.divider()
    st.caption("Built as a portfolio project. Not affiliated with Spotify.")


# ----------------------------------------------------------------
# Song search
# ----------------------------------------------------------------
st.subheader("Find a song")

search_col, _ = st.columns([2, 1])
with search_col:
    query = st.text_input(
        "Search by song title",
        placeholder="e.g. Bohemian Rhapsody",
        label_visibility="collapsed",
    )

selected_row = None

if query:
    matches = df[df["name"].str.contains(query, case=False, na=False)]
    if len(matches) == 0:
        st.warning("No songs found matching that title. Try a different search.")
    else:
        matches = matches.sort_values("popularity", ascending=False).head(50)
        options = [
            f"{row['name']} — {row['primary_artist']} ({int(row['year'])})"
            for _, row in matches.iterrows()
        ]
        choice = st.selectbox(f"{len(matches)} match(es) found — select one:", options)
        selected_row = matches.iloc[options.index(choice)]

# ----------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------
if selected_row is not None:
    chosen, recommendations = recommend(
        df, X_scaled,
        song_name=selected_row["name"],
        artist_name=selected_row["primary_artist"],
        n=10,
    )

    if chosen is None:
        st.error("Something went wrong finding that song. Please try another.")
    else:
        st.divider()
        st.markdown(f"### Because you picked: *{chosen['name']}* — {chosen['primary_artist']}")

        cluster_id = int(chosen["cluster"])
        st.markdown(
            f"**Sound category:** {CLUSTER_LABELS.get(cluster_id, f'Cluster {cluster_id}')} "
            f"&nbsp;|&nbsp; **Year:** {int(chosen['year'])} "
            f"&nbsp;|&nbsp; **Popularity:** {int(chosen['popularity'])}/100"
        )

        left, right = st.columns([1.3, 1])

        with left:
            st.markdown("#### Top 10 recommendations")
            for _, rec in recommendations.iterrows():
                rec_cluster = CLUSTER_LABELS.get(int(rec["cluster"]), f"Cluster {int(rec['cluster'])}")
                st.markdown(
                    f"""
                    <div class="song-card">
                        <span class="similarity-badge">{rec['similarity']*100:.1f}% match</span>
                        <div class="song-title">{rec['name']}</div>
                        <div class="song-meta">{clean_artists(rec['artists'])} \u00b7 {int(rec['year'])} \u00b7 {rec_cluster}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown("#### Why these matches?")
            st.caption(
                "This chart compares the chosen song's audio profile against "
                "its closest match, feature by feature."
            )

            top_match = recommendations.iloc[0]
            top_match_full = df[
                (df["name"] == top_match["name"]) & (df["artists"] == top_match["artists"])
            ].iloc[0]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[chosen[f] for f in FEATURES],
                theta=FEATURES,
                fill="toself",
                name=chosen["name"],
                line_color="#1DB954",
            ))
            fig.add_trace(go.Scatterpolar(
                r=[top_match_full[f] for f in FEATURES],
                theta=FEATURES,
                fill="toself",
                name=top_match["name"],
                line_color="#B3B3B3",
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
                    bgcolor="#1E1E1E",
                ),
                paper_bgcolor="#121212",
                font_color="#F5F5F5",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                margin=dict(t=20, b=20),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Sound category breakdown")
            cluster_counts = recommendations["cluster"].map(CLUSTER_LABELS).value_counts()
            st.bar_chart(cluster_counts)

else:
    st.info("Search for a song above to get started.")
    st.markdown("**Try:** Bohemian Rhapsody, Shape of You, Hotel California, Billie Jean")
