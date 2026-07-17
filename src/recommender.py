"""
Content-based song recommendation logic using cosine similarity over
scaled audio features.
"""

import ast

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def clean_artists(artist_str: str) -> str:
    """Convert a stringified artist list like "['A', 'B']" into 'A, B'."""
    try:
        parsed = ast.literal_eval(artist_str)
        return ", ".join(parsed) if isinstance(parsed, list) else str(artist_str)
    except (ValueError, SyntaxError):
        return str(artist_str).strip("[]'\"")


def recommend(df: pd.DataFrame, X_scaled, song_name: str, artist_name: str = None, n: int = 10):
    """
    Recommend the top `n` most similar songs to the given song, based on
    cosine similarity of audio features.

    Parameters
    ----------
    df : pd.DataFrame
        The tracks dataframe (must include 'name', 'artists', 'popularity',
        'year', 'cluster' columns).
    X_scaled : np.ndarray
        Scaled audio feature matrix, same row order as `df`.
    song_name : str
        Title of the song to base recommendations on.
    artist_name : str, optional
        Used to disambiguate when multiple songs share the same title.
    n : int
        Number of recommendations to return.

    Returns
    -------
    (chosen_row, recommendations_df) or (None, None) if no match is found.
    """
    matches = df[df["name"].str.lower() == song_name.lower()]
    if artist_name:
        matches = matches[matches["artists"].str.lower().str.contains(artist_name.lower())]

    if len(matches) == 0:
        return None, None

    # If multiple entries exist for the same song (re-releases, compilations),
    # use the most popular version as the reference point
    chosen = matches.sort_values("popularity", ascending=False).iloc[0]
    idx = chosen.name

    query_vector = X_scaled[idx].reshape(1, -1)
    similarities = cosine_similarity(query_vector, X_scaled)[0]

    results = df.copy()
    results["similarity"] = similarities

    # Exclude every entry of the query song itself (not just the one exact row),
    # so a duplicate/re-release of the query song can't be recommended back to the user
    is_same_song = (
        (results["name"].str.lower() == chosen["name"].lower())
        & (results["artists"] == chosen["artists"])
    )
    results = results[~is_same_song]

    results = results.sort_values("similarity", ascending=False)

    # De-duplicate near-identical re-releases of OTHER songs too, keeping
    # only the closest-matching version of each distinct song
    results["dedupe_key"] = results["name"].str.lower().str.strip() + "||" + results["artists"]
    results = results.drop_duplicates(subset="dedupe_key", keep="first")

    top_n = results.head(n)[["name", "artists", "year", "similarity", "cluster"]]
    return chosen, top_n
