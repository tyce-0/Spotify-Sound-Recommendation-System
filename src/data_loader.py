import os
import pandas as pd

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "spotify_tracks.csv")

AUDIO_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Spotify tracks CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/get_data.py` first."
        )
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> dict:
    """Run basic data quality checks and return a summary dict."""
    summary = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()) if "id" in df.columns else None,
        "year_range": (int(df["year"].min()), int(df["year"].max())) if "year" in df.columns else None,
    }
    return summary


if __name__ == "__main__":
    df = load_raw_data()
    summary = validate_data(df)
    for k, v in summary.items():
        print(f"{k}: {v}")