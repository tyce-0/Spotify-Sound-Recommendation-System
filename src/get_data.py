import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/gabminamedez/spotify-data/master/data.csv"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "spotify_tracks.csv")


def download_dataset(url: str = DATA_URL, out_path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"Dataset already exists at {out_path}, skipping download.")
        return

    print(f"Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    download_dataset()