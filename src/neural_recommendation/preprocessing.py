"""Load MovieLens ratings and save them to the processed data directory."""

from pathlib import Path

import pandas as pd


# This file lives in src/neural_recommendation/, two levels below the root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "movie_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RATINGS_PATH = RAW_DATA_DIR / "ratings.dat"
PROCESSED_RATINGS_PATH = PROCESSED_DATA_DIR / "ratings.csv"
