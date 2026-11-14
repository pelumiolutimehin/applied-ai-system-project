"""Regenerate data/songs.csv from the raw Spotify songs dataset.

Download the source file first (public, no login):
    curl -L -o data/spotify_songs.csv \
      https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv

Usage (from the repo root):
    python -m scripts.build_catalog
    python -m scripts.build_catalog --source data/spotify_songs.csv --out data/songs.csv
"""

import argparse

from src.catalog import build_catalog
from src.config import CATALOG_PATH, RAW_SOURCE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(RAW_SOURCE_PATH))
    parser.add_argument("--out", default=str(CATALOG_PATH))
    parser.add_argument("--per-subgenre", type=int, default=10)
    args = parser.parse_args()

    catalog = build_catalog(args.source, per_subgenre=args.per_subgenre)
    catalog.to_csv(args.out, index=False)

    print(f"Wrote {len(catalog)} songs to {args.out}")
    print("\nSongs per genre:")
    print(catalog["genre"].value_counts().to_string())
    print("\nSongs per mood:")
    print(catalog["mood"].value_counts().to_string())


if __name__ == "__main__":
    main()
