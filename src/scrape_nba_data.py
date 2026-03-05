"""
scrape_nba_data.py
==================
Standalone scraping script that loads real NBA play-by-play data from the
Kaggle basketball dataset (wyattowalsh/basketball), parses each event into
a flat historical possession log, and saves the result to
``data/processed/transitions.parquet``.

The dataset is read from ``basketball.sqlite`` in ``data/raw/``.  If the file
is absent it is downloaded automatically via the Kaggle API (requires
``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables or a
``~/.kaggle/kaggle.json`` credentials file).

Run this **once** (or whenever you want to refresh the dataset) before running
``src/collect_data.py``.

Usage
-----
::

    # Load the default 5 seasons (2019-20 through 2023-24)
    uv run python -m src.scrape_nba_data

    # Load specific seasons
    uv run python -m src.scrape_nba_data --seasons 2022-23 2023-24

    # Dry-run: parse only, skip database queries (uses cached raw files)
    uv run python -m src.scrape_nba_data --dry-run

    # Write output to a custom directory
    uv run python -m src.scrape_nba_data --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data_pipeline import NBAPlayByPlayScraper, PlayByPlayParser

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

DEFAULT_SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]


def scrape(
    seasons: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    dry_run: bool = False,
) -> Path:
    """
    Load NBA play-by-play data from the Kaggle basketball dataset and save
    a flat historical possession log to *processed_dir*.

    Parameters
    ----------
    seasons:
        List of season strings (e.g. ``["2022-23", "2023-24"]``).  Defaults
        to :data:`DEFAULT_SEASONS`.
    raw_dir:
        Directory containing ``basketball.sqlite`` (downloaded automatically
        if absent).
    processed_dir:
        Directory where ``transitions.parquet`` is written.
    dry_run:
        If ``True``, skip database queries and only process whatever raw files
        are already cached locally.

    Returns
    -------
    Path
        The path to the saved ``transitions.parquet`` file.
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("Dry-run mode: loading only cached raw files from %s", raw_dir)
        parquet_files = sorted(raw_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(
                "No cached raw files found in %s; nothing to parse.", raw_dir
            )
            return processed_dir / "transitions.parquet"
        raw_df = pd.concat(
            [pd.read_parquet(p) for p in parquet_files], ignore_index=True
        )
    else:
        logger.info("Scraping seasons: %s", seasons)
        scraper = NBAPlayByPlayScraper(seasons=seasons, raw_dir=raw_dir)
        raw_df = scraper.fetch_all()

    if raw_df.empty:
        logger.warning(
            "No raw play-by-play data available; transitions file not written."
        )
        return processed_dir / "transitions.parquet"

    logger.info(
        "Parsing %d raw events into historical possession records…", len(raw_df)
    )
    parser = PlayByPlayParser(processed_dir=processed_dir)
    transitions_df = parser.parse(raw_df)

    out_path = processed_dir / "transitions.parquet"
    if transitions_df.empty:
        logger.warning("Parser returned no transitions; output file not written.")
    else:
        logger.info(
            "Scraping complete: %d possession records parsed and saved to %s",
            len(transitions_df),
            out_path,
        )

    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.scrape_nba_data",
        description="Scrape NBA play-by-play data and build a flat historical possession log.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        metavar="SEASON",
        default=None,
        help=(
            "Season strings to scrape, e.g. '2022-23 2023-24'. "
            f"Defaults to the last {len(DEFAULT_SEASONS)} seasons."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR,
        metavar="DIR",
        help="Directory where transitions.parquet is written (default: data/processed/).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        metavar="DIR",
        help="Directory containing basketball.sqlite (default: data/raw/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip database queries; parse only locally cached raw files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    out = scrape(
        seasons=args.seasons,
        raw_dir=args.raw_dir,
        processed_dir=args.out_dir,
        dry_run=args.dry_run,
    )
    print(f"Done. Output: {out}")
    sys.exit(0)
