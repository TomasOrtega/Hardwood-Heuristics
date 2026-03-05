import logging
from pathlib import Path
from src.theorems import theorem1, theorem2, theorem3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "assets" / "images"

def generate_all_plots() -> None:
    """Generate plots for all theorems and save them to ``docs/assets/images/``.

    Loads pre-computed sweep data from ``data/processed/`` (populated by
    ``python -m src.collect_data``) and writes SVG figures for Theorems 1--3.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating plots...")
    p1 = theorem1.plot(PROCESSED_DIR, IMAGES_DIR)
    p2 = theorem2.plot(PROCESSED_DIR, IMAGES_DIR)
    p3 = theorem3.plot(PROCESSED_DIR, IMAGES_DIR)
    print(f"Saved: {p1}\nSaved: {p2}\nSaved: {p3}")

if __name__ == "__main__":
    generate_all_plots()
