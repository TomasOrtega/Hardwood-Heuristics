"""
generate_docs.py
================
Regenerates the theorem Markdown documentation files with numbers derived
directly from the historical win-rate results stored in ``data/processed/``.

Call this module after running ``python -m src.collect_data`` (which invokes
:func:`generate_all_docs` automatically), or run it standalone::

    python -m src.generate_docs

Adding a new theorem
--------------------
1. Implement a ``_generate_<key>_doc`` helper below.
2. Register it in :data:`_DOC_GENERATORS`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Theorem 1 doc generator
# ---------------------------------------------------------------------------


def _generate_theorem1_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    from src.theorems.theorem1 import generate_doc

    return generate_doc(processed_dir=processed_dir, docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Theorem 2 doc generator
# ---------------------------------------------------------------------------


def _generate_theorem2_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Write the static Theorem 2 Markdown file."""
    from src.theorems.theorem2 import generate_doc

    return generate_doc(processed_dir=processed_dir, docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Theorem 3 doc generator
# ---------------------------------------------------------------------------


def _generate_theorem3_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    from src.theorems.theorem3 import generate_doc

    return generate_doc(processed_dir=processed_dir, docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DOC_GENERATORS: Dict = {
    "theorem1": _generate_theorem1_doc,
    "theorem2": _generate_theorem2_doc,
    "theorem3": _generate_theorem3_doc,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_all_docs(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> List[Path]:
    """
    Regenerate all theorem Markdown files from historical analysis results.

    Reads pre-computed sweep data from *processed_dir* and writes updated
    Markdown files to *docs_dir*.  Call after ``collect_all()`` so that the
    processed data is up-to-date.

    Returns
    -------
    List of Path objects for all generated Markdown files.
    """
    paths: List[Path] = []
    for key, generator in _DOC_GENERATORS.items():
        logger.info("Generating doc for %s...", key)
        paths.append(generator(processed_dir=processed_dir, docs_dir=docs_dir))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generated = generate_all_docs()
    for p in generated:
        print(f"Written: {p}")
