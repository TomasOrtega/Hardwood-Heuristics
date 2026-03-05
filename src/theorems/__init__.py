"""
src/theorems
============
Per-theorem analysis modules.  Each sub-module contains the data-collection,
visualisation, and documentation-generation logic for a single theorem.

Adding a new theorem
--------------------
1. Create ``src/theorems/theorem<N>.py`` with ``collect``, ``plot``, and
   ``generate_doc`` functions that follow the same signatures as the existing
   modules.
2. Import and register the new module in the orchestrators:
   ``collect_data._COLLECTORS``, ``generate_docs._DOC_GENERATORS``,
   ``visualizations._PLOTTERS``.
"""

from __future__ import annotations

from src.theorems import theorem1, theorem2, theorem3

__all__ = ["theorem1", "theorem2", "theorem3"]
