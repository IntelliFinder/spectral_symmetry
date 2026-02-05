#!/usr/bin/env python
"""Entry point for training the Spectral Transformer classifier."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.experiments.spectral_transformer.train import main  # noqa: E402

if __name__ == "__main__":
    main()
