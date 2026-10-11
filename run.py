"""Convenience entry point.

Usage:
    python run.py            # start the web app
    python run.py train      # train (or retrain) the model
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _train() -> None:
    from ml.train import train_and_save

    train_and_save()


def _serve() -> None:
    from app.main import main

    main()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        _train()
    else:
        _serve()


if __name__ == "__main__":
    main()
