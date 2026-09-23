"""Build the aligned_100/aligned_250 MASSIVE samples and save them to data/samples.

Usage:
    uv run python scripts/build_massive_samples.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset_loaders.massive import build_all_samples  # noqa: E402


def main() -> None:
    for name, path in build_all_samples().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
