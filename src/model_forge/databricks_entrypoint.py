from __future__ import annotations

import argparse
from pathlib import Path

from model_forge.pipeline import run_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_demo(args.config, args.output, clean=False)
    print(result)


if __name__ == "__main__":
    main()
