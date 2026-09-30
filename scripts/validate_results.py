import argparse
import sys
from pathlib import Path

from src.data import EXPERIMENTS_DIR
from src.results import load_all


def main():
    parser = argparse.ArgumentParser(description="experiments/*/results.json をスキーマと整合ルールで検証する")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    args = parser.parse_args()

    report = load_all(args.experiments_dir)
    n_errors = 0
    for path, _, errors, warnings in report:
        label = path.parent.name
        for e in errors:
            print(f"[ERROR] {label}: {e}")
        for w in warnings:
            print(f"[WARN]  {label}: {w}")
        n_errors += len(errors)

    print(f"{len(report)} 件を検証: エラー {n_errors} 件")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
