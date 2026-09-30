import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.cv import N_SPLITS
from src.data import EXPERIMENTS_DIR, PROJECT_ROOT
from src.results import load_all

COMPETITION = "spaceship-titanic"
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
PLACEHOLDER = "__PAYLOAD__"


def main():
    parser = argparse.ArgumentParser(description="experiments/*/results.json を集約して静的 HTML を生成する")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dashboard" / "index.html")
    args = parser.parse_args()

    report = load_all(args.experiments_dir)
    # 不正な results.json を混ぜて描くと誤った比較を招くため、1件でもエラーがあれば生成しない
    invalid = [path.parent.name for path, _, errors, _ in report if errors]
    if invalid:
        sys.exit(f"検証エラーのある実験: {', '.join(invalid)}。scripts/validate_results.py で詳細を確認する")

    payload = {
        "competition": COMPETITION,
        "n_splits": N_SPLITS,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiments": [results for _, results, _, _ in report],
    }
    # </script> を含む文字列で埋め込み先のタグが閉じないよう、</ をエスケープする
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE.read_text().replace(PLACEHOLDER, data))
    print(f"{args.output} を生成した（{len(report)} 件）")


if __name__ == "__main__":
    main()
