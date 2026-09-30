import argparse
import re
import shutil
import sys

from src.data import EXPERIMENTS_DIR

PREFIXES = {"human": "exp", "claude": "expA"}
COPIED_FILES = ["config.yaml", "train.py"]


def next_id(origin):
    prefix = PREFIXES[origin]
    numbers = [int(m[1]) for p in EXPERIMENTS_DIR.iterdir() if (m := re.fullmatch(rf"{prefix}(\d{{3}})", p.name))]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def set_field(text, key, value):
    # yaml を読み書きし直すとコメントが消えるため、トップレベルの行を置き換える。続く字下げ行は複数行の値として消す
    pattern = rf"^{key}:.*\n(?:[ \t]+.*\n)*"
    if not re.search(pattern, text, flags=re.MULTILINE):
        sys.exit(f"config.yaml にトップレベルの {key}: がない")
    return re.sub(pattern, f"{key}: {value}\n", text, count=1, flags=re.MULTILINE)


def main():
    parser = argparse.ArgumentParser(description="既存の実験を新しい番号で複製する")
    parser.add_argument("--from", dest="parent", required=True, help="派生元の実験 ID")
    parser.add_argument("--origin", required=True, choices=list(PREFIXES), help="起案者")
    args = parser.parse_args()

    src_dir = EXPERIMENTS_DIR / args.parent
    if not src_dir.is_dir():
        sys.exit(f"{src_dir} がない")

    exp_id = next_id(args.origin)
    dst_dir = EXPERIMENTS_DIR / exp_id
    dst_dir.mkdir()
    for name in COPIED_FILES:
        shutil.copy2(src_dir / name, dst_dir / name)

    config_path = dst_dir / "config.yaml"
    text = config_path.read_text()
    text = set_field(text, "origin", args.origin)
    text = set_field(text, "parent", args.parent)
    # 空のままだと Experiment が学習前に止めるため、仮説と変更点の書き忘れを防げる
    text = set_field(text, "hypothesis", '""')
    text = set_field(text, "changes", '""')
    config_path.write_text(text)

    print(f"{dst_dir} を作成した。config.yaml の hypothesis と changes を記入してから実行する")


if __name__ == "__main__":
    main()
