import csv
import io
import json
import subprocess

from src.data import EXPERIMENTS_DIR
from src.results import RESULTS_FILE

COMPETITION = "spaceship-titanic"
COMPLETE = "SubmissionStatus.COMPLETE"


def fetch_submissions():
    out = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMPETITION, "-v", "--page-size", "200"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # CSV の前に案内文が出ることがあるため、ヘッダ行から読む
    return list(csv.DictReader(io.StringIO(out[out.index("ref,") :])))


def to_score(value):
    return float(value) if value else None


def latest_scores(submissions):
    """実験 ID を説明文に持つ採点済みの提出のうち、最新のものを実験ごとに返す。

    make submit は説明文に実験 ID だけを入れるため、説明文の完全一致で突き合わせる。
    """
    latest = {}
    for s in sorted(submissions, key=lambda s: s["date"]):
        if s["status"] == COMPLETE:
            latest[s["description"].strip()] = {
                "public": to_score(s["publicScore"]),
                "private": to_score(s["privateScore"]),
            }
    return latest


def main():
    scores = latest_scores(fetch_submissions())
    updated = 0
    for path in sorted(EXPERIMENTS_DIR.glob(f"*/{RESULTS_FILE}")):
        exp_id = path.parent.name
        if exp_id not in scores:
            continue
        results = json.loads(path.read_text())
        if results["lb"] == scores[exp_id]:
            continue
        print(f"{exp_id}: {results['lb']} -> {scores[exp_id]}")
        results["lb"] = scores[exp_id]
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        updated += 1
    print(f"{updated} 件の lb を更新した")


if __name__ == "__main__":
    main()
