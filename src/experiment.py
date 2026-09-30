import json
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.data import EXPERIMENTS_DIR, PROJECT_ROOT
from src.metric import METRIC
from src.results import RESULTS_FILE, validate


def git_hash(exclude):
    """HEAD の短縮ハッシュ。未コミットの変更があれば -dirty を付ける。

    実行を止めずに記録だけ残す。止めると小さな確認実験のたびにコミットが必要になるため。
    """
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    # 失敗した実験を再実行するとき、前回の results.json だけで dirty 扱いにならないよう除外する
    pathspec = [f":!{exclude.relative_to(PROJECT_ROOT)}"] if exclude.is_relative_to(PROJECT_ROOT) else []
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", *pathspec],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not status:
        return head
    warnings.warn("未コミットの変更があるため git_hash に -dirty を付ける。再現性のためコミットしてから実行する")
    return f"{head}-dirty"


class Experiment:
    """train.py を with 文で包み、results.json の出力と検証を担う。

    例外で抜けたときも status=failed を書き残し、失敗した試行も一覧に残す。
    """

    def __init__(self, train_file):
        self.dir = Path(train_file).resolve().parent
        self.id = self.dir.name
        self.config = yaml.safe_load((self.dir / "config.yaml").read_text())
        self.params = self.config["params"]
        self.results_path = self.dir / RESULTS_FILE

    def __enter__(self):
        if self.results_path.exists() and json.loads(self.results_path.read_text())["status"] == "done":
            raise FileExistsError(
                f"{self.id} は実行済み。上書きせず新しい番号で複製する: make new FROM={self.id} ORIGIN=human|claude"
            )

        cfg = self.config
        self.results = {
            "exp_id": self.id,
            "origin": cfg["origin"],
            "parent": cfg["parent"],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_hash": git_hash(exclude=self.results_path),
            "hypothesis": cfg["hypothesis"],
            "changes": cfg["changes"],
            "model": cfg["model"],
            "config": self.params,
            "metric": METRIC,
            "cv": None,
            "lb": {"public": None, "private": None},
            "train_time_sec": None,
            "status": "running",
            "conclusion": "",
            "tags": cfg.get("tags", []),
        }
        # config.yaml の記入漏れは学習前に止める。学習後に気づくと実行時間が無駄になるため
        self._check()
        self._write()
        self._started = time.perf_counter()
        return self

    def record_cv(self, fold_scores):
        self.results["cv"] = {
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "folds": [float(s) for s in fold_scores],
        }
        print(f"[cv] {self.results['cv']['mean']:.4f} +- {self.results['cv']['std']:.4f}")

    def save_oof(self, oof):
        np.save(self.dir / "oof.npy", oof)

    def save_submission(self, df):
        df.to_csv(self.dir / "submission.csv", index=False)

    def __exit__(self, exc_type, exc, tb):
        self.results["train_time_sec"] = round(time.perf_counter() - self._started, 1)
        succeeded = exc_type is None and self.results["cv"] is not None
        self.results["status"] = "done" if succeeded else "failed"
        self._write()
        if exc_type is None and not succeeded:
            raise RuntimeError("record_cv() を呼ばずに終了したため status=failed とした")
        if succeeded:
            self._check()
        return False

    def _check(self):
        known_ids = {p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()}
        errors, _ = validate(self.results, self.dir, known_ids)
        if errors:
            raise ValueError(f"{self.id} の results が不正:\n" + "\n".join(f"  - {e}" for e in errors))

    def _write(self):
        self.results_path.write_text(json.dumps(self.results, ensure_ascii=False, indent=2) + "\n")
