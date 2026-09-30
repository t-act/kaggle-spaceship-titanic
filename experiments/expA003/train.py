import json

import numpy as np

from src.cv import CV_SCHEME, get_folds
from src.data import EXPERIMENTS_DIR, ID_COL, TARGET, load_data
from src.experiment import Experiment
from src.metric import fold_scores
from src.results import RESULTS_FILE


def load_member(exp_id):
    member_dir = EXPERIMENTS_DIR / exp_id
    results = json.loads((member_dir / RESULTS_FILE).read_text())
    # 分割方式が違う OOF を平均すると、どの分割の検証スコアとも言えない値になるため拒否する
    if results["status"] != "done" or results["cv"]["scheme"] != CV_SCHEME:
        raise ValueError(f"{exp_id} は {CV_SCHEME} の分割で完了した実験でない")
    return np.load(member_dir / "oof.npy"), np.load(member_dir / "test_proba.npy")


def main():
    with Experiment(__file__) as exp:
        params = exp.params
        if len(params["members"]) != len(params["weights"]):
            raise ValueError("members と weights の長さが一致しない")
        train, test = load_data()

        members = [load_member(m) for m in params["members"]]
        weights = np.array(params["weights"]) / np.sum(params["weights"])
        oof = sum(w * member_oof for w, (member_oof, _) in zip(weights, members))
        test_proba = sum(w * member_test for w, (_, member_test) in zip(weights, members))

        exp.record_cv(fold_scores(train[TARGET], oof, get_folds(train)))
        exp.save_oof(oof)
        exp.save_test_predictions(test[ID_COL], test_proba)


if __name__ == "__main__":
    main()
