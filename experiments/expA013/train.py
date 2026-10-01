import json

import numpy as np

from src.cv import CV_SCHEME, get_repeats
from src.data import EXPERIMENTS_DIR, ID_COL, load_data
from src.experiment import Experiment
from src.features import build_features
from src.metric import repeated_fold_scores
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
        _, y_train, _ = build_features(train, test)

        members = [load_member(m) for m in params["members"]]
        weights = np.array(params["weights"], dtype=float) / np.sum(params["weights"])
        oofs = sum(w * member_oofs for w, (member_oofs, _) in zip(weights, members, strict=True))
        test_proba = sum(w * member_test for w, (_, member_test) in zip(weights, members, strict=True))

        exp.record_cv(repeated_fold_scores(y_train, oofs, get_repeats(train)))
        exp.save_oof(oofs)
        exp.save_test_predictions(test[ID_COL], test_proba)


if __name__ == "__main__":
    main()
