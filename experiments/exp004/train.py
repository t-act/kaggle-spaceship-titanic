import itertools
from collections import Counter

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import get_repeats, passenger_group
from src.data import ID_COL, load_data
from src.experiment import Experiment
from src.features import CATEGORICAL_FEATURES, build_features, to_catboost
from src.metric import repeated_fold_scores


def search_params(x_tr, y_tr, groups_tr, params, train_dir):
    """学習側の内側の検証で、探索範囲から AUC が最良のパラメータと木の数を選ぶ。

    探索は seed 1つで行う。全 seed で探索すると学習回数が seed 数倍になるが、選ばれる値はほぼ変わらないため。
    """
    inner = StratifiedGroupKFold(n_splits=params["inner_splits"], shuffle=True, random_state=params["inner_seed"])
    idx_fit, idx_es = next(inner.split(x_tr, y_tr, groups_tr))
    keys = list(params["search"])
    best = None
    for values in itertools.product(*params["search"].values()):
        candidate = dict(zip(keys, values, strict=True))
        probe = CatBoostClassifier(
            **params["catboost"], **candidate, random_seed=params["seeds"][0], train_dir=train_dir
        )
        probe.fit(
            x_tr.iloc[idx_fit],
            y_tr.iloc[idx_fit],
            cat_features=CATEGORICAL_FEATURES,
            eval_set=(x_tr.iloc[idx_es], y_tr.iloc[idx_es]),
        )
        auc = probe.get_best_score()["validation"]["AUC"]
        if best is None or auc > best[0]:
            best = (auc, candidate, probe.get_best_iteration() + 1)
    return best[1], best[2]


def main():
    with Experiment(__file__) as exp:
        params = exp.params
        train, test = load_data()
        x_train, y_train, x_test = build_features(train, test)
        x_train, x_test = to_catboost(x_train), to_catboost(x_test)
        train_dir = str(exp.dir / "catboost_info")
        final_params = {k: v for k, v in params["catboost"].items() if k != "early_stopping_rounds"}

        repeats = get_repeats(train)
        groups = passenger_group(train)
        oofs = np.zeros((len(repeats), len(x_train)))
        test_proba = []
        chosen = Counter()
        for r, folds in enumerate(repeats):
            for idx_tr, idx_va in folds:
                x_tr, y_tr = x_train.iloc[idx_tr], y_train.iloc[idx_tr]
                best, n_iterations = search_params(x_tr, y_tr, groups.iloc[idx_tr], params, train_dir)
                chosen[tuple(best.items())] += 1
                print(f"repeat={r}: {best}, iterations={n_iterations}")
                for seed in params["seeds"]:
                    model = CatBoostClassifier(
                        **{**final_params, "iterations": n_iterations}, **best, random_seed=seed, train_dir=train_dir
                    )
                    model.fit(x_tr, y_tr, cat_features=CATEGORICAL_FEATURES)
                    oofs[r, idx_va] += model.predict_proba(x_train.iloc[idx_va])[:, 1] / len(params["seeds"])
                    test_proba.append(model.predict_proba(x_test)[:, 1])

        print("選ばれたパラメータ:", dict(chosen))
        exp.record_cv(repeated_fold_scores(y_train, oofs, repeats))
        exp.save_oof(oofs)
        # fold と seed ごとの予測ラベルの多数決ではなく確率を平均する。多数決は境界付近の情報を捨てる
        exp.save_test_predictions(test[ID_COL], np.mean(test_proba, axis=0))


if __name__ == "__main__":
    main()
