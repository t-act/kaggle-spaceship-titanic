import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import get_repeats, passenger_group
from src.data import ID_COL, load_data
from src.experiment import Experiment
from src.features import CATEGORICAL_FEATURES, build_features, to_catboost
from src.metric import repeated_fold_scores


def fit_catboost(x_tr, y_tr, groups_tr, params, seed, train_dir):
    """木の数を検証 fold ではなく、学習側を内側で分けた検証で決める。

    検証 fold で early stopping し最良時点のモデルを使うと、木の数が検証 fold に合わせて選ばれ OOF が楽観的になるため。
    学習し直すときにデータが増える分だけ木の数を増やす補正はしない。補正の係数が新たな調整対象になるため。
    """
    inner = StratifiedGroupKFold(n_splits=params["inner_splits"], shuffle=True, random_state=seed)
    idx_fit, idx_es = next(inner.split(x_tr, y_tr, groups_tr))
    probe = CatBoostClassifier(**params["catboost"], random_seed=seed, train_dir=train_dir)
    probe.fit(
        x_tr.iloc[idx_fit],
        y_tr.iloc[idx_fit],
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(x_tr.iloc[idx_es], y_tr.iloc[idx_es]),
    )
    final_params = {k: v for k, v in params["catboost"].items() if k != "early_stopping_rounds"}
    model = CatBoostClassifier(
        **{**final_params, "iterations": probe.get_best_iteration() + 1}, random_seed=seed, train_dir=train_dir
    )
    model.fit(x_tr, y_tr, cat_features=CATEGORICAL_FEATURES)
    return model


def main():
    with Experiment(__file__) as exp:
        params = exp.params
        train, test = load_data()
        x_train, y_train, x_test = build_features(train, test, extra=params["extra_features"])
        x_train, x_test = to_catboost(x_train), to_catboost(x_test)
        train_dir = str(exp.dir / "catboost_info")

        repeats = get_repeats(train)
        groups = passenger_group(train)
        oofs = np.zeros((len(repeats), len(x_train)))
        test_proba = []
        for r, folds in enumerate(repeats):
            for idx_tr, idx_va in folds:
                x_tr, y_tr = x_train.iloc[idx_tr], y_train.iloc[idx_tr]
                for seed in params["seeds"]:
                    model = fit_catboost(x_tr, y_tr, groups.iloc[idx_tr], params, seed, train_dir)
                    oofs[r, idx_va] += model.predict_proba(x_train.iloc[idx_va])[:, 1] / len(params["seeds"])

        # CV は fold モデルで測り、test の予測だけ train 全体で学習し直したモデルで作る。
        # fold モデルの平均は train の 80% しか見ていないため。全体モデルの性能は CV では測れず、LB で確かめる
        for seed in params["seeds"]:
            model = fit_catboost(x_train, y_train, groups, params, seed, train_dir)
            test_proba.append(model.predict_proba(x_test)[:, 1])

        exp.record_cv(repeated_fold_scores(y_train, oofs, repeats))
        exp.save_oof(oofs)
        # fold と seed ごとの予測ラベルの多数決ではなく確率を平均する。多数決は境界付近の情報を捨てる
        exp.save_test_predictions(test[ID_COL], np.mean(test_proba, axis=0))


if __name__ == "__main__":
    main()
