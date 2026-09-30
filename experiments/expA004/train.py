import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import get_folds, passenger_group
from src.data import ID_COL, TARGET, load_data
from src.experiment import Experiment
from src.metric import fold_scores

SPEND_SERVICES = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LUXURY_SERVICES = ["RoomService", "Spa", "VRDeck"]
BASIC_SERVICES = ["FoodCourt", "ShoppingMall"]
CABINS = ["Deck", "Num", "Side"]
NAMES = ["FirstName", "FamilyName", "Name"]
CATEGORICAL_COLS = ["HomePlanet", "Destination", "VIP", "Deck", "Side", *NAMES]


def impute_by_logic(df):
    # コールドスリープ中の人は支出できない → 支出の欠損を0で埋める
    cryo = df["CryoSleep"] == 1
    df.loc[cryo, SPEND_SERVICES] = df.loc[cryo, SPEND_SERVICES].fillna(0)

    # 支出がある人はコールドスリープしていない → CryoSleep の欠損を0で埋める
    has_spend = df[SPEND_SERVICES].sum(axis=1) > 0
    df.loc[df["CryoSleep"].isna() & has_spend, "CryoSleep"] = 0.0
    return df


def split_name(df):
    return df["Name"].str.split(" ", expand=True).set_axis(["FirstName", "FamilyName"], axis=1)


def build_features(df, family_counts, logical_imputation):
    df = df.copy()

    # NaN を残したいので astype(bool) ではなく map で数値化する
    df["CryoSleep"] = df["CryoSleep"].map({True: 1.0, False: 0.0})

    df[CABINS] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = df["Num"].astype("float")

    if logical_imputation:
        df = impute_by_logic(df)

    df["TotalSpend"] = df[SPEND_SERVICES].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(float)
    df["SpendCount"] = (df[SPEND_SERVICES] > 0).sum(axis=1)
    df["LuxurySpend"] = df[LUXURY_SERVICES].sum(axis=1)
    df["BasicSpend"] = df[BASIC_SERVICES].sum(axis=1)

    df[["FirstName", "FamilyName"]] = split_name(df)
    df["FamilySize"] = df["FamilyName"].map(family_counts).astype("float")

    # CatBoost はカテゴリ列の NaN を受け付けないため文字列にする。NaN は "nan" という1水準になる
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(str)
    return df


def fit_catboost(x_tr, y_tr, groups_tr, cat_features, params, seed, train_dir):
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
        cat_features=cat_features,
        eval_set=(x_tr.iloc[idx_es], y_tr.iloc[idx_es]),
    )
    n_iterations = probe.get_best_iteration() + 1

    final_params = {k: v for k, v in params["catboost"].items() if k != "early_stopping_rounds"}
    model = CatBoostClassifier(**{**final_params, "iterations": n_iterations}, random_seed=seed, train_dir=train_dir)
    model.fit(x_tr, y_tr, cat_features=cat_features)
    return model, n_iterations


def main():
    with Experiment(__file__) as exp:
        params = exp.params
        train, test = load_data()

        # 同じ家族が train と test に分かれて乗っているため、両方を合わせて数える
        family_counts = split_name(pd.concat([train, test]))["FamilyName"].value_counts()
        train = build_features(train, family_counts, params["logical_imputation"])
        test = build_features(test, family_counts, params["logical_imputation"])

        x_train, y_train = train[params["features"]], train[TARGET]
        x_test = test[params["features"]]

        cat_features = [c for c in params["features"] if c in CATEGORICAL_COLS]
        folds = get_folds(train)
        groups = passenger_group(train)
        oof = np.zeros(len(x_train))
        test_proba = []
        for idx_tr, idx_va in folds:
            x_tr, y_tr = x_train.iloc[idx_tr], y_train.iloc[idx_tr]
            x_va = x_train.iloc[idx_va]
            for seed in params["seeds"]:
                model, n_iterations = fit_catboost(
                    x_tr, y_tr, groups.iloc[idx_tr], cat_features, params, seed, str(exp.dir / "catboost_info")
                )
                print(f"seed={seed}: iterations={n_iterations}")
                oof[idx_va] += model.predict_proba(x_va)[:, 1] / len(params["seeds"])
                test_proba.append(model.predict_proba(x_test)[:, 1])

        exp.record_cv(fold_scores(y_train, oof, folds))
        exp.save_oof(oof)
        # fold と seed ごとの予測ラベルの多数決ではなく確率を平均する。多数決は境界付近の情報を捨てる
        exp.save_test_predictions(test[ID_COL], np.mean(test_proba, axis=0))


if __name__ == "__main__":
    main()
