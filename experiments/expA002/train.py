import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.cv import get_folds
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
        oof = np.zeros(len(x_train))
        test_proba = []
        for idx_tr, idx_va in folds:
            x_tr, y_tr = x_train.iloc[idx_tr], y_train.iloc[idx_tr]
            x_va, y_va = x_train.iloc[idx_va], y_train.iloc[idx_va]
            for seed in params["seeds"]:
                model = CatBoostClassifier(
                    **params["catboost"], random_seed=seed, train_dir=str(exp.dir / "catboost_info")
                )
                model.fit(x_tr, y_tr, cat_features=cat_features, eval_set=(x_va, y_va))
                oof[idx_va] += model.predict_proba(x_va)[:, 1] / len(params["seeds"])
                test_proba.append(model.predict_proba(x_test)[:, 1])

        exp.record_cv(fold_scores(y_train, oof, folds))
        exp.save_oof(oof)
        # fold と seed ごとの予測ラベルの多数決ではなく確率を平均する。多数決は境界付近の情報を捨てる
        exp.save_test_predictions(test[ID_COL], np.mean(test_proba, axis=0))


if __name__ == "__main__":
    main()
