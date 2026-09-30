import lightgbm as lgb
import numpy as np
import pandas as pd

from src.cv import get_folds
from src.data import ID_COL, TARGET, load_data
from src.experiment import Experiment
from src.metric import THRESHOLD, fold_scores

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

    # train と test で category の水準が揃わなくても、LightGBM は predict 時に学習時の水準へ読み替える
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")
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

        folds = get_folds(y_train)
        oof = np.zeros(len(x_train))
        test_proba = []
        for idx_tr, idx_va in folds:
            x_tr, y_tr = x_train.iloc[idx_tr], y_train.iloc[idx_tr]
            x_va, y_va = x_train.iloc[idx_va], y_train.iloc[idx_va]
            model = lgb.LGBMClassifier(**params["lgb"])
            model.fit(
                x_tr,
                y_tr,
                eval_X=x_va,
                eval_y=y_va,
                callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)],
            )
            oof[idx_va] = model.predict_proba(x_va)[:, 1]
            test_proba.append(model.predict_proba(x_test)[:, 1])

        exp.record_cv(fold_scores(y_train, oof, folds))
        exp.save_oof(oof)
        # fold ごとの予測ラベルの多数決ではなく確率を平均する。多数決は5票のため境界付近の情報を捨てる
        exp.save_submission(pd.DataFrame({ID_COL: test[ID_COL], TARGET: np.mean(test_proba, axis=0) > THRESHOLD}))


if __name__ == "__main__":
    main()
