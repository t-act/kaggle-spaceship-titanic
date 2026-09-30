import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import get_repeats, passenger_group
from src.data import ID_COL, TARGET, load_data
from src.experiment import Experiment
from src.metric import repeated_fold_scores

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


GROUP_FEATURES = ["GroupSize", "GroupCryoRate", "GroupSpend", "GroupSpendOthers", "CabinSize"]


def add_group_features(train, test):
    """同行グループと同じ Cabin の集計を足す。

    train と test でグループは重ならないが、Cabin の数え漏れを防ぐため両方を合わせて集計する。
    集計に使う列だけを結合する。全列を結合すると category 型が object 型に変わるため。
    目的変数は使わない。グループ単位の CV でも、目的変数の集計は同じ学習側の中で漏れを生むため。
    """
    cols = [ID_COL, "CryoSleep", "TotalSpend", "Cabin"]
    both = pd.concat([train[cols], test[cols]], ignore_index=True)
    group = passenger_group(both)
    feats = pd.DataFrame(index=both.index)
    feats["GroupSize"] = group.map(group.value_counts()).astype(float)
    feats["GroupCryoRate"] = both.groupby(group)["CryoSleep"].transform("mean")
    feats["GroupSpend"] = both.groupby(group)["TotalSpend"].transform("sum")
    # 自分の支出は TotalSpend で既に持っているため、同行者の分だけを別に持つ
    feats["GroupSpendOthers"] = feats["GroupSpend"] - both["TotalSpend"]
    feats["CabinSize"] = both["Cabin"].map(both["Cabin"].value_counts()).astype(float)

    train = train.copy()
    test = test.copy()
    train[GROUP_FEATURES] = feats.iloc[: len(train)].to_numpy()
    test[GROUP_FEATURES] = feats.iloc[len(train) :].to_numpy()
    return train, test


def fit_lgb(x_tr, y_tr, groups_tr, params):
    """木の数を検証 fold ではなく、学習側を内側で分けた検証で決める。

    検証 fold で early stopping すると、木の数が検証 fold に合わせて選ばれ OOF が楽観的になるため。
    学習し直すときにデータが増える分だけ木の数を増やす補正はしない。補正の係数が新たな調整対象になるため。
    """
    inner = StratifiedGroupKFold(n_splits=params["inner_splits"], shuffle=True, random_state=params["inner_seed"])
    idx_fit, idx_es = next(inner.split(x_tr, y_tr, groups_tr))
    probe = lgb.LGBMClassifier(**params["lgb"])
    probe.fit(
        x_tr.iloc[idx_fit],
        y_tr.iloc[idx_fit],
        eval_X=x_tr.iloc[idx_es],
        eval_y=y_tr.iloc[idx_es],
        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)],
    )
    model = lgb.LGBMClassifier(**{**params["lgb"], "n_estimators": probe.best_iteration_})
    model.fit(x_tr, y_tr)
    return model


def main():
    with Experiment(__file__) as exp:
        params = exp.params
        train, test = load_data()

        # 同じ家族が train と test に分かれて乗っているため、両方を合わせて数える
        family_counts = split_name(pd.concat([train, test]))["FamilyName"].value_counts()
        train = build_features(train, family_counts, params["logical_imputation"])
        test = build_features(test, family_counts, params["logical_imputation"])
        train, test = add_group_features(train, test)

        x_train, y_train = train[params["features"]], train[TARGET]
        x_test = test[params["features"]]

        repeats = get_repeats(train)
        groups = passenger_group(train)
        oofs = np.zeros((len(repeats), len(x_train)))
        test_proba = []
        for r, folds in enumerate(repeats):
            for idx_tr, idx_va in folds:
                x_tr, y_tr = x_train.iloc[idx_tr], y_train.iloc[idx_tr]
                model = fit_lgb(x_tr, y_tr, groups.iloc[idx_tr], params)
                oofs[r, idx_va] = model.predict_proba(x_train.iloc[idx_va])[:, 1]
                test_proba.append(model.predict_proba(x_test)[:, 1])

        exp.record_cv(repeated_fold_scores(y_train, oofs, repeats))
        exp.save_oof(oofs)
        # fold ごとの予測ラベルの多数決ではなく確率を平均する。多数決は5票のため境界付近の情報を捨てる
        exp.save_test_predictions(test[ID_COL], np.mean(test_proba, axis=0))


if __name__ == "__main__":
    main()
