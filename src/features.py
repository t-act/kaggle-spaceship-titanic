"""全実験で共有する確定済みの特徴量。

モデルの改善に集中するため、特徴量はここで固定する。変えるときは実験ではなく、この版を上げる判断として扱う。
グループの集計（expA007・expA008）と確実な値での欠損補完（exp003）は CV を上げなかったため入れていない。
"""

import pandas as pd

from src.data import ID_COL, TARGET

FEATURE_VERSION = "v1"

SPEND_SERVICES = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CABINS = ["Deck", "Num", "Side"]
CATEGORICAL_FEATURES = ["HomePlanet", "Destination", "VIP", "Deck", "Side"]
FEATURES = ["Age", "TotalSpend", "HomePlanet", "CryoSleep", "Destination", "VIP", *CABINS, *SPEND_SERVICES]


def _group_features(train, test):
    """PassengerId の先頭4桁から作る追加列。v2 の候補として実験の config で選ぶ。

    Group は数値のまま使う。train と test で番号が重ならず、カテゴリにすると test では未知の水準になるため。
    GroupSize は train と test を合わせて数える。グループは片方にしか現れないが、数え方を揃えるため。
    """
    ids = pd.concat([train[ID_COL], test[ID_COL]], ignore_index=True)
    group = ids.str.split("_").str[0]
    size = group.map(group.value_counts())
    extra = pd.DataFrame(
        {"Group": group.astype(float), "GroupSize": size.astype(float), "Solo": (size == 1).astype(float)}
    )
    return extra.iloc[: len(train)].reset_index(drop=True), extra.iloc[len(train) :].reset_index(drop=True)


EXTRA_FEATURES = ["Group", "GroupSize", "Solo"]


def _build(df):
    df = df.copy()
    # NaN を残したいので astype(bool) ではなく map で数値化する
    df["CryoSleep"] = df["CryoSleep"].map({True: 1.0, False: 0.0})
    df[CABINS] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = df["Num"].astype("float")
    df["TotalSpend"] = df[SPEND_SERVICES].sum(axis=1)
    # train と test で category の水準が揃わなくても、LightGBM は predict 時に学習時の水準へ読み替える
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype("category")
    return df[FEATURES]


def build_features(train, test, extra=()):
    """(x_train, y_train, x_test) を返す。カテゴリ列は category 型。extra には EXTRA_FEATURES から選んだ列名を渡す。"""
    unknown = set(extra) - set(EXTRA_FEATURES)
    if unknown:
        raise ValueError(f"未定義の追加列: {sorted(unknown)}")
    x_train, x_test = _build(train).reset_index(drop=True), _build(test).reset_index(drop=True)
    if extra:
        extra_train, extra_test = _group_features(train, test)
        x_train = pd.concat([x_train, extra_train[list(extra)]], axis=1)
        x_test = pd.concat([x_test, extra_test[list(extra)]], axis=1)
    return x_train, train[TARGET].reset_index(drop=True), x_test


def to_catboost(x):
    # CatBoost はカテゴリ列の NaN を受け付けないため文字列にする。NaN は "nan" という1水準になる
    return x.astype(dict.fromkeys(CATEGORICAL_FEATURES, str))
