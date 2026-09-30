"""全実験で共有する確定済みの特徴量。

モデルの改善に集中するため、特徴量はここで固定する。変えるときは実験ではなく、この版を上げる判断として扱う。
グループの集計（expA007・expA008）と確実な値での欠損補完（exp003）は CV を上げなかったため入れていない。
"""

from src.data import TARGET

FEATURE_VERSION = "v1"

SPEND_SERVICES = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CABINS = ["Deck", "Num", "Side"]
CATEGORICAL_FEATURES = ["HomePlanet", "Destination", "VIP", "Deck", "Side"]
FEATURES = ["Age", "TotalSpend", "HomePlanet", "CryoSleep", "Destination", "VIP", *CABINS, *SPEND_SERVICES]


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


def build_features(train, test):
    """(x_train, y_train, x_test) を返す。カテゴリ列は category 型。"""
    return _build(train), train[TARGET], _build(test)


def to_catboost(x):
    # CatBoost はカテゴリ列の NaN を受け付けないため文字列にする。NaN は "nan" という1水準になる
    return x.astype(dict.fromkeys(CATEGORICAL_FEATURES, str))
