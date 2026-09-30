import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.data import ID_COL, TARGET

# 分割が変わると実験間で CV を比較できなくなるため、全実験でこの設定に固定する
N_SPLITS = 5
CV_SEED = 123

# results.json に記録し、同じ方式の実験どうしだけを比較する。分割を変えたら名前も変える
CV_SCHEME = "group-stratified-5fold-seed123"
LEGACY_CV_SCHEMES = ["stratified-5fold-seed123"]


def passenger_group(df):
    # PassengerId は gggg_pp 形式で、gggg が同行グループ。train と test でグループは重ならない
    return df[ID_COL].str.split("_").str[0]


def get_folds(train):
    """同じグループが学習側と検証側に分かれないよう、グループ単位で分割する。

    test のグループは train と重ならないため、グループをまたいで分割すると CV が test より楽観的になる。
    """
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    return list(splitter.split(np.zeros(len(train)), train[TARGET], groups=passenger_group(train)))
