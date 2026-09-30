import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.data import ID_COL, TARGET

# 分割が変わると実験間で CV を比較できなくなるため、全実験でこの設定に固定する
N_SPLITS = 5
# 分割 seed を1つに固定すると、その分割の当たり外れで全実験の CV が一律にずれる。
# seed=123 の分割は seed 0〜9 のどれよりも CV が高く出ていたため、複数の分割で平均する
CV_SEEDS = [0, 1, 2]

# results.json に記録し、同じ方式の実験どうしだけを比較する。分割を変えたら名前も変える
CV_SCHEME = "group-stratified-5fold-x3-seed012"
# 方式ごとの fold 数。旧方式は検証のために残す
FOLD_COUNTS = {
    CV_SCHEME: N_SPLITS * len(CV_SEEDS),
    "group-stratified-5fold-seed123": N_SPLITS,
    "stratified-5fold-seed123": N_SPLITS,
}


def passenger_group(df):
    # PassengerId は gggg_pp 形式で、gggg が同行グループ。train と test でグループは重ならない
    return df[ID_COL].str.split("_").str[0]


def get_repeats(train):
    """分割 seed ごとの fold のリストを返す。各行は分割ごとに1回ずつ検証側に入る。

    同じグループが学習側と検証側に分かれないよう、グループ単位で分割する。
    test のグループは train と重ならないため、グループをまたいで分割すると CV が test より楽観的になる。
    """
    groups = passenger_group(train)
    return [
        list(
            StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(
                np.zeros(len(train)), train[TARGET], groups=groups
            )
        )
        for seed in CV_SEEDS
    ]
