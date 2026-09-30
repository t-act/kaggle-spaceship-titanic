import numpy as np
from sklearn.model_selection import StratifiedKFold

# 分割が変わると実験間で CV を比較できなくなるため、全実験でこの設定に固定する
N_SPLITS = 5
CV_SEED = 123


def get_folds(y):
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    return list(splitter.split(np.zeros(len(y)), y))
