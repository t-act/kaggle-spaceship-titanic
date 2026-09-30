from sklearn.metrics import accuracy_score

METRIC = {"name": "accuracy", "direction": "maximize"}
THRESHOLD = 0.5


def score(y_true, proba):
    return float(accuracy_score(y_true, proba > THRESHOLD))


def fold_scores(y_true, oof, folds):
    return [score(y_true.iloc[idx_va], oof[idx_va]) for _, idx_va in folds]


def repeated_fold_scores(y_true, oofs, repeats):
    """分割ごとの OOF（形は (分割数, 行数)）から、全分割の fold スコアを1列に並べて返す。"""
    return [s for oof, folds in zip(oofs, repeats, strict=True) for s in fold_scores(y_true, oof, folds)]
