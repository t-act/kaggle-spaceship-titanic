from sklearn.metrics import accuracy_score

METRIC = {"name": "accuracy", "direction": "maximize"}
THRESHOLD = 0.5


def score(y_true, proba):
    return float(accuracy_score(y_true, proba > THRESHOLD))


def fold_scores(y_true, oof, folds):
    return [score(y_true.iloc[idx_va], oof[idx_va]) for _, idx_va in folds]
