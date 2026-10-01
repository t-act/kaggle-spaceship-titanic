import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.cv import get_repeats
from src.data import ID_COL, load_data
from src.experiment import Experiment
from src.features import CATEGORICAL_FEATURES, FEATURES, SPEND_SERVICES, build_features
from src.metric import repeated_fold_scores

NUMERIC = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]
# 支出は 0 に集中し裾が長いため、線形モデルでは対数にしないと少数の高額な人に係数が引っ張られる
SKEWED = [*SPEND_SERVICES, "TotalSpend"]


def make_model(params, seed):
    numeric = make_pipeline(
        FunctionTransformer(
            lambda x: x.assign(**{c: np.log1p(x[c]) for c in SKEWED if c in x}), feature_names_out="one-to-one"
        ),
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore")
    )
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL_FEATURES)])
    mlp = {**params["mlp"], "hidden_layer_sizes": tuple(params["mlp"]["hidden_layer_sizes"])}
    return make_pipeline(pre, MLPClassifier(**mlp, random_state=seed))


def main():
    with Experiment(__file__) as exp:
        params = exp.params
        train, test = load_data()
        x_train, y_train, x_test = build_features(train, test)
        # one-hot に渡すため、category 型を欠損を保ったまま object 型に戻す
        x_train = x_train.astype(dict.fromkeys(CATEGORICAL_FEATURES, object))
        x_test = x_test.astype(dict.fromkeys(CATEGORICAL_FEATURES, object))

        repeats = get_repeats(train)
        oofs = np.zeros((len(repeats), len(x_train)))
        test_proba = []
        for r, folds in enumerate(repeats):
            for idx_tr, idx_va in folds:
                for seed in params["seeds"]:
                    model = make_model(params, seed).fit(x_train.iloc[idx_tr], y_train.iloc[idx_tr])
                    oofs[r, idx_va] += model.predict_proba(x_train.iloc[idx_va])[:, 1] / len(params["seeds"])
                    test_proba.append(model.predict_proba(x_test)[:, 1])

        exp.record_cv(repeated_fold_scores(y_train, oofs, repeats))
        exp.save_oof(oofs)
        exp.save_test_predictions(test[ID_COL], np.mean(test_proba, axis=0))


if __name__ == "__main__":
    main()
