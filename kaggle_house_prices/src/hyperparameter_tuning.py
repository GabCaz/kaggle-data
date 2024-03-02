""" Start hyperparameter tuning for the model. """
from copy import deepcopy
from itertools import product
from typing import Dict, List

import pandas as pd
from xgboost import XGBRegressor

from kaggle_house_prices.src.features import get_x_y, load_data
from kaggle_house_prices.src.model_pipeline import (
    get_model_pipeline_xgb,
    get_preprocessor,
    score_result,
    export_kaggle_submission,
    get_model_pipeline_rf,
)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from kaggle_house_prices.src.config import in_out


def hyperparameter_tuning_rf():
    df_train, df_test = load_data()
    X, y = get_x_y(data=df_train)
    model_pipeline = get_model_pipeline_rf()

    # parameters to search
    param_grid = {
        "n_estimators": list(range(100, 500, 100)),
        "max_depth": [3, 6, 10],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [1, 4, 8],
    }
    grid_renamed_for_model = {
        f"model__{key}": value for key, value in param_grid.items()
    }

    # grid search
    GS = GridSearchCV(
        model_pipeline,
        grid_renamed_for_model,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
    )
    GS.fit(X, y)
    print(f"Best parameters: {GS.best_params_}")

    # drop the best parameters to a YAML file
    best_params = GS.best_params_
    in_out.write_yaml(best_params, "best_params_rf.yaml")

    print(f"Best score: {GS.best_score_:.5f}")
    df_all_models = pd.DataFrame(GS.cv_results_).sort_values("rank_test_score")
    print(df_all_models[["params", "mean_test_score", "rank_test_score"]].head(10))

    # get the best performing model
    best_model = GS.best_estimator_
    export_kaggle_submission(
        df_test, X, y, best_model, fname="random_forest_grid_search_cv.csv"
    )


def main():
    hyperparameter_tuning_rf()
    hyperparameter_tuning_xgboost()


def hyperparameter_tuning_xgboost():
    df_train, df_test = load_data()
    X, y = get_x_y(data=df_train)
    model_pipeline = get_model_pipeline_xgb()
    # parameters to search
    param_grid = {
        "n_estimators": list(range(100, 500, 100)),
        "max_depth": [3, 6, 10],
        "gamma": [0, 0.01, 0.1],
        "learning_rate": [0.1, 0.3, 0.5],
    }
    grid_renamed_for_model = {
        f"model__{key}": value for key, value in param_grid.items()
    }
    # try using early stopping
    preprocessor = get_preprocessor()
    params_grid_early_stopping = deepcopy(param_grid)
    params_grid_early_stopping["n_estimators"] = [1000]
    res_grid_search = grid_search_with_early_stopping(
        preprocessor=preprocessor,
        params=params_grid_early_stopping,
        X=X,
        y=y,
        cv_num=5,
    )
    best_res_early_stopping = res_grid_search.sort_values("score")
    print(best_res_early_stopping.head(10))
    best_params_early_stopping = best_res_early_stopping.iloc[0, :]["parameters"]
    best_model_early_stopping = get_model_pipeline_xgb(**best_params_early_stopping)
    export_kaggle_submission(
        df_test, X, y, best_model_early_stopping, fname="xgboost_early_stopping.csv"
    )
    # grid search
    GS = GridSearchCV(
        model_pipeline,
        grid_renamed_for_model,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
    )
    GS.fit(X, y)

    # drop the best parameters to a YAML file
    best_params = GS.best_params_
    in_out.write_yaml(best_params, "best_params_xgboost.yaml")

    print(f"Best parameters: {GS.best_params_}")
    print(f"Best score: {GS.best_score_:.5f}")
    df_all_models = pd.DataFrame(GS.cv_results_).sort_values("rank_test_score")
    print(df_all_models[["params", "mean_test_score", "rank_test_score"]].head(10))
    # get the best performing model
    best_model = GS.best_estimator_
    export_kaggle_submission(
        df_test, X, y, best_model, fname="xgboost_grid_search_cv.csv"
    )


def get_best_pipeline_xgboost():
    best_params = in_out.read_yaml("best_params_xgboost.yaml")
    # left strip the model__ prefix
    best_params = {key.split("__")[1]: value for key, value in best_params.items()}
    best_model = get_model_pipeline_xgb(**best_params)
    return best_model


def get_best_pipeline_rf():
    best_params = in_out.read_yaml("best_params_rf.yaml")
    # left strip the model__ prefix
    best_params = {key.split("__")[1]: value for key, value in best_params.items()}
    best_model = get_model_pipeline_rf(**best_params)
    return best_model


def grid_search_with_early_stopping(
    preprocessor,
    params: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_num: int = 5,
    early_stopping_rounds: int = 10,
) -> pd.DataFrame:
    """Grid search with early stopping."""
    cv = KFold(n_splits=cv_num, shuffle=True, random_state=0)
    res = list()
    iter_params = _iter_params(params)
    X_test, X_train = train_test_split(X, random_state=0)
    y_test, y_train = y[X_test.index], y[X_train.index]

    # for each product of parameters
    for (cand_idx, parameters), (split_idx, (train_index, test_index)) in product(
        enumerate(iter_params), enumerate(cv.split(X_train, y_train))
    ):
        # split the data
        X_train_train, X_valid_train = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        preprocessor.fit(X_train_train)
        model = XGBRegressor(**parameters)
        X_valid_preprocessed = preprocessor.transform(X_valid_train)
        X_train_preprocessed = preprocessor.transform(X_train_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        y_valid_pred = model.fit(
            X_train_preprocessed,
            y_train,
            eval_set=[(X_valid_preprocessed, y_valid)],
            early_stopping_rounds=early_stopping_rounds,
        ).predict(X_test_preprocessed)
        score_result_baseline = score_result(y_test, y_valid_pred)
        res.append(
            {
                "cand_idx": cand_idx,
                "split_idx": split_idx,
                "score": score_result_baseline,
                "parameters": parameters,
            }
        )
    res_df = pd.DataFrame(res)
    return res_df


def _iter_params(params: Dict[str, List]):
    """Iterate over all combinations of parameters."""
    keys = params.keys()
    for values in product(*params.values()):
        yield dict(zip(keys, values))


if __name__ == "__main__":
    main()
    print("DONE")
