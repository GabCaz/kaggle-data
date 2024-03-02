"""
Model
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor
from kaggle_house_prices.out.path_out import OUT_DIR
from kaggle_house_prices.src.features import (
    NOMINAL_FEATURES,
    ORDERED_LEVELS,
    QUANT_FEATURES,
    Y_COL,
    get_x_y,
    load_data,
)


def get_preprocessor() -> ColumnTransformer:
    # decide what treatment to apply for what column
    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[ORDERED_LEVELS[col] for col in ORDERED_LEVELS.keys()]
                ),
            ),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    processor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, QUANT_FEATURES),
            ("nom", nominal_transformer, NOMINAL_FEATURES),
            ("ord", ordinal_transformer, list(ORDERED_LEVELS.keys())),
        ]
    )
    return processor


def get_encode_pipeline() -> ColumnTransformer:
    """Get the pipeline to encode the features."""
    # apply one-hot encoding to nominal features
    one_hot_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    # encode ordinal features (appropriate for tree-based model)
    ordinal_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ORDERED_LEVELS[col] for col in ORDERED_LEVELS.keys()]
                ),
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", one_hot_transformer, NOMINAL_FEATURES),
            ("ord", ordinal_transformer, list(ORDERED_LEVELS.keys())),
        ]
    )
    return preprocessor


def score_dataset(X, y, model):
    score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


def score_result(y_true, y_pred):
    """Mean squared error"""
    return np.mean((y_true - y_pred) ** 2)


def export_kaggle_submission(df_test, X, y, my_pipeline, fname="submission.csv"):
    # fit the model on ALL the data for the final submission
    my_pipeline.fit(X, y)
    # and make predictions on the test set
    X_test = df_test.drop(columns=[Y_COL])
    y_pred = np.exp(my_pipeline.predict(X_test))
    # save the predictions for submission
    output = pd.DataFrame({"Id": X_test.index, "SalePrice": y_pred}).set_index("Id")
    output.to_csv(OUT_DIR / fname, index=True)


def get_model_pipeline(*args, **kwargs):
    model = XGBRegressor(**kwargs)
    preprocessor = get_preprocessor()
    my_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return my_pipeline


def main():
    full_model = get_model_pipeline()
    df_train, df_test = load_data()
    X, y = get_x_y(data=df_train)

    # cross-validation score
    baseline_score = score_dataset(X, y, model=full_model)
    print(f"Baseline score: {baseline_score:.5f} RMSLE")

    # export baseline submission
    export_kaggle_submission(df_test, X, y, full_model)


if __name__ == "__main__":
    main()
    print("DONE")
