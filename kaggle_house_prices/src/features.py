"""
Feature engineering on the House price Kaggle competition.

Source:
https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor
from kaggle_house_prices.out.path_out import OUT_DIR

from kaggle_house_prices.src.read_data import PATH_DATA

Y_COL = "SalePrice"


# the quantitative features
QUANT_FEATURES = [
    "GarageArea",
    "MiscVal",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "YrSold",
    "Fireplaces",
    "2ndFlrSF",
    "BedroomAbvGr",
    "OpenPorchSF",
    "YearRemodAdd",
    "YearBuilt",
    "KitchenAbvGr",
    "GarageCars",
    "FullBath",
    "BsmtFinSF1",
    "EnclosedPorch",
    "WoodDeckSF",
    "BsmtFinSF2",
    "MoSold",
    "ScreenPorch",
    "GrLivArea",
    "BsmtFullBath",
    "1stFlrSF",
    "LowQualFinSF",
    "3SsnPorch",
    "PoolArea",
    "MasVnrArea",
    "LotArea",
    "LotFrontage",
    "BsmtHalfBath",
    "TotRmsAbvGrd",
    "GarageYrBlt",
    "HalfBath",
]

# The nominative (unordered) categorical features
NOMINAL_FEATURES = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]


# The ordinal (ordered) categorical features

# Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ORDERED_LEVELS = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Add a None level for missing values
ORDERED_LEVELS = {key: ["None"] + value for key, value in ORDERED_LEVELS.items()}


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


def encode(df):
    # Nominal categories
    for name in NOMINAL_FEATURES:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name] = df[name].cat.add_categories("None")
    # Ordinal categories
    for name, levels in ORDERED_LEVELS.items():
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
    return df


def get_x_y(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = data.drop(columns=[Y_COL])
    y = np.log(data[Y_COL])
    return X, y


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(
        X.fillna(-1), y, discrete_features=discrete_features, random_state=0
    )
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def load_data():
    # Read data
    data_dir = PATH_DATA
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])
    # Preprocessing
    df = encode(df)
    # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    return df_train, df_test


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


def main():
    model = XGBRegressor()
    df_train, df_test = load_data()
    X, y = get_x_y(data=df_train)
    preprocessor = get_preprocessor()
    my_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # look into one example of splitting the data
    df_train_train, df_train_valid = train_test_split(
        df_train, test_size=0.2, random_state=42
    )
    X_train_train, y_train_train = get_x_y(data=df_train_train)
    X_train_valid, y_train_valid = get_x_y(data=df_train_valid)

    my_pipeline.fit(X_train_train, y_train_train)
    y_pred_valid = my_pipeline.predict(X_train_valid)
    plt.scatter(y_train_valid, y_pred_valid, alpha=0.2)
    plt.show()

    # cross-validation score
    baseline_score = score_dataset(X, y, model=my_pipeline)
    print(f"Baseline score: {baseline_score:.5f} RMSLE")

    # export baseline submission
    export_kaggle_submission(df_test, X, y, my_pipeline)


def export_kaggle_submission(df_test, X, y, my_pipeline):
    my_pipeline.fit(X, y)
    # and make predictions on the test set
    X_test = df_test.drop(columns=[Y_COL])
    y_pred = np.exp(my_pipeline.predict(X_test))
    # save the predictions for submission
    output = pd.DataFrame({"Id": X_test.index, "SalePrice": y_pred}).set_index("Id")
    output.to_csv(OUT_DIR / "submission.csv", index=True)


if __name__ == "__main__":
    main()
    print("DONE")
