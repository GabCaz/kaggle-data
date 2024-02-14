"""
Feature engineering on the House price Kaggle competition.

Source:
https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_regression

# Get the current file's directory as a Path object
PATH_DATA = Path(__file__).resolve().parent.parent / "data"


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


def main():
    df_train, _ = load_data()
    X, y = get_x_y(df_train)
    print(f"Most important feature:\n{make_mi_scores(X, y).to_frame().head(10)}")


if __name__ == "__main__":
    main()
    print("DONE")
