""" Explainable Machine Learning """

import pandas as pd
from sklearn.inspection import permutation_importance
from kaggle_house_prices.src.hyperparameter_tuning import (
    get_best_pipeline_rf,
    get_model_pipeline_xgb,
)
from kaggle_house_prices.src.features import get_x_y, load_data
import matplotlib.pyplot as plt
import seaborn as sns


def compute_and_show_rf_importance():
    df_train, df_test = load_data()
    X, y = get_x_y(data=df_train)
    model = get_best_pipeline_rf()
    model.fit(X, y)
    # show the feature importance according to permutation importance
    rf_permutation_importance = get_permutation_importance(model, X, y)
    plot_feature_importance(
        rf_permutation_importance, title="RF Permutation Importance"
    )


def compute_and_show_xgb_importance():
    df_train, df_test = load_data()
    X, y = get_x_y(data=df_train)
    model = get_model_pipeline_xgb()
    model.fit(X, y)
    # show the feature importance according to the permutation importance
    xgb_permutation_importance = get_permutation_importance(model, X, y)
    plot_feature_importance(
        xgb_permutation_importance, title="XGB Permutation Importance"
    )


def get_permutation_importance(model, X, y) -> pd.Series:
    """Get permutation importance for a model."""
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    mean_importances = pd.Series(result.importances_mean, index=X.columns).sort_values(
        ascending=False
    )
    return mean_importances


def main():
    compute_and_show_rf_importance()
    compute_and_show_xgb_importance()


def plot_feature_importance(
    importances: pd.Series, title: str = "Feature Importance", max_n_show: int = 30
):
    """Plot feature importance."""
    importances = importances.head(max_n_show)
    # make a large plot for better readability, depending on the number of features to show
    if len(importances) > 20:
        plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=importances.index, ax=ax)
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    main()
    print("DONE")
