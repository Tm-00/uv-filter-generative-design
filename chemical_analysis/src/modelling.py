import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.model_selection import train_test_split


DEFAULT_FEATURE_COLUMNS = [
    "MolWt",
    "LogP",
    "TPSA",
    "SA_Score",
    "Lmax",
    "OS"
]


def get_feature_columns(df=None, exclude=None):
    """
    Return the default set of descriptor feature columns used
    for modelling.

    If df is provided, only columns that actually exist in df
    are returned (in DEFAULT_FEATURE_COLUMNS order). Any column
    named in `exclude` (e.g. the modelling target) is dropped so
    it can never leak into the feature set.
    """

    columns = DEFAULT_FEATURE_COLUMNS

    if df is not None:

        columns = [
            c for c in columns
            if c in df.columns
        ]

    if exclude:

        exclude = (
            {exclude}
            if isinstance(exclude, str)
            else set(exclude)
        )

        columns = [
            c for c in columns
            if c not in exclude
        ]

    return columns


def prepare_xy(
    df,
    target,
    features=None
):
    """
    Extract feature matrix X and target vector y from df,
    dropping rows with missing values in either.
    """

    if features is None:

        features = get_feature_columns(
            df,
            exclude=target
        )

    subset = df[
        features + [target]
    ].dropna()

    X = subset[features]
    y = subset[target]

    return X, y, features


def train_random_forest(
    df,
    target,
    features=None,
    task="regression",
    test_size=0.2,
    random_state=42,
    **rf_kwargs
):
    """
    Train a Random Forest model to predict `target` from
    molecular descriptors.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the feature columns and the target column.

    target : str
        Column name to predict (e.g. "UV_Filter_Score", or a
        future experimental property such as "SPF" or "Lmax_exp").

    features : list of str, optional
        Descriptor columns to use. Defaults to
        get_feature_columns(df, exclude=target).

    task : "regression" or "classification"

    test_size : float
        Fraction of data held out for evaluation.

    random_state : int

    rf_kwargs : dict
        Extra keyword arguments passed to the sklearn estimator
        (e.g. n_estimators, max_depth).

    Returns
    -------
    dict with keys:
        model, features, X_train, X_test, y_train, y_test,
        y_pred_train, y_pred_test
    """

    X, y, features = prepare_xy(
        df,
        target,
        features
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    default_kwargs = dict(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1
    )

    default_kwargs.update(
        rf_kwargs
    )

    if task == "regression":

        model = RandomForestRegressor(
            **default_kwargs
        )

    elif task == "classification":

        model = RandomForestClassifier(
            **default_kwargs
        )

    else:

        raise ValueError(
            f"Unknown task type: {task}. "
            "Use 'regression' or 'classification'."
        )

    model.fit(
        X_train,
        y_train
    )

    y_pred_train = model.predict(
        X_train
    )

    y_pred_test = model.predict(
        X_test
    )

    return {
        "model": model,
        "target": target,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test
    }


def predict(
    model,
    df,
    features
):
    """
    Generate predictions for df using a fitted model.
    Rows with missing feature values are skipped; predictions
    are returned aligned to df's index (NaN where skipped).
    """

    subset = df[features]

    valid = subset.dropna().index

    preds = pd.Series(
        np.nan,
        index=df.index,
        name="Prediction"
    )

    preds.loc[valid] = model.predict(
        subset.loc[valid]
    )

    return preds


def get_feature_importance(
    model,
    features
):
    """
    Return feature importances from a fitted tree-based model
    as a DataFrame sorted from most to least important.
    """

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    ).reset_index(
        drop=True
    )

    return importance_df


def train_model_per_method(
    df,
    target,
    features=None,
    task="regression",
    **rf_kwargs
):
    """
    Train a separate Random Forest per Method group, useful for
    comparing which descriptors drive the score within each
    optimisation method (RL / HC / NO-RL).

    Returns
    -------
    dict keyed by method, each value the output of
    train_random_forest for that method's subset.
    """

    results = {}

    for method, group in df.groupby("Method"):

        results[method] = train_random_forest(
            group,
            target,
            features=features,
            task=task,
            **rf_kwargs
        )

    return results
