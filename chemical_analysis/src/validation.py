import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import KFold, RepeatedKFold


def regression_metrics(
    y_true,
    y_pred
):
    """
    Standard QSAR regression metrics.

    Returns
    -------
    dict with R2, RMSE, MAE
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }


def cross_validate_model(
    estimator,
    X,
    y,
    k=5,
    random_state=42
):
    """
    k-fold cross validation for a regression estimator.

    Parameters
    ----------
    estimator : unfitted sklearn-compatible estimator
        Will be cloned for each fold, so the original is
        never mutated.

    X, y : array-like / DataFrame / Series

    k : int
        Number of folds.

    Returns
    -------
    dict with:
        fold_metrics : DataFrame, one row per fold (R2, RMSE, MAE)
        mean : dict of mean metric values
        std  : dict of std metric values
    """

    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    kf = KFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        kf.split(X),
        start=1
    ):

        model = clone(
            estimator
        )

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        y_pred = model.predict(
            X.iloc[test_idx]
        )

        metrics = regression_metrics(
            y.iloc[test_idx],
            y_pred
        )

        metrics["Fold"] = fold_i

        fold_results.append(metrics)

    fold_df = pd.DataFrame(
        fold_results
    )[["Fold", "R2", "RMSE", "MAE"]]

    mean = fold_df[["R2", "RMSE", "MAE"]].mean().to_dict()
    std = fold_df[["R2", "RMSE", "MAE"]].std().to_dict()

    return {
        "fold_metrics": fold_df,
        "mean": mean,
        "std": std
    }


def repeated_cross_validate_model(
    estimator,
    X,
    y,
    k=5,
    n_repeats=10,
    random_state=42
):
    """
    Repeated k-fold cross validation, giving a more stable
    estimate of model performance and its variance than a
    single k-fold split.

    Returns
    -------
    dict with:
        fold_metrics : DataFrame, one row per fold per repeat
        mean : dict of mean metric values across all folds
        std  : dict of std metric values across all folds
    """

    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    rkf = RepeatedKFold(
        n_splits=k,
        n_repeats=n_repeats,
        random_state=random_state
    )

    fold_results = []

    for i, (train_idx, test_idx) in enumerate(
        rkf.split(X),
        start=1
    ):

        model = clone(
            estimator
        )

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        y_pred = model.predict(
            X.iloc[test_idx]
        )

        metrics = regression_metrics(
            y.iloc[test_idx],
            y_pred
        )

        metrics["Repeat"] = (
            (i - 1) // k
        ) + 1

        metrics["Fold"] = (
            (i - 1) % k
        ) + 1

        fold_results.append(metrics)

    fold_df = pd.DataFrame(
        fold_results
    )[["Repeat", "Fold", "R2", "RMSE", "MAE"]]

    mean = fold_df[["R2", "RMSE", "MAE"]].mean().to_dict()
    std = fold_df[["R2", "RMSE", "MAE"]].std().to_dict()

    return {
        "fold_metrics": fold_df,
        "mean": mean,
        "std": std
    }


def y_scrambling(
    estimator,
    X,
    y,
    n_iterations=100,
    k=5,
    random_state=42
):
    """
    Y-scrambling (response permutation) test.

    Repeatedly shuffles the target vector and evaluates the
    model with k-fold cross-validated R2 (Q2), then compares
    against the true model's cross-validated R2.

    Cross-validated R2 is used rather than in-sample R2 because
    flexible models such as Random Forest can fit noise closely
    in-sample even when the target is pure chance; an out-of-fold
    score is needed to actually detect a chance correlation. A
    real, non-chance relationship between descriptors and target
    should show scrambled Q2 values clustered near zero (or
    negative), well below the true model's Q2.

    Returns
    -------
    dict with:
        scrambled_r2 : list of cross-validated R2 values from
            scrambled fits
        mean_scrambled_r2 : float
        max_scrambled_r2 : float
        true_r2 : float, cross-validated R2 of the model fit on
            the real target
    """

    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    rng = np.random.RandomState(
        random_state
    )

    true_cv = cross_validate_model(
        estimator,
        X,
        y,
        k=k,
        random_state=random_state
    )

    true_r2 = true_cv["mean"]["R2"]

    scrambled_r2 = []

    for _ in range(n_iterations):

        y_shuffled = y.sample(
            frac=1.0,
            random_state=rng.randint(0, 1_000_000)
        ).reset_index(
            drop=True
        )

        cv = cross_validate_model(
            estimator,
            X,
            y_shuffled,
            k=k,
            random_state=random_state
        )

        scrambled_r2.append(
            cv["mean"]["R2"]
        )

    return {
        "scrambled_r2": scrambled_r2,
        "mean_scrambled_r2": float(np.mean(scrambled_r2)),
        "max_scrambled_r2": float(np.max(scrambled_r2)),
        "true_r2": true_r2
    }


def applicability_domain(
    X_train,
    X_query,
    warning_threshold_multiplier=3.0
):
    """
    Leverage-based applicability domain (Williams plot approach).

    For each compound in X_query, computes its leverage h with
    respect to the training set descriptor space:

        h_i = x_i (X^T X)^-1 x_i^T

    A warning leverage h* = 3(p+1)/n is used as the threshold
    (n = training set size, p = number of descriptors). Compounds
    with h_i > h* are considered outside the applicability domain
    (extrapolation) and predictions for them should be treated
    with caution.

    Parameters
    ----------
    X_train : array-like / DataFrame
        Descriptor matrix used to train the model.

    X_query : array-like / DataFrame
        Descriptor matrix for compounds to assess (can be the
        training set itself, the test set, or new molecules).

    warning_threshold_multiplier : float
        Multiplier used in h* = multiplier * (p+1) / n.
        3.0 is the standard QSAR convention.

    Returns
    -------
    DataFrame indexed as X_query, with columns:
        Leverage, Warning_Leverage, In_Domain
    """

    X_train_arr = np.asarray(
        X_train,
        dtype=float
    )

    X_query_df = pd.DataFrame(
        X_query
    )

    X_query_arr = X_query_df.to_numpy(
        dtype=float
    )

    n, p = X_train_arr.shape

    warning_leverage = (
        warning_threshold_multiplier
        * (p + 1)
        / n
    )

    # (X^T X)^-1, with a small ridge term for numerical
    # stability if the descriptor matrix is near-singular.
    gram = X_train_arr.T @ X_train_arr

    gram += np.eye(p) * 1e-8

    gram_inv = np.linalg.pinv(
        gram
    )

    leverages = np.einsum(
        "ij,jk,ik->i",
        X_query_arr,
        gram_inv,
        X_query_arr
    )

    result = pd.DataFrame(
        {
            "Leverage": leverages,
            "Warning_Leverage": warning_leverage,
            "In_Domain": leverages <= warning_leverage
        },
        index=X_query_df.index
    )

    return result
