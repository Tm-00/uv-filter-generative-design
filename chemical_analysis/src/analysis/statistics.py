import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.ensemble import RandomForestRegressor

from src.analysis.pareto import (
    identify_pareto,
    calculate_hypervolume
)
from src.analysis.objectives import get_normalised_objective_columns
from src.modelling import prepare_xy
from src.validation import repeated_cross_validate_model


def kruskal_uv_test(df):

    groups = [
        group["UV_Filter_Score"].values
        for _, group in df.groupby("Method")
    ]

    stat, p = kruskal(*groups)

    return pd.DataFrame({
        "Test": ["Kruskal-Wallis"],
        "Statistic": [stat],
        "p_value": [p]
    })


def pairwise_uv_tests(df):

    methods = df["Method"].unique()

    results = []

    for i in range(len(methods)):

        for j in range(i+1, len(methods)):

            a = methods[i]
            b = methods[j]

            x = df[
                df["Method"] == a
            ]["UV_Filter_Score"]

            y = df[
                df["Method"] == b
            ]["UV_Filter_Score"]


            stat, p = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )


            results.append(
                {
                    "Method_A": a,
                    "Method_B": b,
                    "Statistic": stat,
                    "p_value": p
                }
            )

    return pd.DataFrame(results)


# ------------------------------------------------------------
# Generic significance-testing helpers
# ------------------------------------------------------------
#
# Used to compare methods on metrics that don't naturally exist
# as a per-molecule column (e.g. hypervolume, cross-validated
# model R2), given a dict of {group_name: array_of_values}.


def kruskal_test(distributions, label="Groups"):

    stat, p = kruskal(
        *distributions.values()
    )

    return pd.DataFrame(
        {
            "Test": [f"Kruskal-Wallis: {label}"],
            "Statistic": [stat],
            "p_value": [p]
        }
    )


def pairwise_tests(distributions, label="Groups"):

    names = list(
        distributions.keys()
    )

    results = []

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            a = names[i]
            b = names[j]

            stat, p = mannwhitneyu(
                distributions[a],
                distributions[b],
                alternative="two-sided"
            )

            results.append(
                {
                    "Comparison": label,
                    "Group_A": a,
                    "Group_B": b,
                    "Statistic": stat,
                    "p_value": p
                }
            )

    return pd.DataFrame(results)


# ------------------------------------------------------------
# Hypervolume significance
# ------------------------------------------------------------


def hypervolume_significance(
    df,
    objectives=None,
    n_bootstrap=50,
    sample_size=2000,
    reference_point=None,
    ideal_point=None,
    hv_samples=10_000,
    random_state=42
):
    """
    Test whether methods differ significantly in the hypervolume
    (trade-off quality) of their own Pareto fronts, rather than
    just comparing the single hypervolume number in
    results.hypervolume_summary.

    Hypervolume is a single scalar per method, so there is no
    natural distribution to run a significance test against.
    This builds one via bootstrap resampling: for each method,
    `n_bootstrap` resamples (with replacement) are drawn from
    that method's molecules, the resample's own non-dominated
    front and hypervolume are computed, and the resulting spread
    of hypervolume values is compared across methods with
    Kruskal-Wallis and pairwise Mann-Whitney U tests.

    Note on cost: identifying a non-dominated front is O(n^2),
    and this repeats that `n_bootstrap` times per method. For
    large datasets (tens of thousands of molecules per method),
    `sample_size` sub-samples each bootstrap draw to keep runtime
    reasonable; the trade-off is a noisier hypervolume estimate.
    Raise `sample_size` (or set it to None to use the full
    method-group each draw) for a slower, higher-fidelity result.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain "Method" and the objective columns.

    objectives : list of str, optional
        Defaults to get_normalised_objective_columns(), matching
        pareto.pareto_efficiency_summary.

    n_bootstrap : int
        Number of bootstrap resamples per method.

    sample_size : int or None
        Number of molecules drawn (with replacement) per
        bootstrap resample. None uses the full method-group size.

    reference_point, ideal_point : array-like, optional
        Passed to pareto.calculate_hypervolume. Default to
        all-zeros / all-ones for normalised [0, 1] objectives.

    hv_samples : int
        Monte Carlo samples used per hypervolume estimate.

    Returns
    -------
    dict with:
        bootstrap_distributions : DataFrame, long format
            (Method, Hypervolume)
        kruskal : DataFrame
        pairwise : DataFrame
    """

    if objectives is None:

        objectives = get_normalised_objective_columns()

    n_objectives = len(
        objectives
    )

    if reference_point is None:

        reference_point = np.zeros(
            n_objectives
        )

    if ideal_point is None:

        ideal_point = np.ones(
            n_objectives
        )

    rng = np.random.RandomState(
        random_state
    )

    distributions = {}

    for method, group in df.groupby("Method"):

        base = group[objectives]

        n = (
            sample_size
            if sample_size is not None
            else len(base)
        )

        values = []

        for _ in range(n_bootstrap):

            resample = base.sample(
                n=n,
                replace=True,
                random_state=rng.randint(0, 1_000_000)
            )

            mask = identify_pareto(
                resample,
                objectives
            )

            front = resample.values[mask]

            hv = calculate_hypervolume(
                front,
                reference_point=reference_point,
                ideal_point=ideal_point,
                n_samples=hv_samples,
                random_state=rng.randint(0, 1_000_000)
            )

            values.append(
                hv
            )

        distributions[method] = np.array(
            values
        )

    kruskal_df = kruskal_test(
        distributions,
        label="Hypervolume (bootstrap)"
    )

    pairwise_df = pairwise_tests(
        distributions,
        label="Hypervolume (bootstrap)"
    )

    long_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Method": method,
                    "Hypervolume": values
                }
            )
            for method, values in distributions.items()
        ],
        ignore_index=True
    )

    return {
        "bootstrap_distributions": long_df,
        "kruskal": kruskal_df,
        "pairwise": pairwise_df
    }


# ------------------------------------------------------------
# Model performance significance
# ------------------------------------------------------------


def model_performance_significance(
    df,
    target,
    features=None,
    estimator=None,
    k=5,
    n_repeats=10,
    random_state=42
):
    """
    Test whether the descriptor-target relationship is
    significantly stronger for one method than another, by
    comparing per-fold cross-validated R2 distributions rather
    than just the mean R2 in results.model_performance_summary.

    For each method, a model is repeatedly k-fold
    cross-validated (validation.repeated_cross_validate_model)
    on that method's own molecules, giving a distribution of
    per-fold R2 scores. Kruskal-Wallis and pairwise Mann-Whitney
    U tests are then run across these distributions.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain "Method", the feature columns, and target.

    target : str
        Column to predict (e.g. "UV_Filter_Score").

    features : list of str, optional
        Defaults to the standard descriptor set (see
        modelling.get_feature_columns), excluding target.

    estimator : unfitted sklearn-compatible estimator, optional
        Defaults to RandomForestRegressor(n_estimators=300).

    k, n_repeats : int
        Passed to validation.repeated_cross_validate_model.

    Returns
    -------
    dict with:
        fold_r2 : DataFrame, long format
            (Method, Repeat, Fold, R2)
        kruskal : DataFrame
        pairwise : DataFrame
    """

    if estimator is None:

        estimator = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        )

    distributions = {}

    fold_frames = []

    for method, group in df.groupby("Method"):

        X, y, model_features = prepare_xy(
            group,
            target,
            features=features
        )

        cv = repeated_cross_validate_model(
            estimator,
            X,
            y,
            k=k,
            n_repeats=n_repeats,
            random_state=random_state
        )

        fold_df = cv["fold_metrics"].copy()

        fold_df["Method"] = method

        fold_frames.append(
            fold_df
        )

        distributions[method] = fold_df["R2"].values

    fold_r2 = pd.concat(
        fold_frames,
        ignore_index=True
    )[["Method", "Repeat", "Fold", "R2"]]

    kruskal_df = kruskal_test(
        distributions,
        label=f"Model performance: {target}"
    )

    pairwise_df = pairwise_tests(
        distributions,
        label=f"Model performance: {target}"
    )

    return {
        "fold_r2": fold_r2,
        "kruskal": kruskal_df,
        "pairwise": pairwise_df
    }

# ------------------------------------------------------------
# Y-scrambling summary
# ------------------------------------------------------------


def y_scrambling_summary(scrambling_results, n_iterations):
    """
    Package validation.y_scrambling's output into a save-ready
    form: the raw scrambled R2 distribution as a DataFrame, plus
    a summary dict (including an empirical permutation p-value)
    ready for cache.save_metadata.

    Parameters
    ----------
    scrambling_results : dict
        Output of validation.y_scrambling (keys: scrambled_r2,
        mean_scrambled_r2, max_scrambled_r2, true_r2).

    n_iterations : int
        Number of scrambling iterations used, passed separately
        since it isn't stored in scrambling_results itself.

    Returns
    -------
    dict with:
        scrambled_r2_table : DataFrame, one column (scrambled_r2)
        summary : dict with true_r2, mean_scrambled_r2,
            max_scrambled_r2, n_iterations, and
            permutation_p_value - the fraction of scrambled runs
            scoring >= the true model, using the standard
            (1 + count) / (1 + n_iterations) correction so p is
            never reported as exactly zero.
    """

    scrambled_r2 = np.array(
        scrambling_results["scrambled_r2"]
    )

    permutation_p = (
        1
        + np.sum(scrambled_r2 >= scrambling_results["true_r2"])
    ) / (
        1 + n_iterations
    )

    scrambled_r2_table = pd.DataFrame(
        {"scrambled_r2": scrambling_results["scrambled_r2"]}
    )

    summary = {
        "true_r2": scrambling_results["true_r2"],
        "mean_scrambled_r2": scrambling_results["mean_scrambled_r2"],
        "max_scrambled_r2": scrambling_results["max_scrambled_r2"],
        "n_iterations": n_iterations,
        "permutation_p_value": permutation_p
    }

    return {
        "scrambled_r2_table": scrambled_r2_table,
        "summary": summary
    }