import pandas as pd

from src.analysis.pareto import pareto_efficiency_summary


def top_1_percent_summary(
    df,
    similarity_df=None
):

    results = []

    for method, group in df.groupby("Method"):

        top_n = max(
            1,
            int(len(group) * 0.01)
        )

        top = group.nlargest(
            top_n,
            "UV_Filter_Score"
        )

        result = {
            "Method": method,
            "Top_1_percent_molecules": len(top),
            "Mean_UV_Filter_Score": top["UV_Filter_Score"].mean(),
            "Unique_Scaffolds": (
                top["Scaffold"]
                .nunique()
                if "Scaffold" in top.columns
                else None
            )
        }


        if similarity_df is not None:

            similarity_top = similarity_df[
                similarity_df.index.isin(top.index)
            ]

            result[
                "Mean_Octocrylene_Similarity"
            ] = (
                similarity_top[
                    "Reference_Similarity"
                ]
                .mean()
            )


        results.append(result)


    return pd.DataFrame(results)

def method_objective_summary(df):

    objectives = [
        "Lmax",
        "OS",
        "UV_Filter_Score",
        "LogP",
        "SA_Score"
    ]


    summary = (
        df
        .groupby("Method")[objectives]
        .mean()
        .reset_index()
    )


    return summary

def pareto_method_summary(
    pareto_front,
    all_molecules
):

    pareto_counts = (
        pareto_front
        .groupby("Method")
        .size()
        .rename(
            "Pareto_Molecules"
        )
    )


    total_counts = (
        all_molecules
        .groupby("Method")
        .size()
        .rename(
            "Total_Molecules"
        )
    )


    summary = pd.concat(
        [
            pareto_counts,
            total_counts
        ],
        axis=1
    )


    summary["Pareto_Percentage"] = (
        summary["Pareto_Molecules"]
        /
        summary["Total_Molecules"]
        *
        100
    )


    return summary.reset_index()


def hypervolume_summary(
    all_molecules,
    pareto_front,
    objectives=None,
    n_samples=100_000,
    random_state=42
):
    """
    Combine Pareto molecule counts/percentages with hypervolume
    and spacing metrics, giving a single table for comparing how
    well each optimisation method trades off the objectives.

    Parameters
    ----------
    all_molecules : pandas.DataFrame
    pareto_front : pandas.DataFrame
        Output of pareto.calculate_pareto(all_molecules), used
        only for the combined Pareto molecule counts.

    Returns
    -------
    pandas.DataFrame with columns:
        Method, Pareto_Molecules, Total_Molecules,
        Pareto_Percentage, Own_Pareto_Molecules, Hypervolume,
        Spacing
    """

    counts = pareto_method_summary(
        pareto_front,
        all_molecules
    )

    efficiency = pareto_efficiency_summary(
        all_molecules,
        objectives=objectives,
        n_samples=n_samples,
        random_state=random_state
    )

    return counts.merge(
        efficiency,
        on="Method",
        how="outer"
    )


def model_performance_summary(cv_results):
    """
    Summarise cross-validated model performance across one or
    more prediction targets.

    Parameters
    ----------
    cv_results : dict
        Maps target name (e.g. "UV_Filter_Score", "SPF") to the
        dict returned by validation.cross_validate_model or
        validation.repeated_cross_validate_model (must contain
        "mean" and "std" keys).

    Returns
    -------
    pandas.DataFrame with columns:
        Target, Mean_R2, Std_R2, Mean_RMSE, Std_RMSE,
        Mean_MAE, Std_MAE
    """

    rows = []

    for target, result in cv_results.items():

        mean = result["mean"]
        std = result["std"]

        rows.append(
            {
                "Target": target,
                "Mean_R2": mean["R2"],
                "Std_R2": std["R2"],
                "Mean_RMSE": mean["RMSE"],
                "Std_RMSE": std["RMSE"],
                "Mean_MAE": mean["MAE"],
                "Std_MAE": std["MAE"]
            }
        )

    return pd.DataFrame(
        rows
    )


def feature_importance_summary(importance_by_target):
    """
    Combine per-target feature importance rankings into a single
    wide table for easy comparison of which descriptors matter
    across different targets/models.

    Parameters
    ----------
    importance_by_target : dict
        Maps target name to the DataFrame returned by
        modelling.get_feature_importance (columns: Feature,
        Importance).

    Returns
    -------
    pandas.DataFrame indexed by Feature, one column of
    importances per target, sorted by mean importance
    (descending).
    """

    series = {}

    for target, importance_df in importance_by_target.items():

        series[target] = importance_df.set_index(
            "Feature"
        )["Importance"]

    wide = pd.DataFrame(
        series
    )

    wide["Mean_Importance"] = wide.mean(
        axis=1
    )

    wide = wide.sort_values(
        "Mean_Importance",
        ascending=False
    )

    return wide.reset_index()