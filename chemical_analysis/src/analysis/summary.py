import pandas as pd


def summarise_methods(df):

    summary = (
        df
        .groupby("Method")
        .agg(
            Molecules=("Canonical_SMILES", "count"),
            Mean_UV=("UV_Filter_Score", "mean"),
            Max_UV=("UV_Filter_Score", "max"),
            Mean_MolWt=("MolWt", "mean"),
            Mean_LogP=("LogP", "mean"),
            Mean_TPSA=("TPSA", "mean"),
            Mean_SA=("SA_Score", "mean"),
            Mean_Lmax=("Lmax", "mean"),
            Mean_OS=("OS", "mean")
        )
    )

    return summary


def score_threshold_summary(df, thresholds=[0.5, 0.7, 0.8, 0.9]):

    results = []

    for method, group in df.groupby("Method"):

        row = {
            "Method": method,
            "Total": len(group)
        }

        for threshold in thresholds:
            count = (
                group["UV_Filter_Score"] >= threshold
            ).sum()

            row[f"UV >= {threshold}"] = count
            row[f"% >= {threshold}"] = (
                count / len(group) * 100
            )

        results.append(row)

    return pd.DataFrame(results).set_index("Method")