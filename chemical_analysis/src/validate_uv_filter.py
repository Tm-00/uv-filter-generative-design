import pandas as pd

from acegen_loader import test_uv_filter

from config import (
    ECHA_REFERENCE_PATH,
    TABLE_DIR,
    FIGURE_DIR
)

from src.validation import regression_metrics
from src.analysis.plots import plot_predicted_vs_actual


def load_echa_reference(path=ECHA_REFERENCE_PATH):
    """
    Load the ECHA reference UV filter dataset: real,
    regulator-approved UV filters with experimentally measured
    Lmax and OS.

    Expected columns: Ref, Name, CAS, SMILES, Lmax, OS, Source
    """

    df = pd.read_csv(
        path,
        encoding="utf-8-sig"
    )

    df["Name"] = df["Name"].str.strip()

    df["CAS"] = df["CAS"].str.strip()

    df["SMILES"] = df["SMILES"].str.strip()

    return df


def predict_uv_properties(df, uv_filter):
    """
    Predict Lmax and OS for each compound using the same scorer
    that scores every ACEGEN-generated molecule in
    descriptors.calculate_descriptors, so the comparison reflects
    exactly what drives the RL reward signal.
    """

    df = df.copy()

    smiles = df["SMILES"].tolist()

    df["Lmax_predicted"] = (
        uv_filter
        ._scorer
        .lmax_predictor
        .predict_batch(smiles)
    )

    df["OS_predicted"] = (
        uv_filter
        ._scorer
        .os_predictor
        .predict_batch(smiles)
    )

    return df


def evaluate(df):
    """
    Compare predicted vs experimental Lmax and OS.

    Some compounds have multiple experimental measurements from
    different sources (e.g. Homosalate appears twice with
    different Lmax/OS) - each row is evaluated separately, since
    the disagreement between sources is itself informative about
    measurement variability, not something to average away here.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of predict_uv_properties. Must contain Lmax,
        Lmax_predicted, OS, OS_predicted.

    Returns
    -------
    dict with:
        lmax_metrics : dict (R2, RMSE, MAE), all rows
        os_metrics : dict (R2, RMSE, MAE), rows with a
            non-null experimental OS only
        per_compound : DataFrame with prediction errors,
            sorted by absolute Lmax error (largest first)
    """

    lmax_metrics = regression_metrics(
        df["Lmax"],
        df["Lmax_predicted"]
    )

    os_subset = df.dropna(
        subset=["OS"]
    )

    os_metrics = regression_metrics(
        os_subset["OS"],
        os_subset["OS_predicted"]
    )

    per_compound = df.copy()

    per_compound["Lmax_error"] = (
        per_compound["Lmax_predicted"]
        -
        per_compound["Lmax"]
    )

    per_compound["OS_error"] = (
        per_compound["OS_predicted"]
        -
        per_compound["OS"]
    )

    per_compound = per_compound[
        [
            "Name",
            "CAS",
            "SMILES",
            "Lmax",
            "Lmax_predicted",
            "Lmax_error",
            "OS",
            "OS_predicted",
            "OS_error",
            "Source"
        ]
    ].sort_values(
        "Lmax_error",
        key=abs,
        ascending=False
    ).reset_index(
        drop=True
    )

    return {
        "lmax_metrics": lmax_metrics,
        "os_metrics": os_metrics,
        "per_compound": per_compound
    }


def run_validation():
    """
    Run the full external validation: load the uv_filter scorer,
    load the ECHA reference compounds, predict Lmax/OS for each,
    compare against the experimental values, save a per-compound
    error table and two predicted-vs-actual plots.
    """

    uv_filter = test_uv_filter()

    reference = load_echa_reference()

    predicted = predict_uv_properties(
        reference,
        uv_filter
    )

    results = evaluate(
        predicted
    )

    print(
        "Lmax validation metrics (n="
        f"{len(predicted)}):",
        results["lmax_metrics"]
    )

    print(
        "OS validation metrics (n="
        f"{predicted['OS'].notna().sum()}):",
        results["os_metrics"]
    )

    results["per_compound"].to_csv(
        TABLE_DIR / "echa_uv_filter_validation.csv",
        index=False
    )

    plot_predicted_vs_actual(
        predicted["Lmax"],
        predicted["Lmax_predicted"],
        title="Lmax: Predicted vs Experimental (ECHA UV Filters)",
        save_path=FIGURE_DIR / "echa_lmax_validation.png"
    )

    os_subset = predicted.dropna(
        subset=["OS"]
    )

    plot_predicted_vs_actual(
        os_subset["OS"],
        os_subset["OS_predicted"],
        title="OS: Predicted vs Experimental (ECHA UV Filters)",
        save_path=FIGURE_DIR / "echa_os_validation.png"
    )

    return results


if __name__ == "__main__":
    run_validation()
