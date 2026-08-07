import numpy as np
from sklearn.preprocessing import MinMaxScaler

def gaussian_score(
    value,
    target,
    sigma
):
    """
    Converts a molecular descriptor into a score
    based on distance from an ideal target.

    Higher score = closer to target.
    """

    return np.exp(
        -((value - target) ** 2) /
        (2 * sigma ** 2)
    )


def add_objectives(df):
    """
    Add optimisation objectives used for Pareto analysis.

    Objectives:
        - UV_Filter_Score:
            maximise

        - SA_inverse:
            minimise synthetic accessibility score

        - TPSA_score:
            target moderate polarity

        - LogP_score:
            target suitable lipophilicity

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing objective scores
    """

    df = df.copy()


    # Synthetic accessibility
    # lower SA score is preferable
    df["SA_inverse"] = (
        -df["SA_Score"]
    )


    # LogP objective
    #
    # Target chosen to encourage:
    # - skin retention
    # - UV filter compatibility
    # - avoid excessive hydrophobicity
    #
    df["LogP_score"] = gaussian_score(
        df["LogP"],
        target=3.0,
        sigma=1.5
    )


    # TPSA objective
    #
    # Moderate polarity:
    # - enough functionality
    # - reduced excessive penetration
    #
    df["TPSA_score"] = gaussian_score(
        df["TPSA"],
        target=45.0,
        sigma=15.0
    )


    return df



def get_objective_columns():

    return [
        "UV_Filter_Score",
        "SA_inverse",
        "TPSA_score",
        "LogP_score"
    ]
    

def get_normalised_objective_columns():

    return [
        "UV_Filter_Score_norm",
        "SA_inverse_norm",
        "TPSA_score_norm",
        "LogP_score_norm"
    ]
    

def normalise_objectives(df):

    df = df.copy()

    cols = get_objective_columns()

    scaler = MinMaxScaler()

    df[
        [f"{c}_norm" for c in cols]
    ] = scaler.fit_transform(
        df[cols]
    )

    return df