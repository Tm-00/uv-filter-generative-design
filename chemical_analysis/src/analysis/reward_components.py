import numpy as np


# These replicate score_lmax / score_os / score_logp from
# acegen-chemical/acegen/scoring_functions/uv_filter.py exactly, so
# that Pareto/hypervolume analysis can be run on genuinely monotonic,
# maximise-is-better objectives - rather than on the raw Lmax/OS/LogP
# values, which don't satisfy that assumption (Lmax in particular is a
# target-window gate, not something to maximise or minimise).


def score_lmax(lmax, lb=290, ub=330):
    """
    Vectorised replica of uv_filter.score_lmax: 1.0 if lmax falls in
    [lb, ub], else 0.0. NaN-safe.
    """

    lmax = np.asarray(
        lmax,
        dtype=float
    )

    in_range = (
        (lmax >= lb)
        &
        (lmax <= ub)
    )

    result = np.where(
        in_range,
        1.0,
        0.0
    )

    return np.where(
        np.isnan(lmax),
        np.nan,
        result
    )


def score_os(os_value, center=0.4, k=10):
    """
    Vectorised replica of uv_filter.score_os: sigmoid centred at
    `center`. NaN propagates naturally.
    """

    os_value = np.asarray(
        os_value,
        dtype=float
    )

    return 1 / (
        1
        +
        np.exp(-k * (os_value - center))
    )


def score_logp(logp, center=2.5, k=1):
    """
    Vectorised replica of uv_filter.score_logp: sigmoid centred at
    `center`. NaN propagates naturally.
    """

    logp = np.asarray(
        logp,
        dtype=float
    )

    return 1 / (
        1
        +
        np.exp(-k * (logp - center))
    )


REWARD_COMPONENT_COLUMNS = [
    "Lmax_reward_score",
    "OS_reward_score",
    "LogP_reward_score"
]


def add_reward_component_scores(df):
    """
    Add the three transformed, monotonic reward-component scores
    (derived directly from the raw Lmax/OS/LogP columns already in df)
    so that Pareto/hypervolume analysis can be run on the scoring
    function's actual mechanics rather than raw descriptor values.

    Requires df to already contain Lmax, OS, LogP (from
    descriptors.calculate_descriptors).
    """

    df = df.copy()

    df["Lmax_reward_score"] = score_lmax(
        df["Lmax"]
    )

    df["OS_reward_score"] = score_os(
        df["OS"]
    )

    df["LogP_reward_score"] = score_logp(
        df["LogP"]
    )

    return df


def get_reward_component_columns():

    return list(
        REWARD_COMPONENT_COLUMNS
    )
