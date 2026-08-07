import numpy as np
import pandas as pd

from src.analysis.objectives import (
    add_objectives,
    normalise_objectives,
    get_normalised_objective_columns,
    get_objective_columns
)


def identify_pareto(
    df,
    objectives
):

    values = df[objectives].values

    is_pareto = np.ones(
        len(values),
        dtype=bool
    )


    for i, candidate in enumerate(values):

        if not is_pareto[i]:
            continue


        dominates_candidate = (
            np.all(
                values >= candidate,
                axis=1
            )
            &
            np.any(
                values > candidate,
                axis=1
            )
        )


        if dominates_candidate.any():

            is_pareto[i] = False

            continue


        dominated_by_candidate = (
            np.all(
                candidate >= values,
                axis=1
            )
            &
            np.any(
                candidate > values,
                axis=1
            )
        )


        is_pareto[dominated_by_candidate] = False


    return is_pareto



def calculate_pareto(df):

    objectives = get_objective_columns()

    mask = identify_pareto(
        df,
        objectives
    )

    return df.loc[mask].copy()



# ------------------------------------------------------------
# Hypervolume and Pareto efficiency metrics
# ------------------------------------------------------------
#
# All objectives here are assumed to be normalised to a
# maximise-is-better [0, 1] scale (see
# analysis.objectives.normalise_objectives), so a single shared
# reference point of all-zeros and a shared ideal point of
# all-ones can be used to make hypervolumes comparable across
# methods.


def calculate_hypervolume(
    points,
    reference_point,
    ideal_point=None,
    n_samples=100_000,
    random_state=42
):
    """
    Estimate the hypervolume dominated by a set of (maximised)
    Pareto points, using Monte Carlo integration.

    This avoids adding an exact hypervolume dependency (e.g.
    pymoo) and scales to any number of objectives.

    Parameters
    ----------
    points : array-like, shape (n_points, n_objectives)
        Points to measure (typically a Pareto front). Higher is
        better in every dimension.

    reference_point : array-like, shape (n_objectives,)
        Lower bound / worst-case point. Hypervolume is measured
        for the region above this point.

    ideal_point : array-like, shape (n_objectives,), optional
        Upper bound used for the sampling box. Defaults to the
        max of `points` in each dimension. Pass a fixed ideal
        point (e.g. np.ones(n_objectives) for normalised
        objectives) when comparing hypervolumes across several
        point sets, so they share the same bounding box.

    n_samples : int
        Number of Monte Carlo samples.

    random_state : int

    Returns
    -------
    float
        Estimated hypervolume.
    """

    points = np.asarray(
        points,
        dtype=float
    )

    reference_point = np.asarray(
        reference_point,
        dtype=float
    )

    if points.size == 0:
        return 0.0

    if ideal_point is None:

        ideal_point = points.max(
            axis=0
        )

    else:

        ideal_point = np.asarray(
            ideal_point,
            dtype=float
        )

    box_dims = ideal_point - reference_point

    if np.any(box_dims <= 0):
        return 0.0

    box_volume = np.prod(
        box_dims
    )

    rng = np.random.RandomState(
        random_state
    )

    samples = rng.uniform(
        low=reference_point,
        high=ideal_point,
        size=(n_samples, len(reference_point))
    )

    # A sample is dominated (i.e. inside the hypervolume) if at
    # least one Pareto point is >= the sample in every objective.
    dominated = np.zeros(
        n_samples,
        dtype=bool
    )

    chunk_size = 2000

    for start in range(0, len(points), chunk_size):

        chunk = points[
            start:start + chunk_size
        ]

        # shape: (chunk_size, n_samples)
        covers = np.all(
            chunk[:, None, :] >= samples[None, :, :],
            axis=2
        )

        dominated |= covers.any(
            axis=0
        )

    fraction_dominated = dominated.mean()

    return float(
        fraction_dominated * box_volume
    )


def spacing_metric(points):
    """
    Spacing indicator: measures how evenly a Pareto front's
    solutions are distributed. Lower values indicate a more
    uniform spread (no large gaps or tight clusters).

    Defined as the standard deviation of nearest-neighbour
    (Euclidean) distances between points on the front.
    """

    points = np.asarray(
        points,
        dtype=float
    )

    n = len(points)

    if n < 2:
        return 0.0

    nearest_distances = []

    for i in range(n):

        diffs = points - points[i]

        distances = np.linalg.norm(
            diffs,
            axis=1
        )

        distances[i] = np.inf

        nearest_distances.append(
            distances.min()
        )

    return float(
        np.std(nearest_distances)
    )


def pareto_efficiency_summary(
    df,
    objectives=None,
    reference_point=None,
    ideal_point=None,
    n_samples=100_000,
    random_state=42
):
    """
    Compare optimisation methods on how good a trade-off they
    achieve between objectives, rather than just how many
    Pareto-optimal molecules they produce.

    For each method, identifies that method's own non-dominated
    front (among its own molecules only) and computes:
        - Hypervolume dominated by that front
        - Spacing (evenness of the front)
        - Number of molecules on that method's own front

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a "Method" column and the objective columns.

    objectives : list of str, optional
        Defaults to get_normalised_objective_columns(), since
        hypervolume comparisons require objectives on a shared
        scale.

    reference_point : array-like, optional
        Defaults to all-zeros (the minimum of a normalised
        [0, 1] objective).

    ideal_point : array-like, optional
        Defaults to all-ones, so hypervolumes are directly
        comparable between methods.

    Returns
    -------
    pandas.DataFrame with columns:
        Method, Own_Pareto_Molecules, Hypervolume, Spacing
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

    results = []

    for method, group in df.groupby("Method"):

        mask = identify_pareto(
            group,
            objectives
        )

        own_front = group.loc[mask, objectives].values

        hypervolume = calculate_hypervolume(
            own_front,
            reference_point=reference_point,
            ideal_point=ideal_point,
            n_samples=n_samples,
            random_state=random_state
        )

        spacing = spacing_metric(
            own_front
        )

        results.append(
            {
                "Method": method,
                "Own_Pareto_Molecules": len(own_front),
                "Hypervolume": hypervolume,
                "Spacing": spacing
            }
        )

    return pd.DataFrame(
        results
    )