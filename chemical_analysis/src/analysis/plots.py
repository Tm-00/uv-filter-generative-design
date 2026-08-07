from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


METHOD_COLOURS = {
    "RL": "#0072B2",      # blue
    "HC": "#E69F00",      # orange
    "NO-RL": "#009E73"    # green
}


METHOD_ORDER = [
    "NO-RL",
    "HC",
    "RL"
]


def chemical_space_pca(fp_matrix, methods):

    pca = PCA(
        n_components=2
    )

    results = pca.fit_transform(
        fp_matrix
    )

    pca_df = pd.DataFrame(
        {
            "PC1": results[:,0],
            "PC2": results[:,1],
            "Method": methods
        }
    )

    variance = pca.explained_variance_ratio_

    return pca_df, variance



def plot_pca(
    pca_df,
    variance=None,
    save_path=None
):

    plt.figure(figsize=(7,5))

    for method in METHOD_ORDER:

        group = pca_df[
            pca_df["Method"] == method
        ]

        plt.scatter(
            group["PC1"],
            group["PC2"],
            s=10,
            alpha=0.45,
            edgecolors="none",
            label=method,
            color=METHOD_COLOURS[method]
        )

    if variance is not None:
        plt.xlabel(f"PC1 ({variance[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({variance[1]*100:.1f}%)")
    else:
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    plt.title("Chemical Space Distribution")

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


def plot_pareto_front(
    all_molecules,
    pareto_front,
    x="UV_Filter_Score_norm",
    y="TPSA_score_norm",
    save_path=None
):

    required_columns = [x, y]
    
    plt.figure(
        figsize=(7, 5)
    )
    
    for name, df in [
        ("all_molecules", all_molecules),
        ("pareto_front", pareto_front)
    ]:
    
        missing = [
            c for c in required_columns
            if c not in df.columns
        ]
    
        if missing:
            raise ValueError(
                f"{name} is missing columns: {missing}"
            )
    
    # Plot all molecules in background
    for method in METHOD_ORDER:

        background = all_molecules[
            all_molecules["Method"] == method
        ]

        plt.scatter(
            background[x],
            background[y],
            s=8,
            alpha=0.10,
            color=METHOD_COLOURS[method],
            label=f"{method} molecules"
        )


    # Overlay Pareto-optimal molecules
    for method in METHOD_ORDER:

        group = pareto_front[
            pareto_front["Method"] == method
        ]

        plt.scatter(
            group[x],
            group[y],
            s=70,
            alpha=0.90,
            label=f"{method} Pareto",
            color=METHOD_COLOURS[method],
            edgecolor="black",
            linewidth=0.5
        )


    plt.xlabel(
        x.replace("_", " ")
    )

    plt.ylabel(
        y.replace("_", " ")
    )


    plt.title(
        "Pareto Front of Molecular Optimisation\n"
        f"{x.replace('_', ' ')} vs {y.replace('_', ' ')}"
    )


    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )


    plt.grid(
        alpha=0.25
    )


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()

    plt.close()



def plot_property_distribution(
    df,
    property_name,
    save_path=None
):

    plt.figure(figsize=(7,5))


    for method in METHOD_ORDER:

        group = df[
            df["Method"] == method
        ]

        values = group[property_name].dropna()


        kde = gaussian_kde(
            values
        )


        x_range = np.linspace(
            values.min(),
            values.max(),
            500
        )


        plt.plot(
            x_range,
            kde(x_range),
            linewidth=2,
            label=method,
            color=METHOD_COLOURS[method]
        )


    plt.xlabel(property_name)
    plt.ylabel(
        "Density"
    )

    plt.title(
        f"{property_name} Distribution"
    )

    plt.legend()


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()



def plot_thresholds(
    threshold_df,
    save_path=None
):

    pct_df = threshold_df.filter(
        like="%"
    ).T


    colors = [
        METHOD_COLOURS[method]
        for method in pct_df.columns
    ]

    ax = pct_df.plot(
        kind="bar",
        figsize=(8, 5),
        color=colors
    )

    ax.set_ylabel(
        "% Molecules"
    )

    ax.set_title(
        "Fraction of Molecules Passing UV Thresholds"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.2f%%",
            fontsize=7,
            padding=2
        )

    ax.legend(
        title="Method",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()


    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()

def plot_reference_similarity(
    df,
    save_path=None
):

    plt.figure(figsize=(7,5))


    for method in METHOD_ORDER:

        group = df[
            df["Method"] == method
        ]

        values = (
            group["Reference_Similarity"]
            .dropna()
            .values
        )


        kde = gaussian_kde(values)


        x = np.linspace(
            values.min(),
            values.max(),
            500
        )


        plt.plot(
            x,
            kde(x),
            linewidth=2,
            label=method,
            color=METHOD_COLOURS[method]
        )


    plt.xlabel(
        "Tanimoto Similarity to Octocrylene"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Structural Similarity Distribution to Octocrylene"
    )


    plt.legend(
        bbox_to_anchor=(1.02,1),
        loc="upper left",
        frameon=False
    )


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


def plot_elite_distribution(
    df,
    property_name,
    percentile=1,
    save_path=None
):

    plt.figure(figsize=(7,5))


    elite = (
        df.groupby("Method", group_keys=False)
        .apply(
            lambda x: x.nlargest(
                max(1, int(len(x) * percentile / 100)),
                "UV_Filter_Score"
            )
        )
    )


    for method in METHOD_ORDER:

        values = (
            elite[
                elite["Method"] == method
            ][property_name]
            .dropna()
        )

        kde = gaussian_kde(values)

        x = np.linspace(
            values.min(),
            values.max(),
            500
        )

        plt.plot(
            x,
            kde(x),
            linewidth=2,
            label=method,
            color=METHOD_COLOURS[method]
        )


    plt.xlabel(property_name)
    plt.ylabel("Density")

    plt.title(
        f"Top {percentile}% Molecules: {property_name}"
    )

    plt.legend()


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()
    

def plot_generation_progress(
    df,
    property_name,
    window=1000,
    save_path=None
):

    plt.figure(figsize=(7,5))


    for method in METHOD_ORDER:

        group = df[
            df["Method"] == method
        ].copy()


        group = group.reset_index(
            drop=True
        )


        means = []

        positions = []


        for i in range(
            0,
            len(group),
            window
        ):

            batch = group.iloc[
                i:i+window
            ]


            means.append(
                batch[property_name].mean()
            )

            positions.append(
                i
            )


        plt.plot(
            positions,
            means,
            marker="o",
            linewidth=2,
            label=method,
            color=METHOD_COLOURS[method]
        )


    plt.xlabel(
        "Generation order"
    )

    plt.ylabel(
        f"Mean {property_name}"
    )

    plt.title(
        f"{property_name} During Generation"
    )


    plt.legend(
        bbox_to_anchor=(1.02,1),
        loc="upper left",
        frameon=False
    )


    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


# ------------------------------------------------------------
# Optimisation / Pareto plots
# ------------------------------------------------------------


def plot_objective_radar(
    df,
    objectives=None,
    save_path=None
):
    """
    Radar (spider) plot comparing the mean normalised objective
    values achieved by each method. Makes multi-objective
    trade-offs visually comparable in a single figure.
    """

    if objectives is None:

        objectives = [
            "UV_Filter_Score_norm",
            "SA_inverse_norm",
            "TPSA_score_norm",
            "LogP_score_norm"
        ]

    means = (
        df.groupby("Method")[objectives]
        .mean()
    )

    n_vars = len(objectives)

    angles = np.linspace(
        0,
        2 * np.pi,
        n_vars,
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw=dict(polar=True)
    )

    for method in METHOD_ORDER:

        if method not in means.index:
            continue

        values = means.loc[method, objectives].tolist()

        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            label=method,
            color=METHOD_COLOURS[method]
        )

        ax.fill(
            angles,
            values,
            alpha=0.10,
            color=METHOD_COLOURS[method]
        )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        [o.replace("_", " ") for o in objectives]
    )

    ax.set_title(
        "Mean Objective Profile by Method"
    )

    ax.legend(
        bbox_to_anchor=(1.15, 1.05),
        frameon=False
    )


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


def plot_objective_correlation_heatmap(
    df,
    objectives=None,
    save_path=None
):
    """
    Heatmap of pairwise correlations between objectives, useful
    for spotting conflicting or redundant objectives.
    """

    if objectives is None:

        objectives = [
            "UV_Filter_Score",
            "SA_inverse",
            "TPSA_score",
            "LogP_score"
        ]

    corr = df[objectives].corr()

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        corr.values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(len(objectives))
    )

    ax.set_yticks(
        range(len(objectives))
    )

    ax.set_xticklabels(
        [o.replace("_", " ") for o in objectives],
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        [o.replace("_", " ") for o in objectives]
    )

    for i in range(len(objectives)):

        for j in range(len(objectives)):

            ax.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9
            )

    ax.set_title(
        "Objective Correlation Matrix"
    )

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.8,
        label="Pearson r"
    )

    plt.tight_layout()


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


def plot_pareto_chemical_space(
    pca_df,
    pareto_index,
    save_path=None
):
    """
    Overlay Pareto-optimal molecules on the PCA chemical space
    plot, so the reader can see where trade-off-optimal
    molecules sit relative to the full generated population.

    Parameters
    ----------
    pca_df : pandas.DataFrame
        Output of chemical_space_pca (columns PC1, PC2, Method),
        indexed the same as the molecules DataFrame it was built
        from.

    pareto_index : pandas.Index
        Index values (from the molecules DataFrame) identifying
        which rows of pca_df are Pareto-optimal, e.g.
        pareto_front.index.
    """

    plt.figure(
        figsize=(7, 5)
    )

    for method in METHOD_ORDER:

        background = pca_df[
            pca_df["Method"] == method
        ]

        plt.scatter(
            background["PC1"],
            background["PC2"],
            s=8,
            alpha=0.10,
            color=METHOD_COLOURS[method],
            label=f"{method} molecules"
        )

    pareto_points = pca_df.loc[
        pca_df.index.isin(pareto_index)
    ]

    for method in METHOD_ORDER:

        group = pareto_points[
            pareto_points["Method"] == method
        ]

        plt.scatter(
            group["PC1"],
            group["PC2"],
            s=70,
            alpha=0.9,
            color=METHOD_COLOURS[method],
            edgecolor="black",
            linewidth=0.5,
            label=f"{method} Pareto"
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        "Pareto-Optimal Molecules in Chemical Space"
    )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


# ------------------------------------------------------------
# Modelling / QSAR plots
# ------------------------------------------------------------


def plot_feature_importance(
    importance_df,
    title="Feature Importance",
    save_path=None
):
    """
    Horizontal bar chart of descriptor feature importances,
    on a log x-axis since importances here span several orders
    of magnitude (e.g. Lmax/OS/LogP vs. near-zero contributions
    from TPSA/SA_Score/MolWt).

    Parameters
    ----------
    importance_df : pandas.DataFrame
        Output of modelling.get_feature_importance
        (columns: Feature, Importance).
    """

    ordered = importance_df.sort_values(
        "Importance"
    )


    floor = 1e-6

    plot_values = ordered["Importance"].clip(
        lower=floor
    )

    plt.figure(
        figsize=(7, max(3, 0.4 * len(ordered)))
    )

    plt.barh(
        ordered["Feature"],
        plot_values,
        color="#0072B2"
    )

    plt.xscale("log")


    plt.xlim(
        left=plot_values.min() / 3
    )

    plt.xlabel(
        "Importance (log scale)"
    )

    plt.title(
        title
    )

    plt.tight_layout()


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


def plot_predicted_vs_actual(
    y_true,
    y_pred,
    title="Predicted vs Actual",
    save_path=None
):
    """
    Scatter of predicted vs actual values with a 1:1 reference
    line, the standard QSAR model diagnostic plot.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_true,
        y_pred,
        s=15,
        alpha=0.5,
        color="#0072B2",
        edgecolors="none"
    )

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]

    plt.plot(
        lims,
        lims,
        linestyle="--",
        color="black",
        linewidth=1,
        label="1:1"
    )

    plt.xlabel(
        "Actual"
    )

    plt.ylabel(
        "Predicted"
    )

    plt.title(
        title
    )

    plt.legend(
        frameon=False
    )

    plt.gca().set_aspect(
        "equal",
        adjustable="box"
    )


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


def plot_residuals(
    y_true,
    y_pred,
    title="Residuals",
    save_path=None
):
    """
    Residual plot (predicted vs residual), used to check for
    systematic bias or heteroscedasticity in model errors.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        y_pred,
        residuals,
        s=15,
        alpha=0.5,
        color="#0072B2",
        edgecolors="none"
    )

    plt.axhline(
        0,
        linestyle="--",
        color="black",
        linewidth=1
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Residual (Actual - Predicted)"
    )

    plt.title(
        title
    )


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


def plot_y_scrambling(
    scrambling_result,
    title="Y-Scrambling Test",
    save_path=None
):
    """
    Histogram of scrambled-target Q2 values compared to the true
    model's cross-validated Q2, showing whether the real model's
    performance is distinguishable from chance correlation.

    Parameters
    ----------
    scrambling_result : dict
        Output of validation.y_scrambling.
    """

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        scrambling_result["scrambled_r2"],
        bins=20,
        color="#999999",
        alpha=0.8,
        label="Scrambled Q\u00b2"
    )

    plt.axvline(
        scrambling_result["true_r2"],
        color="#D55E00",
        linewidth=2,
        label=f"True Q\u00b2 = {scrambling_result['true_r2']:.2f}"
    )

    plt.xlabel(
        "Cross-validated Q\u00b2"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        title
    )

    plt.legend(
        frameon=False
    )


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()
    plt.close()


# ------------------------------------------------------------
# Top molecule structure grid
# ------------------------------------------------------------


def plot_top_molecules_grid(
    df,
    n_per_method=5,
    rank_by="UV_Filter_Score",
    novelty_column=None,
    mols_per_row=5,
    sub_img_size=(280, 300),
    save_path=None
):
    """
    Render 2D structures of the top-N molecules per method, labelled with
    their key scores. Intended as the "here's what it actually found"
    closing figure - a concrete complement to the population-level
    statistics elsewhere in the pipeline.

    Legends are kept to a single line (Method | score[ | novelty]),
    since RDKit's MolsToGridImage legend area is sized for one line of
    text and does not reliably render multi-line legends - passing a
    multi-line string there causes clipped/overlapping text rather than
    a wrapped label. Full per-molecule descriptor values belong in the
    accompanying results table, not repeated here.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain "Method", "Canonical_SMILES", and the column named
        in `rank_by`.

    n_per_method : int
        How many top molecules to draw per method.

    rank_by : str
        Column used to select the top molecules within each method
        (highest values first), and shown in the legend.

    novelty_column : str, optional
        Column name containing a boolean/label indicating whether a
        molecule was found in an external database (e.g. from a PubChem
        lookup). If provided, "Known" / "Novel*" is appended to each
        label.

    mols_per_row : int
        Grid width.

    sub_img_size : (int, int)
        Pixel size of each structure panel. Slightly taller than the
        previous default to keep the single-line legend comfortably
        legible.

    save_path : str or Path, optional
        Where to save the grid image (PNG).

    Returns
    -------
    PIL.Image (also written to save_path if given)
    """

    from rdkit import Chem
    from rdkit.Chem import Draw

    mols = []
    legends = []

    for method in METHOD_ORDER:

        if method not in df["Method"].unique():
            continue

        top = df[
            df["Method"] == method
        ].nlargest(
            n_per_method,
            rank_by
        )

        for _, row in top.iterrows():

            mol = Chem.MolFromSmiles(
                row["Canonical_SMILES"]
            )

            if mol is None:
                continue

            mols.append(mol)

            legend = f"{method} | {rank_by}={row[rank_by]:.2f}"

            if novelty_column and novelty_column in df.columns:

                status = (
                    "Novel*"
                    if row[novelty_column]
                    else "Known"
                )

                legend += f" | {status}"

            legends.append(
                legend
            )

    grid = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        returnPNG=False
    )

    if save_path:

        grid.save(
            save_path
        )

    return grid


# ------------------------------------------------------------
# Pairwise reward-component Pareto plots
# ------------------------------------------------------------


def plot_reward_component_pareto_pair(
    df,
    x_col,
    y_col,
    x_label=None,
    y_label=None,
    save_path=None
):
    """
    2D Pareto front between two reward-component scores (e.g. the
    transformed Lmax/OS/LogP scores from reward_components.py, which
    are genuinely monotonic maximise-is-better objectives - unlike
    the raw descriptor values).

    Same visual convention as plot_pareto_front: faded background
    population per method, solid Pareto-optimal points on top.
    """

    from src.analysis.pareto import identify_pareto

    mask = identify_pareto(
        df,
        [x_col, y_col]
    )

    pareto_front = df.loc[mask]

    plt.figure(
        figsize=(7, 5)
    )

    for method in METHOD_ORDER:

        background = df[
            df["Method"] == method
        ]

        plt.scatter(
            background[x_col],
            background[y_col],
            s=8,
            alpha=0.10,
            color=METHOD_COLOURS[method],
            label=f"{method} molecules"
        )

    for method in METHOD_ORDER:

        group = pareto_front[
            pareto_front["Method"] == method
        ]

        plt.scatter(
            group[x_col],
            group[y_col],
            s=70,
            alpha=0.90,
            label=f"{method} Pareto",
            color=METHOD_COLOURS[method],
            edgecolor="black",
            linewidth=0.5
        )

    plt.xlabel(
        x_label or x_col.replace("_", " ")
    )

    plt.ylabel(
        y_label or y_col.replace("_", " ")
    )

    plt.title(
        "Reward-Component Pareto Front\n"
        f"{x_label or x_col} vs {y_label or y_col}"
    )

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.grid(
        alpha=0.25
    )

    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()


def plot_all_reward_component_pareto_pairs(
    df,
    figure_dir,
    columns=None,
    labels=None
):
    """
    Generate all pairwise Pareto plots between the reward-component
    scores (3 columns -> 3 pairs). Requires
    reward_components.add_reward_component_scores(df) to have been
    run first.

    Parameters
    ----------
    df : pandas.DataFrame
    figure_dir : Path
        Directory to save each pair's PNG into.
    columns : list of str, optional
        Defaults to reward_components.get_reward_component_columns().
    labels : dict, optional
        Maps column name -> display label (e.g. "Lmax_reward_score":
        "Lmax score"). Defaults to a cleaned-up column name.
    """

    from pathlib import Path
    from itertools import combinations
    from src.analysis.reward_components import get_reward_component_columns

    if columns is None:

        columns = get_reward_component_columns()

    if labels is None:

        labels = {}

    figure_dir = Path(
        figure_dir
    )

    for x_col, y_col in combinations(
        columns,
        2
    ):

        x_label = labels.get(
            x_col,
            x_col.replace("_reward_score", "")
        )

        y_label = labels.get(
            y_col,
            y_col.replace("_reward_score", "")
        )

        filename = (
            f"reward_pareto_{x_label}_vs_{y_label}.png"
            .replace(" ", "_")
        )

        plot_reward_component_pareto_pair(
            df,
            x_col,
            y_col,
            x_label=x_label,
            y_label=y_label,
            save_path=figure_dir / filename
        )