import pandas as pd
from datetime import datetime

from config import (
    DATASETS,
    TABLE_DIR,
    FIGURE_DIR,
    CACHE_DIR,
    PIPELINE_VERSION,
    RANDOM_STATE,
    MODELLING_TARGET,
    N_ESTIMATORS,
    CV_FOLDS,
    CV_REPEATS,
    Y_SCRAMBLE_ITERATIONS,
    APPLICABILITY_DOMAIN_MULTIPLIER,
    HYPERVOLUME_MC_SAMPLES,
    HV_BOOTSTRAP_ITERATIONS,
    HV_BOOTSTRAP_SAMPLE_SIZE,
    HV_BOOTSTRAP_MC_SAMPLES,
    MODEL_SIGNIFICANCE_CV_FOLDS,
    MODEL_SIGNIFICANCE_CV_REPEATS,
    TOP_MOLECULES_PER_METHOD
)

from acegen_loader import test_uv_filter

from src.dataset import build_dataset

from src.analysis.summary import (
    summarise_methods,
    score_threshold_summary
)

from src.analysis.pareto import (
    calculate_pareto,
    pareto_efficiency_summary
)

from src.analysis.reward_components import (
    add_reward_component_scores,
    get_reward_component_columns
)

from src.fingerprint_utils import fingerprints_to_matrix

from src.analysis.diversity import (
    calculate_diversity,
    calculate_reference_similarity
)

from src.analysis.objectives import (
    add_objectives,
    normalise_objectives
)

from src.analysis.plots import (
    chemical_space_pca,
    plot_pca,
    plot_pareto_front,
    plot_thresholds,
    plot_property_distribution,
    plot_reference_similarity,
    plot_elite_distribution,
    plot_generation_progress,  
    plot_objective_radar,
    plot_objective_correlation_heatmap,
    plot_pareto_chemical_space,
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_y_scrambling,
    plot_all_reward_component_pareto_pairs,
    plot_top_molecules_grid
)

from src.analysis.statistics import (
    kruskal_uv_test,
    pairwise_uv_tests,
    hypervolume_significance,
    model_performance_significance,
    y_scrambling_summary
)

from src.analysis.results import (
    top_1_percent_summary,
    method_objective_summary,
    pareto_method_summary,
    hypervolume_summary,
    model_performance_summary,
    feature_importance_summary
)

from src.modelling import (
    train_random_forest,
    get_feature_importance,
    prepare_xy
)

from src.validation import (
    repeated_cross_validate_model,
    y_scrambling,
    applicability_domain
)

from src.cache import (
    save_pickle,
    load_pickle,
    save_metadata,
    cache_valid
)

OCTOCRYLENE_SMILES = (
    "CC(C)(C)c1ccc(cc1)C(=O)"
    "N(c2ccc(cc2)C(C)(C)C)"
)


def run_experiment():

    uv_filter = test_uv_filter()


    # -----------------
    # Dataset creation
    # -----------------

    CACHE_FILE = (
        CACHE_DIR /
        "all_molecules.pkl"
    )
    
    METADATA_FILE = (
        CACHE_DIR /
        "metadata.json"
    )
    
    
    if (
        cache_valid(
            METADATA_FILE,
            PIPELINE_VERSION
        )
        and CACHE_FILE.exists()
    ):
    
        print(
            "Loading cached dataset..."
        )
    
        all_molecules = load_pickle(
            CACHE_FILE
        )
    
    
    else:
    
        print(
            "Generating molecular dataset..."
        )
    
        all_molecules = build_dataset(
            DATASETS,
            uv_filter
        )
    
    
        save_pickle(
            all_molecules,
            CACHE_FILE
        )
    
    
        save_metadata(
            METADATA_FILE,
            {
                "pipeline_version": PIPELINE_VERSION,
                "created": datetime.now().isoformat(),
                "datasets": [
                    str(x)
                    for x in DATASETS.values()
                ],
                "n_molecules": len(all_molecules)
            }
        )

    
    # -----------------
    # Summary analysis
    # -----------------

    summary = summarise_methods(
        all_molecules
    )


    thresholds = score_threshold_summary(
        all_molecules
    )

    diversity = calculate_diversity(
        all_molecules
    )

    # -----------------
    # Optimisation objectives
    # -----------------
    
    all_molecules = add_objectives(
        all_molecules
    )
    
    all_molecules = normalise_objectives(
        all_molecules
    )

    # -----------------
    # Pareto analysis
    # -----------------
    
    pareto_front = calculate_pareto(
        all_molecules
    )

    # -----------------
    # Reward-component Pareto analysis
    # -----------------
    #
    # Replicates the UV filter scorer's own score_lmax/score_os/
    # score_logp transforms (see reward_components.py) to get
    # genuinely monotonic, maximise-is-better versions of the three
    # raw inputs the reward is actually built from. This lets Pareto/
    # hypervolume analysis be run one level closer to the scorer's
    # real mechanics than the 4-objective analysis above, which mixes
    # in SA_inverse/TPSA_score - objectives the reward never trains on.

    all_molecules = add_reward_component_scores(
        all_molecules
    )

    reward_component_columns = get_reward_component_columns()

    reward_component_hv_summary = pareto_efficiency_summary(
        all_molecules,
        objectives=reward_component_columns,
        n_samples=HYPERVOLUME_MC_SAMPLES
    )

    reward_component_significance = hypervolume_significance(
        all_molecules,
        objectives=reward_component_columns,
        n_bootstrap=HV_BOOTSTRAP_ITERATIONS,
        sample_size=HV_BOOTSTRAP_SAMPLE_SIZE,
        hv_samples=HV_BOOTSTRAP_MC_SAMPLES,
        random_state=RANDOM_STATE
    )

    # -----------------
    # Predictive modelling
    # -----------------
    #
    # Determines which molecular descriptors influence
    # MODELLING_TARGET (UV_Filter_Score for the current
    # optimisation project; swap this for an experimental
    # property such as SPF or Lmax for future QSAR work).

    model_result = train_random_forest(
        all_molecules,
        target=MODELLING_TARGET,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    feature_importance = get_feature_importance(
        model_result["model"],
        model_result["features"]
    )

    # -----------------
    # Model validation
    # -----------------

    X, y, model_features = prepare_xy(
        all_molecules,
        MODELLING_TARGET,
        features=model_result["features"]
    )

    cv_results = repeated_cross_validate_model(
        model_result["model"],
        X,
        y,
        k=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE
    )

    scrambling_results = y_scrambling(
        model_result["model"],
        X,
        y,
        n_iterations=Y_SCRAMBLE_ITERATIONS,
        k=CV_FOLDS,
        random_state=RANDOM_STATE
    )

    applicability_domain_df = applicability_domain(
        model_result["X_train"],
        X,
        warning_threshold_multiplier=APPLICABILITY_DOMAIN_MULTIPLIER
    )

    model_performance = model_performance_summary(
        {
            MODELLING_TARGET: cv_results
        }
    )

    feature_importance_table = feature_importance_summary(
        {
            MODELLING_TARGET: feature_importance
        }
    )

    # -----------------
    # Significance testing
    # -----------------
    #
    # Compares methods on hypervolume (via bootstrap resampling,
    # since hypervolume is a single number per method rather
    # than a per-molecule distribution) and on cross-validated
    # model R2, rather than just eyeballing the summary tables.

    hv_significance = hypervolume_significance(
        all_molecules,
        n_bootstrap=HV_BOOTSTRAP_ITERATIONS,
        sample_size=HV_BOOTSTRAP_SAMPLE_SIZE,
        hv_samples=HV_BOOTSTRAP_MC_SAMPLES,
        random_state=RANDOM_STATE
    )

    performance_significance = model_performance_significance(
        all_molecules,
        MODELLING_TARGET,
        k=MODEL_SIGNIFICANCE_CV_FOLDS,
        n_repeats=MODEL_SIGNIFICANCE_CV_REPEATS,
        random_state=RANDOM_STATE
    )

    # -----------------
    # Octocrylene similarity
    # -----------------

    reference_similarity_df = calculate_reference_similarity(
        all_molecules,
        OCTOCRYLENE_SMILES
    )


    oct_similarity_summary = (
        reference_similarity_df
        .groupby("Method")["Reference_Similarity"]
        .agg(
            [
                "mean",
                "median",
                "std",
                "max"
            ]
        )
    )


    # -----------------
    # Statistics
    # -----------------

    kruskal_results = kruskal_uv_test(
        all_molecules
    )


    pairwise_results = pairwise_uv_tests(
        all_molecules
    )


    # -----------------
    # Save tables
    # -----------------

    all_molecules.to_csv(
        TABLE_DIR / "all_molecules.csv",
        index=False
    )


    summary.to_csv(
        TABLE_DIR / "summary.csv"
    )


    thresholds.to_csv(
        TABLE_DIR / "thresholds.csv"
    )


    pareto_front.to_csv(
        TABLE_DIR / "pareto_front.csv",
        index=False
    )


    diversity.to_csv(
        TABLE_DIR / "diversity.csv"
    )


    reference_similarity_df.to_csv(
        TABLE_DIR / "octocrylene_similarity.csv",
        index=False
    )


    oct_similarity_summary.to_csv(
        TABLE_DIR / "octocrylene_similarity_summary.csv"
    )

    # -----------------
    # Results summaries
    # -----------------
    
    top1_summary = top_1_percent_summary(
        all_molecules,
        reference_similarity_df
    )
    
    
    objective_summary = method_objective_summary(
        all_molecules
    )
    
    
    pareto_summary = pareto_method_summary(
        pareto_front,
        all_molecules
    )


    hv_summary = hypervolume_summary(
        all_molecules,
        pareto_front,
        n_samples=HYPERVOLUME_MC_SAMPLES
    )

    y_scrambling_results = y_scrambling_summary(
    scrambling_results,
    Y_SCRAMBLE_ITERATIONS
    )


    kruskal_results.to_csv(
        TABLE_DIR / "kruskal_test.csv",
        index=False
    )


    pairwise_results.to_csv(
        TABLE_DIR / "pairwise_tests.csv",
        index=False
    )

    top1_summary.to_csv(
        TABLE_DIR / "top_1_percent_summary.csv",
        index=False
    )


    objective_summary.to_csv(
        TABLE_DIR / "method_objective_summary.csv",
        index=False
    )
    
    
    pareto_summary.to_csv(
        TABLE_DIR / "pareto_method_summary.csv",
        index=False
    )


    hv_summary.to_csv(
        TABLE_DIR / "hypervolume_summary.csv",
        index=False
    )


    model_performance.to_csv(
        TABLE_DIR / "model_performance.csv",
        index=False
    )


    feature_importance_table.to_csv(
        TABLE_DIR / "feature_importance.csv",
        index=False
    )

    applicability_domain_df.to_csv(
        TABLE_DIR / "applicability_domain.csv"
    )


    hv_significance["bootstrap_distributions"].to_csv(
        TABLE_DIR / "hypervolume_bootstrap_distributions.csv",
        index=False
    )


    hv_significance["kruskal"].to_csv(
        TABLE_DIR / "hypervolume_kruskal_test.csv",
        index=False
    )


    hv_significance["pairwise"].to_csv(
        TABLE_DIR / "hypervolume_pairwise_tests.csv",
        index=False
    )


    performance_significance["fold_r2"].to_csv(
        TABLE_DIR / "model_performance_fold_r2.csv",
        index=False
    )


    performance_significance["kruskal"].to_csv(
        TABLE_DIR / "model_performance_kruskal_test.csv",
        index=False
    )


    performance_significance["pairwise"].to_csv(
        TABLE_DIR / "model_performance_pairwise_tests.csv",
        index=False
    )


    reward_component_hv_summary.to_csv(
        TABLE_DIR / "reward_component_hypervolume_summary.csv",
        index=False
    )


    reward_component_significance["bootstrap_distributions"].to_csv(
        TABLE_DIR / "reward_component_hypervolume_bootstrap_distributions.csv",
        index=False
    )


    reward_component_significance["kruskal"].to_csv(
        TABLE_DIR / "reward_component_hypervolume_kruskal_test.csv",
        index=False
    )


    reward_component_significance["pairwise"].to_csv(
        TABLE_DIR / "reward_component_hypervolume_pairwise_tests.csv",
        index=False
    )

    y_scrambling_results["scrambled_r2_table"].to_csv(
    TABLE_DIR / "y_scrambling.csv",
    index=False
    )
    

    cv_results["fold_metrics"].to_csv(
    TABLE_DIR / "cv_fold_metrics.csv",
    index=False
    )

    # -----------------
    #  Pickle/metadata
    # -----------------
    
    save_pickle(
        model_result["model"],
        CACHE_DIR / "rf_model.pkl"
    )
    
    save_metadata(
        TABLE_DIR / "model_features.json",
        {"features": model_result["features"]}
    )

    save_metadata(
        TABLE_DIR / "y_scrambling_summary.json",
        y_scrambling_results["summary"]
    )
    
    # -----------------
    # Descriptor plots
    # -----------------

    for descriptor in [
        "UV_Filter_Score",
        "Lmax",
        "OS",
        "LogP",
        "TPSA",
        "MolWt",
        "SA_Score"
    ]:
    
        plot_property_distribution(
            all_molecules,
            descriptor,
            FIGURE_DIR / f"{descriptor}_distribution.png"
        )
    
        plot_elite_distribution(
            all_molecules,
            descriptor,
            percentile=1,
            save_path=
            FIGURE_DIR /
            f"elite_{descriptor}_distribution.png"
        )

        plot_generation_progress(
            all_molecules,
            descriptor,
            window=1000,
            save_path=FIGURE_DIR / f"generation_progress_{descriptor}.png"
        )

    for objective in [
        "UV_Filter_Score",
        "SA_inverse",
        "TPSA_score",
        "LogP_score"
    ]:
    
        plot_property_distribution(
            all_molecules,
            objective,
            FIGURE_DIR / f"objective_{objective}_distribution.png"
        )
    
        plot_elite_distribution(
            all_molecules,
            objective,
            percentile=1,
            save_path=
            FIGURE_DIR /
            f"elite_objective_{objective}_distribution.png"
        )  

    plot_objective_radar(
        all_molecules,
        save_path=FIGURE_DIR / "objective_radar.png"
    )


    plot_objective_correlation_heatmap(
        all_molecules,
        save_path=FIGURE_DIR / "objective_correlation_heatmap.png"
    )

    # -----------------
    # Pareto
    # -----------------

    plot_pareto_front(
        all_molecules,
        pareto_front,
        save_path=
        FIGURE_DIR /
        "pareto_front.png"
    )


    plot_all_reward_component_pareto_pairs(
        all_molecules,
        FIGURE_DIR
    )

    # -----------------
    # Modelling / QSAR diagnostics
    # -----------------

    plot_feature_importance(
        feature_importance,
        title=f"Feature Importance: {MODELLING_TARGET}",
        save_path=FIGURE_DIR / f"feature_importance_{MODELLING_TARGET}.png"
    )


    plot_predicted_vs_actual(
        model_result["y_test"],
        model_result["y_pred_test"],
        title=f"Predicted vs Actual: {MODELLING_TARGET}",
        save_path=FIGURE_DIR / f"predicted_vs_actual_{MODELLING_TARGET}.png"
    )


    plot_residuals(
        model_result["y_test"],
        model_result["y_pred_test"],
        title=f"Residuals: {MODELLING_TARGET}",
        save_path=FIGURE_DIR / f"residuals_{MODELLING_TARGET}.png"
    )


    plot_y_scrambling(
        scrambling_results,
        title=f"Y-Scrambling: {MODELLING_TARGET}",
        save_path=FIGURE_DIR / f"y_scrambling_{MODELLING_TARGET}.png"
    )


    # -----------------
    # Thresholds
    # -----------------

    plot_thresholds(
        thresholds,
        FIGURE_DIR / "thresholds.png"
    )


    # -----------------
    # PCA chemical space
    # -----------------

    fp_matrix = fingerprints_to_matrix(
        all_molecules["Fingerprint"]
    )


    methods = all_molecules["Method"].values


    pca_df, variance = chemical_space_pca(
        fp_matrix,
        methods
    )


    plot_pca(
        pca_df,
        variance=variance,
        save_path=FIGURE_DIR / "chemical_space.png"
    )


    pca_df.to_csv(
        TABLE_DIR / "pca_coordinates.csv",
        index=False
    )


    variance_df = pd.DataFrame(
        {
            "Component": [
                "PC1",
                "PC2"
            ],
            "Explained_Variance": variance
        }
    )


    variance_df.to_csv(
        TABLE_DIR / "pca_variance.csv",
        index=False
    )


    plot_pareto_chemical_space(
        pca_df.set_axis(all_molecules.index),
        pareto_front.index,
        save_path=FIGURE_DIR / "pareto_chemical_space_overlay.png"
    )


    # -----------------
    # Octocrylene similarity plot
    # -----------------

    plot_reference_similarity(
        reference_similarity_df,
        FIGURE_DIR / "octocrylene_similarity.png"
    )


    # -----------------
    # Top molecule structures
    # -----------------
    #
    # Structure grid only - no PubChem novelty lookup here, since that
    # needs network access and is rate-limited. Run
    # src/novelty_check.py separately (it re-loads all_molecules.csv,
    # queries PubChem, and re-renders this grid with known/novel
    # labels via the same plot_top_molecules_grid function).
    
    try:

        plot_top_molecules_grid(
            all_molecules,
            n_per_method=TOP_MOLECULES_PER_METHOD,
            rank_by="UV_Filter_Score",
            save_path=FIGURE_DIR / "top_molecules_grid.png"
        )

    except ImportError as e:

        print(
            "Skipping top_molecules_grid.png - RDKit's Draw submodule "
            f"could not be imported ({e}). This is a missing system "
            "library (e.g. libXrender), not a code error - everything "
            "else in this run completed and was saved normally. See "
            "src/novelty_check.py's module docstring or the project "
            "README for fixes (conda-forge RDKit, an HPC X11/Mesa "
            "module, or adding libxrender1 to the container image)."
        )

    except Exception as e:

        print(
            f"Skipping top_molecules_grid.png - unexpected error: {e}"
        )


    return {
        "molecules": all_molecules,
        "summary": summary,
        "thresholds": thresholds,
        "pareto": pareto_front,
        "diversity": diversity,
        "octocrylene_similarity": reference_similarity_df,
        "kruskal": kruskal_results,
        "pairwise": pairwise_results,
        "pca": pca_df,
        "pca_variance": variance_df,
        "top1": top1_summary,
        "objectives": objective_summary,
        "pareto_summary": pareto_summary,
        "hypervolume_summary": hv_summary,
        "reward_component_hypervolume_summary": reward_component_hv_summary,
        "reward_component_hypervolume_significance": reward_component_significance,
        "model": model_result["model"],
        "model_features": model_result["features"],
        "feature_importance": feature_importance,
        "model_performance": model_performance,
        "cv_results": cv_results,
        "y_scrambling": y_scrambling_results,
        "applicability_domain": applicability_domain_df,
        "hypervolume_significance": hv_significance,
        "model_performance_significance": performance_significance
    }