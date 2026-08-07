from pathlib import Path


# External ACEGEN environment

VENV_PATH = (
    "/rds/projects/h/haywooal-chemical-design/"
    "txm521/venvs/venv310-icelake"
)

ACEGEN_PATH = (
    "/rds/projects/h/haywooal-chemical-design/"
    "txm521/code/acegen-chemical"
)

UV_FILTER_PATH = (
    "/rds/projects/h/haywooal-chemical-design/"
    "txm521/code/acegen-chemical/"
    "acegen/scoring_functions/uv_filter.py"
)


# Project paths

PROJECT_ROOT = Path(
    "/rds/homes/t/txm521/ChemicalAnalysis"
)

DATA_DIR = PROJECT_ROOT / "data"

SRC_DIR = PROJECT_ROOT / "src"

CACHE_DIR = PROJECT_ROOT / "cache"

PIPELINE_VERSION = "1.01"

# Input datasets

DATASETS = {
    "RL": DATA_DIR / "compoundsRL.csv",
    "HC": DATA_DIR / "compoundsHC.csv",
    "NO-RL": DATA_DIR / "compoundsNORL.csv"
}


# External reference data
#
# Real, regulator-approved UV filters with experimentally
# measured Lmax/OS, used to externally validate the uv_filter
# scorer's predictions (see src/validate_uv_filter.py).

ECHA_REFERENCE_PATH = DATA_DIR / "ECHA_UV_Filters.csv"


# Output path

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(
    exist_ok=True
)

FIGURE_DIR.mkdir(
    exist_ok=True
)


# Modelling & validation
#
# Centralised so hyperparameters can be tuned without touching
# run_experiment.py, and so the same values are reused wherever
# they're needed (e.g. RANDOM_STATE for reproducibility across
# modelling, validation, and significance testing).

RANDOM_STATE = 42

MODELLING_TARGET = "UV_Filter_Score"

N_ESTIMATORS = 100

CV_FOLDS = 5

CV_REPEATS = 10

Y_SCRAMBLE_ITERATIONS = 50

APPLICABILITY_DOMAIN_MULTIPLIER = 3.0


# Hypervolume

HYPERVOLUME_MC_SAMPLES = 100_000


# Statistical significance testing
#
# Hypervolume has no natural per-molecule distribution, so
# hypervolume_significance builds one via bootstrap resampling.
# Larger values give a more reliable estimate at the cost of
# runtime (identifying a non-dominated front is O(n^2), repeated
# HV_BOOTSTRAP_ITERATIONS times per method) - see
# analysis.statistics.hypervolume_significance for details.

HV_BOOTSTRAP_ITERATIONS = 50

HV_BOOTSTRAP_SAMPLE_SIZE = 2000

HV_BOOTSTRAP_MC_SAMPLES = 10_000

MODEL_SIGNIFICANCE_CV_FOLDS = 5

MODEL_SIGNIFICANCE_CV_REPEATS = 10


# Top molecule structure grid

TOP_MOLECULES_PER_METHOD = 5