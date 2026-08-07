from environment import setup_environment

setup_environment()

from src.run_experiment import run_experiment


from src.cache import save_pickle
from config import CACHE_DIR

if __name__ == "__main__":
    results = run_experiment()
    save_pickle(results, CACHE_DIR / "full_results.pkl")
    print("Experiment completed successfully")