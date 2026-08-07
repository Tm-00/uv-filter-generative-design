import json
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

from config import TABLE_DIR, FIGURE_DIR, TOP_MOLECULES_PER_METHOD

from src.analysis.plots import plot_top_molecules_grid


PUBCHEM_INCHIKEY_URL = (
    "https://pubchem.ncbi.gov/rest/pug/compound/"
    "inchikey/{inchikey}/cids/JSON"
)

# PubChem's unauthenticated rate limit is ~5 requests/second.
# This is kept conservative since a lookup only needs to run once
# per unique molecule, not once per pipeline run.
REQUEST_DELAY_SECONDS = 0.25

CACHE_PATH = TABLE_DIR / "pubchem_novelty_cache.json"


def load_cache(path=CACHE_PATH):

    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


def save_cache(cache, path=CACHE_PATH):

    path = Path(path)

    with open(path, "w") as f:
        json.dump(
            cache,
            f,
            indent=2
        )


def smiles_to_inchikey(smiles):

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:
        return None

    return Chem.MolToInchiKey(
        mol
    )


def check_pubchem(
    inchikey,
    cache,
    timeout=10
):
    """
    Query PubChem PUG-REST for a given InChIKey.

    Returns
    -------
    bool
        True if a matching PubChem CID was found (i.e. the compound is
        already known to PubChem), False if no match was found.

        This is evidence of novelty, not proof: PubChem is large but
        not exhaustive, so a "not found" result should be reported as
        "not found in PubChem" / "suggests novelty", not as a
        confirmed-novel claim.

    None
        If the lookup itself failed (network error, timeout, rate
        limit) - distinct from a genuine "not found", so these can be
        retried or reported separately rather than silently counted
        as novel.
    """

    if inchikey in cache:
        return cache[inchikey]

    url = PUBCHEM_INCHIKEY_URL.format(
        inchikey=inchikey
    )

    try:

        response = requests.get(
            url,
            timeout=timeout
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        if response.status_code == 200:

            result = True

        elif response.status_code == 404:

            # PubChem returns 404 for "no matching compound"
            result = False

        else:

            # Rate limited, server error, etc. - don't cache, so it
            # gets retried on a future run rather than silently
            # treated as novel.
            return None

    except requests.exceptions.RequestException:

        return None

    cache[inchikey] = result

    return result


def check_novelty(
    df,
    n_per_method=5,
    rank_by="UV_Filter_Score"
):
    """
    Check the top-N molecules per method against PubChem by InChIKey.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain "Method", "Canonical_SMILES", and `rank_by`.

    n_per_method : int
        How many top molecules to check per method.

    rank_by : str
        Column used to select the top molecules within each method.

    Returns
    -------
    pandas.DataFrame
        The selected top molecules with added columns:
        InChIKey, Found_In_PubChem (True/False/None), Novel_Suggested
        (True only where Found_In_PubChem is False - i.e. excludes
        failed lookups from being counted as novel).
    """

    cache = load_cache()

    rows = []

    for method in df["Method"].unique():

        top = df[
            df["Method"] == method
        ].nlargest(
            n_per_method,
            rank_by
        )

        rows.append(
            top
        )

    selected = pd.concat(
        rows
    ).copy()

    inchikeys = []
    found = []

    for smiles in selected["Canonical_SMILES"]:

        key = smiles_to_inchikey(
            smiles
        )

        inchikeys.append(
            key
        )

        if key is None:

            found.append(
                None
            )

            continue

        found.append(
            check_pubchem(
                key,
                cache
            )
        )

    save_cache(
        cache
    )

    selected["InChIKey"] = inchikeys

    selected["Found_In_PubChem"] = found

    selected["Novel_Suggested"] = (
        selected["Found_In_PubChem"] == False  # noqa: E712
    )

    return selected


def run_novelty_check(
    all_molecules_path=None,
    n_per_method=TOP_MOLECULES_PER_METHOD,
    rank_by="UV_Filter_Score"
):
    """
    Run the full novelty check pipeline: load molecules, query PubChem
    for the top-N per method, save the results table, and render a
    structure grid labelled with known/novel status.
    """

    if all_molecules_path is None:

        all_molecules_path = TABLE_DIR / "all_molecules.csv"

    df = pd.read_csv(
        all_molecules_path
    )

    results = check_novelty(
        df,
        n_per_method=n_per_method,
        rank_by=rank_by
    )

    n_failed = results["Found_In_PubChem"].isna().sum()

    if n_failed > 0:

        print(
            f"Warning: {n_failed} lookups failed (network/rate limit) "
            "and were left blank rather than counted as novel. "
            "Re-run to retry - completed lookups are cached."
        )

    print(
        results[
            [
                "Method",
                "Canonical_SMILES",
                rank_by,
                "Found_In_PubChem",
                "Novel_Suggested"
            ]
        ].to_string()
    )

    results.to_csv(
        TABLE_DIR / "top_molecules_novelty.csv",
        index=False
    )

    try:

        plot_top_molecules_grid(
            results,
            n_per_method=n_per_method,
            rank_by=rank_by,
            novelty_column="Novel_Suggested",
            save_path=FIGURE_DIR / "top_molecules_grid.png"
        )

    except ImportError as e:

        print(
            "Skipping top_molecules_grid.png - RDKit's Draw submodule "
            f"could not be imported ({e}). This needs a system-level "
            "2D rendering library (commonly libXrender) that isn't "
            "present here. The novelty results above were still saved "
            "to top_molecules_novelty.csv. Fixes: install RDKit via "
            "conda-forge instead of pip (its build declares this "
            "dependency and pulls it in automatically), check for an "
            "X11/Mesa module on your HPC system (e.g. `module spider "
            "X11`), or add libxrender1 to your container image."
        )

    return results


if __name__ == "__main__":
    run_novelty_check()
