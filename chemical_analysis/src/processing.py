import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from tqdm import tqdm


# Create once, not every function call
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def process_molecules(df, method):
    """
    Process generated molecules from AceGen output.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
            - smiles
            - score_UV_FILTER

    method : str
        Generation method name.

    Returns
    -------
    processed_df : pandas.DataFrame
    """

    records = []

    invalid_count = 0


    for smiles, score in tqdm(
        zip(
            df["smiles"],
            df["score_UV_FILTER"]
        ),
        total=len(df),
        desc=f"Processing molecules ({method})"
    ):

        mol = Chem.MolFromSmiles(
            smiles
        )


        if mol is None:

            invalid_count += 1
            continue


        canonical_smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )


        fingerprint = FP_GENERATOR.GetFingerprint(
            mol
        )


        records.append(
            {
                "Method": method,
                "Original_SMILES": smiles,
                "Canonical_SMILES": canonical_smiles,
                "UV_Filter_Score": float(score),
                "Fingerprint": fingerprint,
            }
        )


    processed_df = pd.DataFrame(
        records
    )


    # -------------------------
    # Quality metrics
    # -------------------------

    processed_df.attrs["invalid_count"] = invalid_count


    processed_df.attrs["validity"] = (
        len(processed_df)
        /
        (len(processed_df) + invalid_count)
    )


    if len(processed_df) > 0:

        processed_df.attrs["uniqueness"] = (
            processed_df["Canonical_SMILES"].nunique()
            /
            len(processed_df)
        )

    else:

        processed_df.attrs["uniqueness"] = 0


    return processed_df