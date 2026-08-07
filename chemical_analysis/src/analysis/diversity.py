from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

import numpy as np
import pandas as pd

from tqdm import tqdm


def scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol
    )



def add_scaffolds(df):

    df = df.copy()

    tqdm.pandas(
        desc="Generating scaffolds"
    )

    df["Scaffold"] = (
        df["Canonical_SMILES"]
        .progress_apply(scaffold)
    )

    return df



def diversity_metrics(
    df,
    sample_size=1000
):

    sample = df.sample(
        min(sample_size, len(df)),
        random_state=42
    )


    fps = [
        fp for fp in sample["Fingerprint"]
        if fp is not None
    ]


    similarities = []

    total = len(fps)


    for i in tqdm(
        range(total),
        desc="Calculating pairwise similarity"
    ):

        similarities.extend(
            DataStructs.BulkTanimotoSimilarity(
                fps[i],
                fps[i+1:]
            )
        )


    return {
        "Mean_similarity": np.mean(similarities),
        "Max_similarity": np.max(similarities),
        "Std_similarity": np.std(similarities)
    }



def calculate_diversity(df):

    results = {}

    for method, group in df.groupby("Method"):

        print(
            f"\nCalculating diversity: {method}"
        )

        results[method] = diversity_metrics(
            group
        )


    return pd.DataFrame(results).T


FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def reference_similarity(
    df,
    reference_smiles
):

    reference_mol = Chem.MolFromSmiles(
        reference_smiles
    )


    reference_fp = FP_GENERATOR.GetFingerprint(
        reference_mol
    )


    similarities = []


    for fp in tqdm(
        df["Fingerprint"],
        desc="Calculating reference similarity"
    ):

        if fp is None:

            similarities.append(
                np.nan
            )

            continue


        similarities.append(
            DataStructs.TanimotoSimilarity(
                fp,
                reference_fp
            )
        )


    result = df.copy()

    result["Reference_Similarity"] = similarities

    return result



def calculate_reference_similarity(
    df,
    reference_smiles
):

    results = []

    for method, group in df.groupby("Method"):

        print(
            f"\nReference similarity: {method}"
        )

        temp = reference_similarity(
            group,
            reference_smiles
        )

        results.append(temp)


    return pd.concat(
        results
    )