import os
import sys
import numpy as np

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, MolLogP, TPSA
from rdkit.Chem import RDConfig

sys.path.append(
    os.path.join(
        RDConfig.RDContribDir,
        "SA_Score"
    )
)

import sascorer


def calculate_descriptors(df, uv_filter):
    """
    Calculate molecular descriptors and UV-related properties.

    Adds:
        MolWt
        LogP
        TPSA
        SA_Score
        Lmax
        OS

    """

    df = df.copy()

    smiles = df["Canonical_SMILES"].tolist()

    molecules = [
        Chem.MolFromSmiles(s)
        for s in smiles
    ]

    df["MolWt"] = [
        MolWt(mol)
        for mol in molecules
    ]

    df["LogP"] = [
        MolLogP(mol)
        for mol in molecules
    ]

    df["TPSA"] = [
        TPSA(mol)
        for mol in molecules
    ]

    df["SA_Score"] = [
        sascorer.calculateScore(mol)
        for mol in molecules
    ]

    df["Lmax"] = (
        uv_filter
        ._scorer
        .lmax_predictor
        .predict_batch(smiles)
    )

    df["OS"] = (
        uv_filter
        ._scorer
        .os_predictor
        .predict_batch(smiles)
    )

    return df