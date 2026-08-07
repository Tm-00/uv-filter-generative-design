import numpy as np
from rdkit import DataStructs
from tqdm import tqdm


def fingerprints_to_matrix(fps):

    matrix = []

    for fp in tqdm(
        fps,
        desc="Converting fingerprints"
    ):

        arr = np.zeros(
            (2048,),
            dtype=int
        )

        DataStructs.ConvertToNumpyArray(
            fp,
            arr
        )

        matrix.append(arr)

    return np.array(matrix)

def get_fingerprint_matrix(df):

    return fingerprints_to_matrix(
        df["Fingerprint"]
    )