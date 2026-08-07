import pandas as pd

from pathlib import Path

from src.processing import process_molecules
from src.descriptors import calculate_descriptors
from src.analysis.objectives import add_objectives


def prepare_dataset(df, method, uv_filter):

    df = process_molecules(
        df,
        method=method
    )

    df = calculate_descriptors(
        df,
        uv_filter
    )

    df = add_objectives(
        df
    )

    return df



def build_dataset(
    datasets,
    uv_filter
):

    validate_datasets(
        datasets
    )

    processed = []

    for method, csv_path in datasets.items():

        df = pd.read_csv(
            csv_path,
            dtype=str
        )

        df = prepare_dataset(
            df,
            method,
            uv_filter
        )

        processed.append(df)


    return pd.concat(
        processed,
        ignore_index=True
    )



def validate_datasets(datasets):

    for method, path in datasets.items():

        if not Path(path).exists():

            raise FileNotFoundError(
                f"{method} dataset not found: {path}"
            )