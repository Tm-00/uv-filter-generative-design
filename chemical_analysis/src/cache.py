import pickle
import json

from pathlib import Path
from datetime import datetime


def save_pickle(
    obj,
    path
):

    path = Path(path)

    with open(path, "wb") as f:
        pickle.dump(
            obj,
            f
        )


def load_pickle(path):

    if not Path(path).exists():

        raise FileNotFoundError(
            f"Cache file missing: {path}"
        )

    with open(path, "rb") as f:

        return pickle.load(f)



def save_metadata(
    path,
    metadata
):

    path = Path(path)

    with open(path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4,
            default=str
        )


def load_metadata(
    path
):

    path = Path(path)

    with open(path, "r") as f:
        return json.load(f)



def cache_valid(
    metadata_file,
    version
):

    if not Path(metadata_file).exists():
        return False


    metadata = load_metadata(
        metadata_file
    )

    return (
        metadata.get("pipeline_version")
        ==
        version
    )