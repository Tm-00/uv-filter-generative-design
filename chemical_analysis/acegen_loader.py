import sys
import importlib.util

from config import UV_FILTER_PATH


def load_uv_filter():

    spec = importlib.util.spec_from_file_location(
        "uv_filter",
        UV_FILTER_PATH
    )

    if spec is None:
        raise ImportError(
            f"Could not load uv_filter from {UV_FILTER_PATH}"
        )

    uv_filter = importlib.util.module_from_spec(spec)

    sys.modules["uv_filter"] = uv_filter

    spec.loader.exec_module(uv_filter)

    return uv_filter

def test_uv_filter():

    uv_filter = load_uv_filter()

    result = uv_filter.uv_filter_scorer(["CCO"])

    print("UV filter loaded successfully")
    print(f"Ethanol score: {result[0]}")

    return uv_filter