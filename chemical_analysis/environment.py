import sys
from pathlib import Path

from config import (
    PROJECT_ROOT,
    VENV_PATH,
    ACEGEN_PATH,
    OUTPUT_DIR
)


def setup_environment():

    OUTPUT_DIR.mkdir(
    exist_ok=True
    )

    paths = [

        # Your analysis project
        PROJECT_ROOT,

        # HPC virtual environment packages
        Path(VENV_PATH)
        / "lib"
        / "python3.10"
        / "site-packages",

        # ACEGEN source
        ACEGEN_PATH
    ]

    for path in paths:

        path = str(path)

        if path not in sys.path:
            sys.path.insert(0, path)