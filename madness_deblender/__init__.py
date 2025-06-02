"""init file."""

import os
import tomli  # or change to 'import tomllib' if Python 3.11+ 

def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    pyproject_path = os.path.join(here, "..", "pyproject.toml")

    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)  # Use tomllib.load(f) in Python 3.11+
    return pyproject_data["tool"]["poetry"]["version"]

__version__ = get_version()
