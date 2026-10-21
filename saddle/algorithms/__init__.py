"""
One file per method. Each file defines a child of Algorithm and sets
ALGORITHM = <that class>, so the simulator can load it by name:

    load("gd")  ->  saddle/algorithms/gd.py  ->  ALGORITHM
"""
import importlib


def load(name):
    """Return the Algorithm class saved in saddle/algorithms/<name>.py."""
    try:
        module = importlib.import_module(f"saddle.algorithms.{name}")
    except ModuleNotFoundError:
        raise SystemExit(f"No algorithm called '{name}' on this branch. "
                         f"Check out its branch first (git checkout {name}).")
    return module.ALGORITHM
