"""
One file per method. Each file defines a child of Algorithm and ends with
ALGORITHM = <that class>, so the program can find it by its file name:

    load("gd")  ->  saddle/algorithms/gd.py  ->  ALGORITHM

available() lists every method that exists on the branch you are on
(on main there are none; on a method branch there is one; on all-methods all four).
"""
import importlib
import pkgutil
import os

HERE = os.path.dirname(__file__)
NOT_METHODS = {"base"}

# The methods the window's picker shows, in this order. None = every method on the branch.
# (A branch with many methods can list only the ones being compared.)
PICKER = ["gd", "pgd", "gdnc", "jin_pgd", "probe_nesterov", "ldh", "ldh_guessed"]


def available():
    """File names of all methods on this branch, e.g. ['gd', 'pgd']."""
    return sorted(m.name for m in pkgutil.iter_modules([HERE]) if m.name not in NOT_METHODS)


def shown():
    """The methods for the window's picker: PICKER if set (only those on this branch)."""
    names = available()
    return [n for n in PICKER if n in names] if PICKER else names


def load(name):
    """Return the Algorithm class saved in saddle/algorithms/<name>.py."""
    if name not in available():
        raise SystemExit(f"No algorithm called '{name}' on this branch "
                         f"(found: {available() or 'none'}). Check out its branch first.")
    return importlib.import_module(f"saddle.algorithms.{name}").ALGORITHM
