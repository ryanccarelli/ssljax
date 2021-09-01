import pkgutil
from pathlib import Path

import pytest


def test_imports():
    cwd = Path.cwd()

    # Go to the root directory for ssljax.
    while cwd.stem != "ssljax":
        cwd = cwd.parent

    paths = [str(cwd.joinpath("ssljax").absolute().resolve())]
    for loader, module_name, is_pkg in pkgutil.walk_packages(paths):
        _module = loader.find_module(module_name).load_module(module_name)
