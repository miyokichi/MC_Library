"""Make the tests directory importable.

``backend="processes"`` uses a process pool; on the ``spawn`` start method
(the default on Windows and macOS) the child processes pickle the trial *by
reference* and re-import its module.  multiprocessing forwards the parent's
``sys.path`` to the children, so inserting this directory here lets the workers
import :mod:`mp_helpers`.
"""

from __future__ import annotations

import os
import sys

_TESTS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.dirname(_TESTS_DIR)

sys.path.insert(0, _TESTS_DIR)
# Also expose the repo root so tests can import the ``examples`` package
# (e.g. the article's numeric algorithms demonstrated in whyfp_numeric.py).
sys.path.insert(0, _REPO_ROOT)
