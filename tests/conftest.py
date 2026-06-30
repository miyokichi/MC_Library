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

sys.path.insert(0, os.path.dirname(__file__))
