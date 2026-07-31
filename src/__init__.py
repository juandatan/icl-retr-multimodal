"""Core package for the CUB-200 retrieval-utility pipeline.

The ``data`` alias keeps historical pickle artifacts readable after the package
was moved away from script-level ``sys.path`` mutation.
"""

import sys

from . import data as _data

sys.modules.setdefault("data", _data)
