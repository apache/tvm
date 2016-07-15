"""Symbolic configuration API."""
from __future__ import absolute_import as _abs
import sys as _sys
import os as _os

try:
    if int(_os.environ.get("NNVM_ENABLE_CYTHON", True)) == 0:
        from .ctypes.symbol import Symbol, Variable
    elif _sys.version_info >= (3, 0):
        from ._cy3.symbol import Symbol, Variable, Group
    else:
        from ._cy2.symbol import Symbol, Variable, Group
except:
    from .ctypes.symbol import Symbol, Variable, Group
