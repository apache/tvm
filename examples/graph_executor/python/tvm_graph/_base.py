from __future__ import absolute_import as _abs
import os
import sys

if sys.version_info[0] == 3:
    import builtins as __builtin__
else:
    import __builtin__

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

if hasattr(__builtin__, "NNVM_BASE_PATH"):
    assert __builtin__.NNVM_BASE_PATH == curr_path
else:
    __builtin__.NNVM_BASE_PATH = curr_path

if hasattr(__builtin__, "NNVM_LIBRARY_NAME"):
    assert __builtin__.NNVM_LIBRARY_NAME == curr_path
else:
    __builtin__.NNVM_LIBRARY_NAME = "libtvm_graph_exec"
