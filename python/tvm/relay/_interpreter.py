"""The interface to the Evaluator exposed from C++."""
from tvm._ffi.function import _init_api

_init_api("relay._interpreter", __name__)
