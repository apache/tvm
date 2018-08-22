# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface to the Environment exposed from C++."""
from tvm._ffi.function import _init_api

_init_api("relay._env", __name__)
