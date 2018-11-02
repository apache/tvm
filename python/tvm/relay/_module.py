# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface to the Module exposed from C++."""
from tvm._ffi.function import _init_api

_init_api("relay._module", __name__)
