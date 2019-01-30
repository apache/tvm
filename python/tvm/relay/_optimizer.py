# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface to the Optimizer exposed from C++."""

from .._ffi.function import _init_api

_init_api("relay._optimize", __name__)
