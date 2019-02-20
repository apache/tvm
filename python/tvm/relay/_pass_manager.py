# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface to the Pass exposed from C++."""

from .._ffi.function import _init_api

_init_api("relay._pass_manager", __name__)
