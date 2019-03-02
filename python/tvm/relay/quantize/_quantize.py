#pylint: disable=unused-argument
"""Internal module for quantization."""
from __future__ import absolute_import
from tvm._ffi.function import _init_api

_init_api("relay._quantize", __name__)
