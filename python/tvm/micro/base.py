"""Base definitions for micro."""

from __future__ import absolute_import

import struct
import logging

from .._ffi.function import _init_api
from .._ffi.base import py_str
from ..contrib import util
from ..api import register_func, convert


# how to call micro_init() in program?
def micro_init(device_type):
    _MicroInit(device_type)


_init_api("tvm.micro", "tvm.micro.base")
