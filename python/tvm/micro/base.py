"""Base definitions for micro."""

from __future__ import absolute_import

import struct
import logging

from .._ffi.function import _init_api
from .._ffi.base import py_str
from ..contrib import util
from ..api import register_func, convert


def micro_init(device_type, init_source, port=0):
    _MicroInit(device_type, init_source, port)


_init_api("tvm.micro", "tvm.micro.base")
