"""Deprecation RPC module"""
# pylint: disable=unused-import
from __future__ import absolute_import as _abs
import warnings
from ..rpc import Server, RPCSession, LocalSession, TrackerSession, connect, connect_tracker

warnings.warn(
    "Please use tvm.rpc instead of tvm.conrtib.rpc. tvm.contrib.rpc is going to be removed in 0.5",
    DeprecationWarning)
