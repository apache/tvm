"""Namespace for NNVM-TVM compiler toolchain"""
from __future__ import absolute_import

import tvm

from . import build_module
from . build_module import build, precompute_prune, _run_graph

from .. import symbol as _symbol
from .. import graph as _graph

from .registry import OpPattern
from .registry import register_compute, register_schedule, register_pattern

from .. import top as _top

tvm.register_extension(_symbol.Symbol, _symbol.Symbol)
tvm.register_extension(_graph.Graph, _graph.Graph)
