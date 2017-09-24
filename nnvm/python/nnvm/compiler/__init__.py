"""NNVM compiler toolchain.

User only need to use :any:`build` and :any:`build_config` to do the compilation.
The other APIs are for more advanced interaction with the compiler toolchain.
"""
from __future__ import absolute_import

import tvm

from . import build_module
from . build_module import build, optimize, build_config
from . compile_engine import engine, graph_key

from .. import symbol as _symbol
from .. import graph as _graph

from .. import top as _top


tvm.register_extension(_symbol.Symbol, _symbol.Symbol)
tvm.register_extension(_graph.Graph, _graph.Graph)
