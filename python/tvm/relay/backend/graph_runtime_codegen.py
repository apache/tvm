# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
A compiler from a Relay expression to TVM's graph runtime.

The compiler is built from a few pieces.

First we define a compiler from a single Relay expression to the
graph langauge. We require the expression to be a function.
The function's parameters correspond to the placeholder/inputs
and model parameters found in the computation graph representation.
The body of the function represents the computation graph.

The compiler's output is a program in the graph language, which is composed of
graph langauge is composed of Node, NodeRef, InputNode, OpNode.
This "little language" represents programs in TVM's graph format.

To connect to the graph runtime, we use a printer that converts our graph format
into TVM's JSON format. The resulting string can be loaded by
contrib.graph_runtime or any other TVM runtime compatible systems.
"""
from __future__ import absolute_import

from tvm.ndarray import empty
from tvm.relay import _build_module
from tvm import target as _target
from tvm import expr as _expr

class GraphRuntimeCodegen(object):
    """The compiler from Relay to the TVM runtime system."""

    def __init__(self, mod, target):
        self._mod = _build_module._GraphRuntimeCodegen()
        self._init = self._mod["init"]
        self._codegen = self._mod["codegen"]
        self._get_graph_json = self._mod["get_graph_json"]
        self._list_params_name = self._mod["list_params_name"]
        self._get_param_by_name = self._mod["get_param_by_name"]
        self._get_lowered_funcs = self._mod["get_lowered_funcs"]
        self._setup(mod, target)

    def _setup(self, mod, target):
        tgts = {}
        if isinstance(target, dict):
            for dev, tgt in target.items():
                if not isinstance(tgt, (str, _target.Target)):
                    raise Exception("Unknown target type")
                tgts[dev] = _target.create(tgt)
        elif isinstance(target, (str, _target.Target)):
            tgts[_expr.IntImm("int32", 0)] = _target.create(target)
        self._init(mod, tgts)

    def codegen(self, func):
        """Compile a single function into a graph.

        Parameters
        ----------
        func: tvm.relay.Expr
            The function to compile.

        Returns
        -------
        graph_json : str
            The graph json that can be consumed by runtime.
        lowered_funcs : List[tvm.LoweredFunc] or Dict[str, List[tvm.LoweredFunc]]
            The lowered functions.
        params : Dict[str, tvm.nd.NDArray]
            Additional constant parameters.
        """
        self._codegen(func)
        graph_json = self._get_graph_json()
        lowered_func = self._get_lowered_funcs()
        param_names = self._list_params_name()
        params = {}
        for name in param_names:
            key = name.value
            arr = self._get_param_by_name(key)
            param = empty(arr.shape, dtype=arr.dtype, ctx=arr.ctx)
            arr.copyto(param)
            params[key] = param
        return graph_json, lowered_func, params
