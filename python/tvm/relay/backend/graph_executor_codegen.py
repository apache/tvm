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
A compiler from a Relay expression to TVM's graph executor.

The compiler is built from a few pieces.

First we define a compiler from a single Relay expression to the
graph language. We require the expression to be a function.
The function's parameters correspond to the placeholder/inputs
and model parameters found in the computation graph representation.
The body of the function represents the computation graph.

The compiler's output is a program in the graph language, which is composed of
Node, NodeRef, InputNode, OpNode. This "little language" represents programs in
TVM's graph format.

To connect to the graph executor, we use a printer that converts our graph format
into TVM's JSON format. The resulting string can be loaded by
contrib.graph_executor or any other TVM runtime compatible systems.
"""
from tvm.runtime.ndarray import empty
from tvm.relay import _build_module
from tvm.target import Target
from tvm.tir import expr as _expr
from .utils import mangle_module_name


class GraphExecutorCodegen(object):
    """The compiler from Relay to the TVM runtime system."""

    def __init__(self, mod, target):
        self._mod = _build_module._GraphExecutorCodegen()
        self._init = self._mod["init"]
        self._codegen = self._mod["codegen"]
        self._get_graph_json = self._mod["get_graph_json"]
        self._list_params_name = self._mod["list_params_name"]
        self._get_param_by_name = self._mod["get_param_by_name"]
        self._get_irmodule = self._mod["get_irmodule"]
        self._setup(mod, target)

    def _setup(self, mod, target):
        tgts = {}
        if isinstance(target, dict):
            for dev, tgt in target.items():
                if not isinstance(tgt, (str, Target)):
                    raise Exception("Unknown target type")
                tgts[dev] = Target(tgt)
        elif isinstance(target, (str, Target)):
            tgts[_expr.IntImm("int32", 0)] = Target(target)
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
        mod : IRModule or Dict[str, IRModule]
            The lowered functions.
        params : Dict[str, tvm.nd.NDArray]
            Additional constant parameters.
        """
        default_mod_name = mangle_module_name("default")
        self._codegen(func, default_mod_name)
        graph_json = self._get_graph_json()
        lowered_func = self._get_irmodule()
        param_names = self._list_params_name()
        params = {}
        for key in param_names:
            arr = self._get_param_by_name(key)
            param = empty(arr.shape, dtype=arr.dtype, device=arr.device)
            arr.copyto(param)
            params[key] = param
        return graph_json, lowered_func, params
