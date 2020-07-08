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
"""Graph runtime factory."""
import numpy as np
import warnings
from tvm._ffi.base import string_types
from tvm._ffi.registry import get_global_func
from tvm._ffi.runtime_ctypes import TVMContext
from tvm.contrib.graph_runtime import get_device_ctx
from .packed_func import _set_class_module
from tvm.rpc import base as rpc_base
from .module import Module
from . import ndarray


def create(graph_json_str, libmod, libmod_name, params):
    """Create a runtime executor module given a graph and module.
    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.
    libmod : tvm.Module
        The module of the corresponding function
    libmod_name: str
        The name of module
    params : dict of str to NDArray
        The parameters of module

    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.
    """
    if not isinstance(graph_json_str, string_types):
        try:
            graph_json_str = graph_json_str._tvm_graph_json()
        except AttributeError:
            raise ValueError("Type %s is not supported" % type(graph_json_str))
    fcreate = get_global_func("tvm.graph_runtime_factory.create")
    args = []
    for k, v in params.items():
        args.append(k)
        args.append(ndarray.array(v))
    return GraphRuntimeFactoryModule(fcreate(graph_json_str, libmod, libmod_name, *args))


class GraphRuntimeFactoryModule(Module):
    """Graph runtime factory module.

    This is a module of graph runtime factory

    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.

    Attributes
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.
    """

    def __init__(self, module):
        self.module = module
        self.graph_json = None
        self.lib = None
        self.params = {}
        self.iter_cnt = 0
        super(GraphRuntimeFactoryModule, self).__init__(self.module.handle)

    def __del__(self):
        pass

    def __iter__(self):
        warnings.warn(
            "legacy graph runtime behaviour of producing json / lib / params will be removed in the next release ",
            DeprecationWarning, 2)
        self.graph_json = self.module["get_json"]()
        self.lib = self.module["get_lib"]()
        for k, v in self.module["get_params"]().items():
            self.params[k] = v
        return self


    def __next__(self):
        if self.iter_cnt > 2:
            raise StopIteration

        objs = [self.graph_json, self.lib, self.params]
        obj = objs[self.iter_cnt]
        self.iter_cnt += 1
        return obj