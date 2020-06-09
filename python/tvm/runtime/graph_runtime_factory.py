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


def create(graph_runtime_kind, graph_json_str, libmod, params, module_name='default'):
    """Create a runtime executor module given a graph and module.
    Parameters
    ----------
    graph_runtime_kind: str
        The kind of graph runtime. Like graphruntime, vm and so on.
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.
    libmod : tvm.Module
        The module of the corresponding function
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
    return GraphRuntimeFactoryModule(fcreate(graph_runtime_kind, graph_json_str, libmod, module_name, *args))


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
        self._select_module = module.get_function("select_module")
        self._import_module = module.get_function("import_module")
        self.selected_module = None
        self.graph_json = module.get_function("get_json")()
        self.lib = module.get_function("get_lib")()
        self.params = {}
        for k, v in module.get_function("get_params")().items():
            self.params[k] = v
        self.iter_cnt = 0
        super(GraphRuntimeFactoryModule, self).__init__(self.module.handle)

    def __del__(self):
        pass

    def runtime_create(self, ctx):
        """Create the runtime using ctx

        Parameters
        ----------
        ctx : TVMContext or list of TVMContext
        """
        ctx, num_rpc_ctx, device_type_id = get_device_ctx(self.selected_module, ctx)
        if num_rpc_ctx == len(ctx):
            fcreate = ctx[0]._rpc_sess.get_function("tvm.graph_runtime_factory.runtime_create")
        else:
            fcreate = get_global_func("tvm.graph_runtime_factory.runtime_create")
        return fcreate(self.selected_module, *device_type_id)

    def import_module(self, mod, mod_name):
        """Create the runtime using ctx

        Parameters
        ----------
        mod : GraphRuntimeFactoryModule
            The graph runtime factory module we want to import
        mod_name: str
            The module name
        """
        return self._import_module(mod, mod_name)

    def __getitem__(self, key='default'):
        """Get specific module

        Parameters
        ----------
        key : str
            The key of module.
        """
        self.selected_module = self._select_module(key)
        self.selected_module._entry = self.runtime_create
        return self.selected_module

    def __iter__(self):
        warnings.warn(
            "legacy graph runtime behaviour of producing json / lib / params will be removed in the next release ",
            DeprecationWarning, 2)
        return self


    def __next__(self):
        if self.iter_cnt > 2:
            raise StopIteration

        objs = [self.graph_json, self.lib, self.params]
        obj = objs[self.iter_cnt]
        self.iter_cnt += 1
        return obj

    def get_json(self):
        return self.graph_json

    def get_lib(self):
        return self.lib

    def get_params(self):
        return self.params