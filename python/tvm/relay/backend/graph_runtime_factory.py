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
import warnings
from tvm._ffi.base import string_types
from tvm._ffi.registry import get_global_func
from tvm.runtime import ndarray


class GraphRuntimeFactoryModule(object):
    """Graph runtime factory module.
    This is a module of graph runtime factory

    Parameters
    ----------
    graph_json_str : str
        The graph to be deployed in json format output by graph compiler.
        The graph can contain operator(tvm_op) that points to the name of
        PackedFunc in the libmod.
    libmod : tvm.Module
        The module of the corresponding function
    libmod_name: str
        The name of module
    params : dict of str to NDArray
        The parameters of module
    """

    def __init__(self, graph_json_str, libmod, libmod_name, params):
        assert isinstance(graph_json_str, string_types)
        fcreate = get_global_func("tvm.graph_runtime_factory.create")
        args = []
        for k, v in params.items():
            args.append(k)
            args.append(ndarray.array(v))
        self.module = fcreate(graph_json_str, libmod, libmod_name, *args)
        self.graph_json = graph_json_str
        self.lib = libmod
        self.libmod_name = libmod_name
        self.params = params
        self.iter_cnt = 0

    def export_library(self, file_name, fcompile=None, addons=None, **kwargs):
        return self.module.export_library(file_name, fcompile, addons, **kwargs)

    # Sometimes we want to get params explicitly.
    # For example, we want to save its params value to
    # an independent file.
    def get_params(self):
        return self.params

    def get_json(self):
        return self.graph_json

    def get_lib(self):
        return self.lib

    def __getitem__(self, item):
        return self.module.__getitem__(item)

    def __iter__(self):
        warnings.warn(
            "legacy graph runtime behaviour of producing json / lib / params will be "
            "removed in the next release ",
            DeprecationWarning,
            2,
        )
        return self

    def __next__(self):
        if self.iter_cnt > 2:
            raise StopIteration

        objs = [self.graph_json, self.lib, self.params]
        obj = objs[self.iter_cnt]
        self.iter_cnt += 1
        return obj
