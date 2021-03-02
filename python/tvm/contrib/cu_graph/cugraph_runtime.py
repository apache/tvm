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
"""Graph runtime test cuGraph"""
import tvm._ffi

from tvm._ffi.base import string_types
from tvm.contrib import graph_runtime


def create(graph_json_str, libmod, ctx):
    assert isinstance(graph_json_str, string_types)
    try:
        ctx, num_rpc_ctx, device_type_id = graph_runtime.get_device_ctx(libmod, ctx)
        if num_rpc_ctx == len(ctx):
            pass
        else:
            fcreate = tvm._ffi.get_global_func("tvm.graph_runtime_cugraph.create")
    except ValueError:
        raise ValueError(
            "Please set '(USE_GRAPH_RUNTIME_CUGRAPH ON)' in "
            "config.cmake and rebuild TVM to enable cu_graph test mode"
        )

    func_obj = fcreate(graph_json_str, libmod, *device_type_id)
    return GraphModuleCuGraph(func_obj, ctx, graph_json_str)


class GraphModuleCuGraph(graph_runtime.GraphModule):
    def __init__(self, module, ctx, graph_json_str):

        self._start_capture = module["start_capture"]
        self._end_capture = module["end_capture"]
        self._run_cuda_graph = module["run_cuda_graph"]

        graph_runtime.GraphModule.__init__(self, module)

    def capture_cuda_graph(self):
        # call cuModuleLoadData before cudaStream API 
        self._run()

        print("====== Start Stream Capture ======")
        self._start_capture()
        print("====== Start Run Ops On Stream ======")
        self._run()
        print("====== End Stream Capture ======")
        self._end_capture()
        

    def run_cuda_graph(self):
        self._run_cuda_graph()

