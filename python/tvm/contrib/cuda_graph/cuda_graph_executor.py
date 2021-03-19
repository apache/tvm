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
"""Graph executor with CUDA Graph"""
import tvm._ffi

from tvm._ffi.base import string_types
from tvm.contrib import graph_executor


def create(graph_json_str, libmod, device):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str
        The graph to be deployed in json format output by json graph.
        The graph can contain operator(tvm_op) that points to the name
        of PackedFunc in the libmod.

    libmod : tvm.runtime.Module
        The module of the corresponding function

    device : Device
        The device to deploy the module, only supports CUDA GPU

    Returns
    -------
    graph_module : GraphModuleCudaGraph
        CUDA graph executor module that can be used to execute the graph.

    Note
    ----
    See also :py:class:`tvm.contrib.cuda_graph.cuda_graph_executor.GraphModuleCudaGraph`
    for examples to directly construct a GraphModuleCudaGraph from an exported
    relay compiled library.
    """
    assert isinstance(graph_json_str, string_types)
    try:
        dev, num_rpc_dev, device_type_id = graph_executor.get_device(libmod, device)
        if num_rpc_dev == len(dev):
            fcreate = dev[0]._rpc_sess.get_function("tvm.graph_executor_cuda_graph.create")
        else:
            fcreate = tvm._ffi.get_global_func("tvm.graph_executor_cuda_graph.create")
    except ValueError:
        raise ValueError(
            "To enable CUDA graph support (experimental), please set "
            "'(USE_GRAPH_EXECUTOR_CUGRAPH ON)' in config.cmake and rebuild TVM"
        )

    return GraphModuleCudaGraph(fcreate(graph_json_str, libmod, *device_type_id))


class GraphModuleCudaGraph(graph_executor.GraphModule):
    """CUDA graph executor module.

    This is a CUDA graph executor wrapper over the TVM runtime.
    Runtime interfaces are wrapped with CUDA graph functionalities.

    Parameters
    ----------
    module : Module
        The internal tvm module that holds the actual graph functions.
    """

    def __init__(self, module):
        self._start_capture = module["start_capture"]
        self._end_capture = module["end_capture"]
        self._run_cuda_graph = module["run_cuda_graph"]
        self._cuda_graph_captured = False
        graph_executor.GraphModule.__init__(self, module)

    def capture_cuda_graph(self):
        """Capture a CUDA graph for tvm_op graph

        This should be called before run_cuda_graph() to capture and
        instantiate a CUDA graph instance.
        """
        self._run()  # call cuModuleLoadData before cudaStream API
        self._start_capture()
        self._run()
        self._end_capture()
        self._cuda_graph_captured = True

    def run_cuda_graph(self):
        """Run the CUDA graph for tvm_op graph

        Run the captured CUDA graph instance instead of the
        for-loop kernel launch of default graph executor
        """
        self._run_cuda_graph()

    def run(self, **input_dict):
        """A run wrapper for graph capture / launch, user can just
        change default graph executor to cuda graph executor, and
        the first call will capture a cuda graph for future launch

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        if not self._cuda_graph_captured:
            self.capture_cuda_graph()
        else:
            self._run_cuda_graph()

    def debug_get_output(self, node, out):
        """Run graph up to node and get the output to out

        Parameters
        ----------
        node : int / str
            The node index or name

        out : NDArray
            The output array container
        """
        raise NotImplementedError("Please use debugger.debug_executor as graph_executor instead.")
