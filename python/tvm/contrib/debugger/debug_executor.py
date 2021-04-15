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
"""Graph debug runtime executes TVM debug packed functions."""

import os
import tempfile
import shutil
import tvm._ffi

from tvm._ffi.base import string_types
from tvm.contrib import graph_executor
from tvm.runtime.ndarray import array
from . import debug_result

_DUMP_ROOT_PREFIX = "tvmdbg_"
_DUMP_PATH_PREFIX = "_tvmdbg_"


def create(graph_json_str, libmod, device, dump_root=None):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str
        The graph to be deployed in json format output by graph compiler.
        The graph can contain operator(tvm_op) that points to the name
        of PackedFunc in the libmod.

    libmod : tvm.Module
        The module of the corresponding function.

    device : Device
        The device to deploy the module, can be local or remote.

    dump_root : str
        To select which folder the outputs should be kept.
        None will make a temp folder in /tmp/tvmdbg<rand_string> and does the dumping
    Returns
    -------
    graph_module : GraphModuleDebug
        Debug Runtime graph module that can be used to execute the graph.
    """
    assert isinstance(graph_json_str, string_types)

    try:
        dev, num_rpc_dev, device_type_id = graph_executor.get_device(libmod, device)
        if num_rpc_dev == len(dev):
            fcreate = dev[0]._rpc_sess.get_function("tvm.graph_executor_debug.create")
        else:
            fcreate = tvm._ffi.get_global_func("tvm.graph_executor_debug.create")
    except ValueError:
        raise ValueError(
            "Please set '(USE_GRAPH_EXECUTOR_DEBUG ON)' in "
            "config.cmake and rebuild TVM to enable debug mode"
        )
    func_obj = fcreate(graph_json_str, libmod, *device_type_id)
    return GraphModuleDebug(func_obj, dev, graph_json_str, dump_root)


class GraphModuleDebug(graph_executor.GraphModule):
    """Graph debug runtime module.

    This is a debug wrapper over the TVM runtime.
    Runtime interfaces are wrapped with debug functionalities.
    Manage the debug framework to format the debug data and
    trigger the user interfaces.

    Parameters
    ----------
    module : Module
        The internal tvm module that holds the actual graph functions.

    device : Device
        The device that this module is under.

    graph_json_str : str or graph class
        Content of graph json file in string format

    dump_root : str
        To select which folder the outputs should be kept.
        None will make a temp folder in /tmp/tvmdbg<rand_string> and does the dumping
    """

    def __init__(self, module, device, graph_json_str, dump_root):
        self._dump_root = dump_root
        self._dump_path = None
        self._get_output_by_layer = module["get_output_by_layer"]
        self._run_individual = module["run_individual"]
        self._profile = module["profile"]
        graph_executor.GraphModule.__init__(self, module)
        self._create_debug_env(graph_json_str, device)

    def _format_device(self, device):
        return str(device[0]).upper().replace("(", ":").replace(")", "")

    def _ensure_dir(self, directory):
        """Create a directory if not exists

        Parameters
        ----------

        directory : str
            File path to create
        """
        if not os.path.exists(directory):
            os.makedirs(directory, 0o700)

    def _get_dump_path(self, device):
        """Make the graph and tensor dump folder and return the path.

        Parameters
        ----------
        device : Device
            The device that this module is under.

        Returns
        -------
        path : str
            Directory path where the graph and node outputs will be stored.
        """
        # save to file
        folder_name = _DUMP_PATH_PREFIX + "device_"
        folder_name = folder_name + device.replace(":", "_")
        path = os.path.join(self._dump_root, folder_name)
        self._ensure_dir(path)
        return path

    def _remove_dump_root(self):
        if os.path.isdir(self._dump_root):
            shutil.rmtree(self._dump_root)

    def _create_debug_env(self, graph_json, device):
        """Create UI wrapper framework to handle multiple UI frontends for tvmdbg

        Parameters
        ----------
        graph_json : json format
            json formatted NNVM graph contain list of each node's name, shape and type.

        nodes_list : list
            List of all the nodes presented in the graph

        device : Device
            The device that this module is under.
        """
        # make the dump folder if not given
        if not self._dump_root:
            self._dump_root = tempfile.mkdtemp(prefix=_DUMP_ROOT_PREFIX)

        # format the device
        device = self._format_device(device)

        # updates the dumping directories
        self._dump_path = self._get_dump_path(device)

        # init the debug dumping environment
        self.debug_datum = debug_result.DebugResult(graph_json, self._dump_path)

    def _run_debug(self):
        """Execute the node specified with index will be executed.
        Each debug output will be copied to the buffer
        Time consumed for each execution will be set as debug output.

        """
        self.debug_datum._time_list = [[float(t)] for t in self.run_individual(10, 1, 1)]
        for i, node in enumerate(self.debug_datum.get_graph_nodes()):
            num_outputs = self.debug_datum.get_graph_node_output_num(node)
            for j in range(num_outputs):
                out_tensor = self._get_output_by_layer(i, j)
                out_tensor = array(out_tensor)
                self.debug_datum._output_tensor_list.append(out_tensor)

    def debug_get_output(self, node, out=None):
        """Run graph up to node and get the output to out

        Parameters
        ----------
        node : int / str
            The node index or name

        out : NDArray
            The output array container
        """
        if isinstance(node, str):
            output_tensors = self.debug_datum.get_output_tensors()
            try:
                out = output_tensors[node]
            except KeyError:
                node_list = output_tensors.keys()
                raise RuntimeError(
                    "Node " + node + " not found, available nodes are: " + str(node_list) + "."
                )
        elif isinstance(node, int):
            output_tensors = self.debug_datum._output_tensor_list
            out = output_tensors[node]
        else:
            raise RuntimeError("Require node index or name only.")
        return out

    def run(self, **input_dict):
        """Run forward execution of the graph with debug

        Parameters
        ----------
        input_dict : dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)

        # Step 1. Execute the graph
        self._run_debug()
        # Step 2. Dump the output tensors to the dump folder
        self.debug_datum.dump_output_tensor()
        # Step 3. Dump the Chrome trace to the dump folder
        self.debug_datum.dump_chrome_trace()
        # Step 4. Display the collected information
        self.debug_datum.display_debug_result()

    def run_individual(self, number, repeat=1, min_repeat_ms=0):
        ret = self._run_individual(number, repeat, min_repeat_ms)
        return ret.strip(",").split(",") if ret else []

    def profile(self, **input_dict):
        """Run forward execution of the graph and collect overall and per-op
        performance metrics.

        Parameters
        ----------
        input_dict : dict of str to NDArray
            List of input values to be feed to
        Return
        ------
        timing_results : str
            Per-operator and whole graph timing results in a table format.
        """
        if input_dict:
            self.set_input(**input_dict)

        return self._profile()

    def exit(self):
        """Exits the dump folder and all its contents"""
        self._remove_dump_root()
