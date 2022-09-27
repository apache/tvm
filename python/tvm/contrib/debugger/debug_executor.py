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

import logging
import os
import shutil
import struct
import tempfile

import tvm._ffi
from tvm._ffi.base import string_types
from tvm.contrib import graph_executor
from tvm.runtime.module import BenchmarkResult

from ...runtime.profiling import Report
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
            "Please set '(USE_PROFILER ON)' in " "config.cmake and rebuild TVM to enable debug mode"
        )
    func_obj = fcreate(graph_json_str, libmod, *device_type_id)
    gmod = GraphModuleDebug(func_obj, dev, graph_json_str, dump_root)

    # Automatically set params if they can be extracted from the libmod
    try:
        params = libmod["get_graph_params"]()
    except (AttributeError, tvm.error.RPCError):
        # Params can not be extracted from the libmod and must be set somewhere else manually
        # Do not set params during RPC communication
        pass
    else:
        gmod.set_input(**params)

    return gmod


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
        self._run_individual = module["run_individual"]
        self._run_individual_node = module["run_individual_node"]
        self._debug_get_output = module["debug_get_output"]
        self._execute_node = module["execute_node"]
        self._get_node_output = module["get_node_output"]
        self._profile = module["profile"]
        self._profile_rpc = module["profile_rpc"]
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

    def _execute_next_node(self, node_index, output_index):
        """Execute node assuming all previous nodes has been executed.
        Return the output of this node.

        Parameters
        ----------
        node_index : int
            The node index
        output_index: int
            The node output index
        Return
        ------
        output_tensors : Array<NDarray>
            Array of output tensors
        """
        output_tensors = self._execute_next_node_get_output(node_index, output_index)
        return output_tensors

    def _run_per_layer(self):
        """Execute up to each node and each debug output will be
        copied to the buffer.

        """
        output_tensors = []
        for i, node in enumerate(self.debug_datum.get_graph_nodes()):
            self._execute_node(i)
            num_outputs = self.debug_datum.get_graph_node_output_num(node)
            for j in range(num_outputs):
                logging.info(
                    "running node=%d, output_ind=%d, with node_name: %s", i, j, node["name"]
                )
                output_tensors.append(self._get_node_output(i, j))
        self.debug_datum.update_output_tensors(output_tensors)

    def _run_debug(
        self,
        number,
        repeat,
        min_repeat_ms,
        limit_zero_time_iterations,
        cooldown_interval_ms,
        repeats_to_cooldown,
    ):
        """Execute the node specified with index will be executed.
        Each debug output will be copied to the buffer
        Time consumed for each execution will be set as debug output.
        """
        # Get timing.
        self.debug_datum._time_list = self.run_individual(
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            limit_zero_time_iterations=limit_zero_time_iterations,
            cooldown_interval_ms=cooldown_interval_ms,
            repeats_to_cooldown=repeats_to_cooldown,
        )

        # Get outputs.
        self._run_per_layer()

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
            node_index = None
            for i, graph_node in enumerate(self.debug_datum.get_graph_nodes()):
                if graph_node["name"] == node:
                    node_index = i
                    break
            else:
                raise AttributeError(f"Could not find a node named {node} in this graph.")
        elif isinstance(node, int):
            node_index = node
        else:
            raise RuntimeError(f"Require node index or name only.")

        self._debug_get_output(node_index, out)

    # pylint: disable=arguments-differ
    def run(
        self,
        number=10,
        repeat=1,
        min_repeat_ms=1,
        limit_zero_time_iterations=100,
        cooldown_interval_ms=0,
        repeats_to_cooldown=1,
        sort_by_time=True,
        **input_dict,
    ):
        """Run forward execution of the graph with debug

        Parameters
        ----------
        number: int, optional
            The number of times to run this function for taking average.
            We call these runs as one `repeat` of measurement.

        repeat: int, optional
            The number of times to repeat the measurement.
            In total, the function will be invoked (1 + number x repeat) times,
            where the first one is warm up and will be discarded.
            The returned result contains `repeat` costs,
            each of which is an average of `number` costs.

        min_repeat_ms: int, optional
            The minimum duration of one `repeat` in milliseconds.
            By default, one `repeat` contains `number` runs. If this parameter is set,
            the parameters `number` will be dynamically adjusted to meet the
            minimum duration requirement of one `repeat`.
            i.e., When the run time of one `repeat` falls below this time, the `number` parameter
            will be automatically increased.

        limit_zero_time_iterations: int, optional
            The maximum number of repeats when measured time is equal to 0.
            It helps to avoid hanging during measurements.

        cooldown_interval_ms: int, optional
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: int, optional
            The number of repeats before the cooldown is activated.

        sort_by_time: bool, optional
            Whether to sort the debug output by time.

        input_dict : dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)

        # Step 1. Execute the graph
        self._run_debug(
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            limit_zero_time_iterations=limit_zero_time_iterations,
            cooldown_interval_ms=cooldown_interval_ms,
            repeats_to_cooldown=repeats_to_cooldown,
        )
        # Step 2. Dump the output tensors to the dump folder
        self.debug_datum.dump_output_tensor()
        # Step 3. Dump the Chrome trace to the dump folder
        self.debug_datum.dump_chrome_trace()
        # Step 4. Display the collected information
        self.debug_datum.display_debug_result(sort_by_time)

    def run_individual(
        self,
        number,
        repeat=1,
        min_repeat_ms=0,
        limit_zero_time_iterations=100,
        cooldown_interval_ms=0,
        repeats_to_cooldown=1,
    ):
        """Run each operation in the graph and get the time per op for all ops.

        number: int
            The number of times to run this function for taking average.
            We call these runs as one `repeat` of measurement.

        repeat: int, optional
            The number of times to repeat the measurement.
            In total, the function will be invoked (1 + number x repeat) times,
            where the first one is warm up and will be discarded.
            The returned result contains `repeat` costs,
            each of which is an average of `number` costs.

        min_repeat_ms: int, optional
            The minimum duration of one `repeat` in milliseconds.
            By default, one `repeat` contains `number` runs. If this parameter is set,
            the parameters `number` will be dynamically adjusted to meet the
            minimum duration requirement of one `repeat`.
            i.e., When the run time of one `repeat` falls below this time, the `number` parameter
            will be automatically increased.

        limit_zero_time_iterations: int, optional
            The maximum number of repeats when measured time is equal to 0.
            It helps to avoid hanging during measurements.

        cooldown_interval_ms: int, optional
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: int, optional
            The number of repeats before the cooldown is activated.

        Returns
        -------
        A 2-dimensional array where the dimensions are: the index of the operation and
        the repeat of the measurement.
        """
        res = self._run_individual(
            number,
            repeat,
            min_repeat_ms,
            limit_zero_time_iterations,
            cooldown_interval_ms,
            repeats_to_cooldown,
        )
        results = []
        offset = 0
        format_size = "@q"
        (nodes_count,) = struct.unpack_from(format_size, res, offset)
        offset += struct.calcsize(format_size)
        format_data = "@" + repeat * "d"
        for _ in range(0, nodes_count):
            ret = struct.unpack_from(format_data, res, offset)
            offset += struct.calcsize(format_data)
            results.append([*ret])
        return results

    def run_individual_node(
        self,
        index,
        number=10,
        repeat=1,
        min_repeat_ms=0,
        limit_zero_time_iterations=100,
        cooldown_interval_ms=0,
        repeats_to_cooldown=1,
    ):
        """Benchmark a single node in the serialized graph.

        This does not do any data transfers and uses arrays already on the device.

        Parameters
        ----------
        index : int
            The index of the node, see `self.debug_datum.get_graph_nodes`

        number: int
            The number of times to run this function for taking average.
            We call these runs as one `repeat` of measurement.

        repeat: int, optional
            The number of times to repeat the measurement.
            In total, the function will be invoked (1 + number x repeat) times,
            where the first one is warm up and will be discarded.
            The returned result contains `repeat` costs,
            each of which is an average of `number` costs.

        min_repeat_ms : int, optional
            The minimum duration of one `repeat` in milliseconds.
            By default, one `repeat` contains `number` runs. If this parameter is set,
            the parameters `number` will be dynamically adjusted to meet the
            minimum duration requirement of one `repeat`.
            i.e., When the run time of one `repeat` falls below this time, the `number` parameter
            will be automatically increased.

        limit_zero_time_iterations: int, optional
            The maximum number of repeats when measured time is equal to 0.
            It helps to avoid hanging during measurements.

        cooldown_interval_ms: int, optional
            The cooldown interval in milliseconds between the number of repeats defined by
            `repeats_to_cooldown`.

        repeats_to_cooldown: int, optional
            The number of repeats before the cooldown is activated.

        Returns
        -------
        A module BenchmarkResult
        """
        # Results are returned as serialized strings which we deserialize
        res = self._run_individual_node(
            index,
            number,
            repeat,
            min_repeat_ms,
            limit_zero_time_iterations,
            cooldown_interval_ms,
            repeats_to_cooldown,
        )
        fmt = "@" + ("d" * repeat)
        results = struct.unpack(fmt, res)
        return BenchmarkResult(list(results))

    def profile(self, collectors=None, **input_dict):
        """Run forward execution of the graph and collect overall and per-op
        performance metrics.

        Parameters
        ----------
        collectors : Optional[Sequence[MetricCollector]]
            Extra metrics to collect. If profiling over RPC, collectors must be `None`.

        input_dict : dict of str to NDArray
            List of input values to be feed to

        Return
        ------
        timing_results : str
            Per-operator and whole graph timing results in a table format.
        """
        if input_dict:
            self.set_input(**input_dict)

        if self.module.type_key == "rpc":
            # We cannot serialize MetricCollectors over RPC
            assert collectors is None, "Profiling with collectors is not supported over RPC"
            return Report.from_json(self._profile_rpc())
        return self._profile(collectors)

    def exit(self):
        """Exits the dump folder and all its contents"""
        self._remove_dump_root()
