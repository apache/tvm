# pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-instance-attributes,too-many-branches,too-many-nested-blocks
"""Base class for graph tuner."""
import logging
import json

from abc import abstractmethod
from nnvm.compiler import graph_attr
from ..utils import get_wkl_map, get_real_node, is_input_node, shape2layout, \
    get_in_nodes, get_out_nodes, is_elemlike_op
from ..tensor_executor.base_tensor_executor import RPCMode
from ..tensor_executor import LayoutTransformExecutor
from ..utils import log_msg

class BaseGraphExecutor(object):
    """Class to search schedules considering both kernel execution time and
    layout transformation time.

    Before creating a Graph Executor instance, schedule candidates for all kernels in
    graph should be provided through tensor searching.

    TODO Develop more effective approximation/learning algorithm.
    """
    def __init__(self, graph, input_shapes, sch_dict, target_op, data_layout,
                 get_wkl_func, infer_layout_shape_func, max_sch_num=20, verbose=True,
                 log_file="graph_tuner.log", log_level=logging.DEBUG,
                 name="graph_tuner"):
        """Create a GlobalTuner instance. Local schedule searching for all nodes with
        target_op in the input graph and layout transformation benchmark need to be
        executed before initialization.

        graph : nnvm Graph
            Input graph

        input_shapes : dict of str to tuple.
            Input shapes of graph

        sch_dict : dict of namedtuple to list of dict
            Schedule candidates for all workloads. Key is workload and value is a
            list of dictionary, which in format {"schedule": sch, "time": execution_time}.
            Time value is in millisecond.

            This dictionary can be created through search_schedule module.

        layout_time_dict : dict of string to float:
            Dictionary for layout transformation time. Key should be of format
            "(input_shape, output_shape)".

        get_wkl_func : function
             Function to convert target_op nodes in a graph to workloads.
             Check get_workload.get_conv2d_workload for reference implementation.

        infer_layout_shape_func : function
            Function to infer actual input and output shapes for layout
            transformation given a workload, current schedule and target schedule.

            Take a CNN as example, a layout transformation can happen
            in two cases:
            1. Between two convolution nodes. Data shape before and after
               layout transformation can be determined purely by workload
               and schedules.
            2. Before element-wise like nodes. Element-wise like nodes
               are defined in _base module. In this case, shape of the
               element-wise like node is required as well.

            Arguments for this function should be (wkl, current_sch, target_sch,
            batch_size, is_elemlike, elemlike_shape), and it should return input_shape,
            output_shape and is_valid. Check utils.infer_layout_shape_avx for reference
            implementation.

        max_sch_num : int, optional
            Maximum number of schedule candidates for each workload.

        target_op : str, optional
            Operator name.

        name : str, optional
            Name of global tuner.
        """
        self._wkl_dict = {}
        self._sch_dict = {}
        self._layout_transform_dict = {}
        self._elemlike_shape_dict = {}
        self. _stage_dict = {}
        self._dep_dict = {}
        self._counted_nodes_set = set()
        self._input_shapes = input_shapes
        self._target_op = target_op
        self._name = name
        self._max_sch_num = max_sch_num
        self._optimal_sch_dict = {}
        self._data_layout = data_layout
        self._get_wkl_func = get_wkl_func
        self._infer_layout_shape_func = infer_layout_shape_func

        self._verbose = verbose
        self._log_level = log_level
        self._log_file = log_file
        self._formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self._formatter)
        self._file_logger = logging.getLogger(__name__ + "_file_logger")
        self._file_logger.addHandler(file_handler)
        self._file_logger.setLevel(log_level)
        if self._verbose:
            self._console_logger = logging.getLogger(__name__ + "_console_logger")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._formatter)
            self._console_logger.addHandler(console_handler)
            self._console_logger.setLevel(log_level)
            self._console_logger.propagate = False
        else:
            self._console_logger = None

        graph = graph_attr.set_shape_inputs(graph, input_shapes)
        graph = graph.apply("InferShape")
        self._graph = graph
        self._in_nodes_dict = get_in_nodes(self._graph, "conv2d", input_shapes.keys())
        self._out_nodes_dict = get_out_nodes(self._in_nodes_dict)
        g_dict = json.loads(self._graph.json())
        self._node_list = g_dict["nodes"]

        # Generate workload and schedule dictionaries.
        workload_list = list(sch_dict.keys())
        self._node_map = get_wkl_map(self._graph, input_shapes, workload_list, target_op,
                                     self._get_wkl_func)
        for key, _ in self._in_nodes_dict.items():
            node_name = self._node_list[key]["name"]
            if node_name in self._input_shapes.keys():
                continue
            if self._node_list[key]["op"] == target_op:
                self._wkl_dict[key] = workload_list[self._node_map[key]]
                sch_list = sch_dict[self._wkl_dict[key]]
                current_sch_list = []
                for j in range(min(self._max_sch_num, len(sch_list))):
                    current_sch_list.append(dict(sch_list[j]))
                    current_sch_list[-1]["schedule"] = current_sch_list[-1]["schedule"]
                self._sch_dict[key] = current_sch_list
            else:
                leftmost_node = get_real_node(self._in_nodes_dict, self._node_list,
                                              self._in_nodes_dict[key][0], target_op)
                self._wkl_dict[key] = workload_list[self._node_map[leftmost_node]]
                sch_list = sch_dict[self._wkl_dict[key]]
                current_sch_list = []
                for j in range(min(self._max_sch_num, len(sch_list))):
                    current_sch_list.append(dict(sch_list[j]))
                    current_sch_list[-1]["schedule"] = current_sch_list[-1]["schedule"]
                self._sch_dict[key] = current_sch_list

        # Record shape of elem-like nodes
        shape_list = g_dict['attrs']['shape'][1]
        node_ptr_map = g_dict["node_row_ptr"]
        for key, _ in self._in_nodes_dict.items():
            if self._node_list[key]["op"] != target_op:
                self._elemlike_shape_dict[key] = shape_list[node_ptr_map[key]]

        self._global_data_dict = {
            "wkl_dict": self._wkl_dict, "sch_dict": self._sch_dict,
            "elemlike_shape_dict": self._elemlike_shape_dict,
            "counted_nodes_set": self._counted_nodes_set,
            "stage_dict": self._stage_dict, "in_nodes_dict": self._in_nodes_dict,
            "out_nodes_dict": self._out_nodes_dict, "dep_dict": self._dep_dict,
            "node_list": self._node_list, "input_shapes": self._input_shapes,
            "infer_layout_shape_func": self._infer_layout_shape_func
        }

    def benchmark_layout_transform(self, target="llvm", dtype="float32", min_exec_num=100,
                                   rpc_mode=RPCMode.local.value, rpc_hosts=None, rpc_ports=None,
                                   random_low=0, random_high=1, export_lib_format=".o"):
        """Benchmark all possible layout transformation in the graph,
        given a set of schedule candidates for each workload of target operator.

        Parameters
        ----------
        target : str, optional
            Build target.

        dtype : str, optional
            Data type.

        min_exec_num : int, optional
            Minimum number of execution. Final execution time is the average of
            all execution time.

        rpc_mode : int, optional
            RPC mode. 0 represents local mode. 1 represents rpc host mode.
            2 represents rpc tracker mode. Currently only 0 and 1 are supported.

        rpc_hosts : list of str, optional
            List of rpc hosts for rpc host mode.

        rpc_ports : list of int, optional
            List of rpc ports for rpc host mode.

        random_low : int, optional
            Lower limit for random generated input data.

        random_high : int, optional
            Higher limit for random generated input data.

        export_lib_format : str, optional
            Remote lib format. Currently ".o", ".so"
            and ".tar" are supported.
        """
        node_anc_dict = get_in_nodes(self._graph, self._target_op, self._input_shapes.keys())
        g_dict = json.loads(self._graph.json())
        node_list = g_dict["nodes"]
        batch_size = list(self._input_shapes.values())[0][0]

        layout_transform_key_set = set()
        layout_transform_key_list = []
        param_list, input_shape_list = [], []
        for key, val in node_anc_dict.items():
            node = node_list[key]
            for i, item in enumerate(val):
                if is_input_node(node_list, self._input_shapes.keys(), item):
                    continue

                if node["op"] == self._target_op:
                    c_idx = get_real_node(node_anc_dict, node_list, item, self._target_op)
                    t_idx = key
                    if is_input_node(
                            node_list, self._input_shapes.keys(), c_idx
                    ) or is_input_node(node_list, self._input_shapes.keys(), t_idx):
                        continue
                    wkl_c = self._wkl_dict[c_idx]
                    sch_current_list = self._sch_dict[c_idx]
                    sch_current = [sch_current_list[j]["schedule"]
                                   for j in range(min(self._max_sch_num, len(sch_current_list)))]
                    wkl_t = self._wkl_dict[t_idx]
                    sch_target_list = self._sch_dict[t_idx]
                    sch_target = [sch_target_list[j]["schedule"]
                                  for j in range(min(self._max_sch_num, len(sch_target_list)))]
                elif i == 0:
                    continue
                else:
                    c_idx = get_real_node(node_anc_dict, node_list, item, self._target_op)
                    t_idx = get_real_node(node_anc_dict, node_list, val[0], self._target_op)
                    wkl_c = self._wkl_dict[c_idx]
                    sch_current_list = self._sch_dict[c_idx]
                    sch_current = [sch_current_list[j]["schedule"]
                                   for j in range(min(self._max_sch_num, len(sch_current_list)))]
                    wkl_t = self._wkl_dict[t_idx]
                    sch_target_list = self._sch_dict[t_idx]
                    sch_target = [sch_target_list[j]["schedule"]
                                  for j in range(min(self._max_sch_num, len(sch_target_list)))]

                for sch_c in sch_current:
                    for sch_t in sch_target:
                        if is_elemlike_op(node):
                            in_shape, out_shape, is_valid = self._infer_layout_shape_func(
                                wkl_c, sch_c, sch_t, batch_size, True,
                                self._elemlike_shape_dict[key])
                        else:
                            in_shape, out_shape, is_valid = self._infer_layout_shape_func(
                                wkl_t, sch_c, sch_t, batch_size)
                        layout_transform_key = str((in_shape, out_shape))
                        if is_valid and layout_transform_key not in layout_transform_key_set:
                            layout_transform_key_set.add(layout_transform_key)
                            if in_shape == out_shape:
                                self._layout_transform_dict[layout_transform_key] = 0.0
                            else:
                                layout_transform_key_list.append(layout_transform_key)
                                in_layout = shape2layout(in_shape, self._data_layout)
                                out_layout = shape2layout(out_shape, self._data_layout)
                                params = {"src_layout": in_layout, "dst_layout": out_layout}
                                input_shapes = {"data": in_shape}
                                param_list.append(params)
                                input_shape_list.append(input_shapes)


        layout_transform_executor = LayoutTransformExecutor(
            schedule_dict={}, target=target, input_dtype=dtype,
            min_exec_num=min_exec_num, verbose=self._verbose,
            rpc_mode=rpc_mode, rpc_hosts=rpc_hosts,
            rpc_ports=rpc_ports, export_lib_format=export_lib_format,
            file_logger=self._file_logger,
            console_logger=self._console_logger,
            log_file=self._log_file, log_level=self._log_level)
        start_msg = "Start to benchmark layout transformation..."
        log_msg(start_msg, self._file_logger, self._console_logger, verbose=self._verbose)
        layout_transform_time_list = layout_transform_executor.parameter_execute(
            param_list, input_shape_list, random_low=random_low, random_high=random_high)
        end_msg = "Finished benchmarking layout transformation. " \
                  "%d possible layout transformation tested." % (len(layout_transform_time_list))
        log_msg(end_msg, self._file_logger, self._console_logger, verbose=self._verbose)
        for layout_transform_key, layout_transform_time in zip(layout_transform_key_list,
                                                               layout_transform_time_list):
            self._layout_transform_dict[layout_transform_key] = layout_transform_time
        self._global_data_dict["layout_time_dict"] = self._layout_transform_dict


    def get_optimal_schedules(self):
        """Convert optimal schedule dictionary to a list of schedules
        with ascending order of node index in graph.

        Returns
        -------
        sch_list : list of namedtuple
            List of schedules with ascending order of node index in graph.
        """
        sch_list = []
        for key, val in self._optimal_sch_dict.items():
            if not is_elemlike_op(self._node_list[key]):
                sch_list.append((key, val))
        ordered_sch_list = sorted(sch_list, key=lambda x: x[0])
        ret = []
        for item in ordered_sch_list:
            node_idx = item[0]
            sch_idx = item[1]
            sch = self._sch_dict[node_idx][sch_idx]["schedule"]
            ret.append(sch)
        return ret

    @abstractmethod
    def run(self, **kwargs):
        """Run graph tuning."""
        pass
