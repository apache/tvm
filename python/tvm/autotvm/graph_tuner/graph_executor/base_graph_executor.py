# pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-instance-attributes,too-many-branches,too-many-nested-blocks
"""Base class for graph tuner."""
import logging
import json

from abc import abstractmethod

import tvm
import topi

from nnvm.compiler import graph_attr
from tvm import autotvm
from tvm.autotvm.task import get_config
from tvm.autotvm.task.nnvm_integration import deserialize_args
from tvm.autotvm.record import encode, load_from_file
from ..utils import get_real_node, is_input_node, shape2layout, \
    get_in_nodes, get_out_nodes, is_elemlike_op, get_wkl_map

class BaseGraphExecutor(object):
    """Class to search schedules considering both kernel execution time and
    layout transformation time.

    Before creating a Graph Executor instance, schedule candidates for all kernels in
    graph should be provided through tensor searching.

    TODO Develop more effective approximation/learning algorithm.
    """
    def __init__(self, graph, input_shapes, records, graph_workload_list, target_op,
                 data_layout, layout_related_fields, infer_layout_shape_func,
                 max_sch_num=20, verbose=True, log_file="graph_tuner.log",
                 log_level=logging.DEBUG, name="graph_tuner"):
        """Create a GlobalTuner instance. Local schedule searching for all nodes with
        target_op in the input graph and layout transformation benchmark need to be
        executed before initialization.

        graph : nnvm Graph
            Input graph

        input_shapes : dict of str to tuple.
            Input shapes of graph

        records : str or iterator of (MeasureInput, MeasureResult)
            Collection of kernel level tuning records.
            If it is str, then it should be the filename of a records log file.
                       Each row of this file is an encoded record pair.
            Otherwise, it is an iterator.

        graph_workload_list : list of tuple
            List contains all workloads of target_op in the input graph. The order
            of workloads should be the ascending order of node index. For conv2d_NCHWc,
            conversion from conv2d workload is required and get_conv2d_NCHWc_AVX_workload
            is provided as built-in function to deal with this. Make sure the workload
            format is consistent with the workload format in records.

        layout_time_dict : dict of string to float:
            Dictionary for layout transformation time. Key should be of format
            "(input_shape, output_shape)".

        layout_related_fields : tuple of str
             Fields name in schedule configuration which are related to data I/O layout.
             For example, the data layout for conv2d of intel avx is NCWHc. "tile_ic" and
             "tile_oc" are related to data layout.

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
        self._records = records
        self._infer_layout_shape_func = infer_layout_shape_func
        self._graph_workload_list = graph_workload_list

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
        sch_dict = self._records2dict(layout_related_fields)
        workload_list = list(sch_dict.keys())
        self._node_map = get_wkl_map(graph, workload_list, target_op, graph_workload_list)

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

    def _records2dict(self, layout_related_fields):
        sch_dict = {}
        sch_record_dict = {}
        if isinstance(self._records, str):
            records = load_from_file(self._records)
        else:
            records = self._records


        # Remove unnecessary schedules w.r.t layout_related_fields
        # For a set of schedules which generate the same layout, only
        # the fastest one needs to be preserved.
        for in_measure, out_measure in records:
            workload = in_measure.task.workload
            schedule = in_measure.config
            exec_time = out_measure.costs[0]
            if workload not in sch_record_dict:
                sch_record_dict[workload] = {}
            sch_record = []
            for field_name in schedule:
                field_value = schedule[field_name]
                if field_name in layout_related_fields:
                    sch_record.append(field_value)
            sch_record = tuple(sch_record)
            if sch_record not in sch_record_dict[workload] or \
                    exec_time < sch_record_dict[workload][sch_record][1]:
                sch_record_dict[workload][sch_record] = (schedule, exec_time)

        # Generate final schedule dictionary.
        for wkl, sch_dict_val in sch_record_dict.items():
            sch_dict[wkl] = []
            for sch, exec_time in sch_dict_val.values():
                sch_dict[wkl].append({"schedule": sch, "time": exec_time})
            sch_dict[wkl] = sorted(sch_dict[wkl], key=lambda item: item["time"])

        return sch_dict

    def benchmark_layout_transform(self, target="llvm", dtype="float32", min_exec_num=100,
                                   use_rpc=False, device_key=None, host="localhost", port=9190,
                                   n_parallel=1, do_fork=True, build_func='default'):
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

        use_rpc : boolean, optional
            Whether to use rpc mode for benchmarking.

        device_key : str, optional
            Remote device key which can be queried by
            python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190

        host : str, optional
            IP address used to create RPC tracker on host machine.

        port : int, optional
            Port number used to create RPC tracker on host machine.
        n_parallel: int, optional
            The number of measurement task that can run in parallel.
            Set this according to the number of cpu cores (for compilation) and
            the number of devices you have (for measuring generate code).
        do_fork: bool, optional
            Whether use multiprocessing (based on fork) for running measure jobs in parallel.
            Set this to False if you want to debug (see trackback) or using fork is not suitable.
            NOTE: If this is False, parallel and timeout do not work.
        build_func: str or callable, optional
            'default': call default builder. This works for normal target (llvm, cuda)

            'ndk': use Android NDK to create shared library. Use this for android target.

            callable: customized build function for other backends (e.g. VTA).
                      See autotvm/measure/measure_methods.py::default_build_func for example.
        """
        node_anc_dict = get_in_nodes(self._graph, self._target_op, self._input_shapes.keys())
        g_dict = json.loads(self._graph.json())
        node_list = g_dict["nodes"]
        batch_size = list(self._input_shapes.values())[0][0]

        args_list = []
        layout_transform_key_list = []
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
                                wkl_c, sch_c, sch_t, self._elemlike_shape_dict[key])
                        else:
                            in_shape, out_shape, is_valid = self._infer_layout_shape_func(
                                wkl_t, sch_c, sch_t)
                        layout_transform_key = str((in_shape, out_shape))
                        if is_valid:
                            if in_shape == out_shape:
                                self._layout_transform_dict[layout_transform_key] = 0.0
                            else:
                                in_layout = shape2layout(in_shape, self._data_layout)
                                out_layout = shape2layout(out_shape, self._data_layout)
                                data_placeholder = tvm.placeholder(in_shape, name="data",
                                                                   dtype=dtype)
                                args = [data_placeholder, in_layout, out_layout, out_shape,
                                        "layout_transform", "injective"]
                                args_list.append(args)
                                layout_transform_key_list.append(layout_transform_key)

        @autotvm.template
        def layout_transform(*args):
            args = deserialize_args(args)
            cfg = get_config()
            cfg.add_flop(-1)
            A = args[0]
            C = topi.cpp.nn.layout_transform(*args)
            s = topi.generic.schedule_injective([C])
            return s, [A, C]

        def log_to_list(record_list):
            def _callback(_, inputs, results):
                """Callback implementation"""
                record_list.append(results)
            return _callback

        builder=autotvm.LocalBuilder(n_parallel=1, build_func=build_func)
        runner = autotvm.LocalRunner(number=min_exec_num, repeat=1)
        if use_rpc:
            if device_key is None:
                raise RuntimeError("device_key need to be set to use rpc tracker mode.")
            runner = autotvm.measure.RPCRunner(device_key, host, port, n_parallel=n_parallel, 
                                               number=min_exec_num, repeat=1)
        measure_option = autotvm.measure_option(builder=builder, runner=runner)
        for args, layout_transform_key in zip(args_list, layout_transform_key_list):
            if layout_transform_key in self._layout_transform_dict:
                continue
            records = []
            task = autotvm.task.create(layout_transform, args=args, target=target)
            tuner = autotvm.tuner.GridSearchTuner(task)
            tuner.tune(n_trial=1, measure_option=measure_option,
                       callbacks=[log_to_list(records)])
            exec_time = records[0][0].costs[0]
            self._layout_transform_dict[layout_transform_key] = exec_time

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

    def write_opt_sch2record_file(self, record_file="graph_opt_schedule.log"):
        if isinstance(self._records, str):
            records = load_from_file(self._records)
        else:
            records = self._records

        # Create dict from (workload, schedule) to record
        record_dict = {}
        for record in records:
            in_measure = record[0]
            workload = in_measure.task.workload
            schedule = in_measure.config
            record_dict[str((workload, schedule))] = record

        with open(record_file, "a") as of:
            for workload, schedule in zip(self._graph_workload_list,
                                          self.get_optimal_schedules()):
                record = record_dict[str((workload, schedule))]
                of.write(encode(record[0], record[1]) + "\n")

    @abstractmethod
    def run(self, **kwargs):
        """Run graph tuning."""
        pass
