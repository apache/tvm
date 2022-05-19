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
# pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-instance-attributes,too-many-branches,too-many-nested-blocks,invalid-name,unused-argument,unused-variable,no-member,no-value-for-parameter
"""Base class for graph tuner."""
import logging
from abc import abstractmethod

import numpy as np
from tvm import topi

import tvm
from tvm import te
from tvm import autotvm, relay
from tvm.autotvm.task import get_config
from tvm.autotvm.record import encode, load_from_file
from tvm.autotvm.measure import MeasureResult, MeasureInput
from tvm.target import Target

from ...target import Target
from .utils import (
    is_boundary_node,
    get_in_nodes,
    get_out_nodes,
    has_multiple_inputs,
    bind_inputs,
    expr2graph,
)
from ._base import INVALID_LAYOUT_TIME

from ._base import OPT_OUT_OP


def get_infer_layout(task_name):
    if task_name.startswith("conv2d"):
        return topi.nn.conv2d_infer_layout
    if task_name.startswith("depthwise_conv2d"):
        return topi.nn.depthwise_conv2d_infer_layout
    raise ValueError("Cannot find infer layout for task %s" % task_name)


@autotvm.template("layout_transform")
def layout_transform(*args):
    """Autotvm layout transform template."""
    cfg = get_config()
    cfg.add_flop(-1)
    data = args[0]
    out = topi.layout_transform(*args)
    sch = topi.generic.schedule_injective([out])
    return sch, [data, out]


class BaseGraphTuner(object):
    """Class to search schedules considering both kernel execution time and
    layout transformation time.

    Before creating a Graph Executor instance, schedule candidates for all kernels in
    graph should be provided through tensor-level tuning.
    """

    def __init__(
        self,
        graph,
        input_shapes,
        records,
        target_ops,
        target,
        max_sch_num=20,
        dtype="float32",
        verbose=True,
        log_file="graph_tuner.log",
        log_level=logging.DEBUG,
        name="graph_tuner",
    ):
        """Create a GlobalTuner instance. Local schedule searching for all nodes with
        target_op in the input graph and layout transformation benchmark need to be
        executed before initialization.

        graph : tvm.relay.function.Function
            Input graph

        input_shapes : dict of str to tuple.
            Input shapes of graph

        records : str or iterator of (MeasureInput, MeasureResult)
            Collection of kernel level tuning records.
            If it is str, then it should be the filename of a records log file.
                       Each row of this file is an encoded record pair.
            Otherwise, it is an iterator.

        target_ops : List of tvm.ir.Op
            Target tuning operators.

        target : str or tvm.target
            Compilation target.

        max_sch_num : int, optional
            Maximum number of schedule candidates for each workload.

        dtype : str, optional
            Data type.

        log_file : str, optional
            graph tuner log file name

        name : str, optional
            Name of global tuner.
        """
        self._node_list = []
        self._layout_transform_perf_records = {}
        self._layout_transform_interlayer_cost = {}
        self._input_shapes = input_shapes
        self._target_ops = target_ops

        self._name = name
        self._max_sch_num = max_sch_num
        self._optimal_sch_dict = {}
        self._records = records
        self._dtype = dtype
        if isinstance(target, str):
            target = Target(target)
        self._target = target
        self._optimal_record_dict = {}

        # Set up logger
        self._verbose = verbose
        self._logger = logging.getLogger(name + "_logger")
        need_file_handler = need_console_handler = True
        for handler in self._logger.handlers:
            if handler.__class__.__name__ == "FileHandler":
                need_file_handler = False
            if handler.__class__.__name__ == "StreamHandler":
                need_console_handler = False
        self._log_level = log_level
        self._log_file = log_file
        self._formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        self._logger.setLevel(log_level)
        if need_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._formatter)
            self._logger.addHandler(file_handler)
        if self._verbose and need_console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._formatter)
            self._logger.addHandler(console_handler)
            self._logger.setLevel(log_level)
            self._logger.propagate = False

        # Generate workload and schedule dictionaries.
        if isinstance(graph, tvm.IRModule):
            graph = graph["main"]

        if isinstance(graph, relay.function.Function):
            node_dict = {}
            graph = bind_inputs(graph, input_shapes, dtype)
            expr2graph(graph, self._target_ops, node_dict, self._node_list, target)
        else:
            raise RuntimeError("Unsupported graph type: %s" % str(type(graph)))

        self._graph = graph
        self._in_nodes_dict = get_in_nodes(self._node_list, self._target_ops, input_shapes.keys())
        if len(self._in_nodes_dict) == 0:
            raise RuntimeError(
                "Could not find any input nodes with whose "
                "operator is one of %s" % self._target_ops
            )
        self._out_nodes_dict = get_out_nodes(self._in_nodes_dict)
        self._fetch_cfg()
        self._opt_out_op = OPT_OUT_OP

        # Setup infer_layout for elemwise-like nodes
        # Note: graph tuner currently only supports tuning of single input and single output
        # op as target op, such as conv2d, dense and conv2d_transpose. In this case, we can
        # reuse infer_layout function from target ops for elemwise-like nodes. The behavior
        # is to modify the first tensor shape of input workload to the output shape of
        # elemwise-like node, and use infer_layout function from input op to generate layouts.
        input_names = self._input_shapes.keys()
        for idx in sorted(self._in_nodes_dict.keys()):
            if has_multiple_inputs(self._node_list, idx, input_names, self._opt_out_op):
                node_entry = self._node_list[idx]
                node_entry["topi_op"] = []
                node_entry["workloads"] = []
                for input_idx in self._in_nodes_dict[idx]:
                    input_node = self._node_list[input_idx]
                    if not is_boundary_node(input_node, input_names):
                        input_topi_op = input_node["topi_op"][0]
                        node_entry["topi_op"].append(input_topi_op)
                        # Only replace the first input tensor
                        input_workload = input_node["workloads"][0]
                        first_tensor = input_workload[1]
                        dtype = first_tensor[-1]
                        new_shape = tuple([val.value for val in node_entry["types"][0].shape])
                        actual_workload = (
                            (input_workload[0],)
                            + (("TENSOR", new_shape, dtype),)
                            + input_workload[2:]
                        )
                        node_entry["workloads"].append(actual_workload)
                        if "record_candidates" not in node_entry:
                            node_entry["record_candidates"] = input_node["record_candidates"]
                    else:
                        node_entry["topi_op"].append(None)
                        node_entry["workloads"].append(None)

    def _fetch_cfg(self):
        """Read and pre-process input schedules."""
        if isinstance(self._records, str):
            records = load_from_file(self._records)
        else:
            records = self._records
        cfg_dict = {}
        for record in records:
            in_measure, _ = record
            workload = in_measure.task.workload
            if workload not in cfg_dict:
                cfg_dict[workload] = []
            cfg_dict[workload].append(record)

        cache_dict = {}
        for key in self._in_nodes_dict:
            node_entry = self._node_list[key]
            if node_entry["op"] not in self._target_ops:
                continue
            workload = node_entry["workloads"][0]
            if workload in cache_dict:
                node_entry["record_candidates"] = cache_dict[workload]
                continue
            record_candidates = []
            infer_layout_func = get_infer_layout(node_entry["topi_op"][0])
            layout_tracking_dict = {}
            for record in cfg_dict[workload]:
                in_measure, out_measure = record
                workload = in_measure.task.workload
                cfg = in_measure.config
                # For multiple cfgs which produces the same in/out layouts,
                # only the most efficient one is preserved.
                with self._target:
                    layouts = infer_layout_func(workload, cfg)
                    if layouts in layout_tracking_dict:
                        cost = out_measure.costs[0]
                        current_best_cost = layout_tracking_dict[layouts][1].costs[0]
                        if cost < current_best_cost:
                            layout_tracking_dict[layouts] = record
                    else:
                        layout_tracking_dict[layouts] = record
            sorted_records = sorted(
                layout_tracking_dict.values(), key=lambda item: item[1].costs[0]
            )
            for i in range(min(self._max_sch_num, len(sorted_records))):
                record_candidates.append(sorted_records[i])
            node_entry["record_candidates"] = record_candidates
            cache_dict[workload] = record_candidates

    def _iterate_layout_transform(self, callback):
        """Iterate all possible layout transformations and execute callback for each
        iteration. callback function accepts 6 arguments: from_node_idx, to_node_idx,
        from_sch_idx, to_sch_idx, args which represent the argument list of layout
        transformation and is_valid showing whether this is a valid layout transformation.
        """
        input_names = self._input_shapes.keys()
        pair_tracker = set()
        for key, val in self._in_nodes_dict.items():
            node_entry = self._node_list[key]
            target_input_idx = -1
            target_input_pos = -1
            if has_multiple_inputs(self._node_list, key, input_names, self._opt_out_op):
                for i, item in enumerate(val):
                    node = self._node_list[item]
                    if not is_boundary_node(node, input_names):
                        target_input_idx = item
                        target_input_pos = i
                        break

            for i, item in enumerate(val):
                i_idx = item
                in_node_entry = self._node_list[i_idx]
                if is_boundary_node(in_node_entry, input_names):
                    continue

                if node_entry["op"] in self._target_ops:
                    o_idx = key
                    o_infer_layout_func = get_infer_layout(node_entry["topi_op"][0])
                    o_wkl = node_entry["workloads"][0]
                    i_topi_op = in_node_entry["topi_op"][0]
                    i_wkl = in_node_entry["workloads"][0]
                    pivot = 0
                    while not i_wkl:
                        pivot += 1
                        i_topi_op = in_node_entry["topi_op"][pivot]
                        i_wkl = in_node_entry["workloads"][pivot]
                    i_infer_layout_func = get_infer_layout(i_topi_op)
                else:
                    o_idx = target_input_idx
                    if i <= target_input_pos:
                        continue
                    o_infer_layout_func = get_infer_layout(node_entry["topi_op"][0])
                    o_wkl = node_entry["workloads"][target_input_pos]
                    i_infer_layout_func = get_infer_layout(node_entry["topi_op"][i])
                    i_wkl = node_entry["workloads"][i]

                if (i_idx, o_idx) in pair_tracker:
                    continue
                pair_tracker.add((i_idx, o_idx))

                for m, i_record in enumerate(in_node_entry["record_candidates"]):
                    for n, o_record in enumerate(node_entry["record_candidates"]):
                        i_cfg, o_cfg = i_record[0].config, o_record[0].config
                        with self._target:
                            i_input_info, i_output_info = i_infer_layout_func(i_wkl, i_cfg)
                            o_input_info, o_output_info = o_infer_layout_func(o_wkl, o_cfg)
                        if (
                            len(i_input_info) > 1
                            or len(i_output_info) > 1
                            or len(o_input_info) > 1
                            or len(o_output_info) > 1
                        ):
                            raise RuntimeError(
                                "Graph tuner only supports target operator "
                                "with single input and single output. "
                                "Please check target_ops argument."
                            )

                        in_shape, in_layout = i_output_info[0]
                        if node_entry["op"] in self._target_ops:
                            _, out_layout = o_input_info[0]
                        else:
                            _, out_layout = o_output_info[0]
                        data_placeholder = te.placeholder(in_shape, name="data", dtype=self._dtype)
                        args = [data_placeholder, in_layout, out_layout]
                        callback(i_idx, o_idx, m, n, args)

    def _create_matrix_callback(self, from_node_idx, to_node_idx, from_sch_idx, to_sch_idx, args):
        """Create dictionary containing matrix format of layout transformation
        between nodes."""
        in_layout, out_layout = args[1], args[2]
        ltf_workload = autotvm.task.args_to_workload(args, "layout_transform")
        idx_pair_key = (from_node_idx, to_node_idx)

        if in_layout == out_layout:
            layout_transform_time = 0
        else:
            layout_transform_time = self._layout_transform_perf_records[ltf_workload][1].costs[0]

        if idx_pair_key not in self._layout_transform_interlayer_cost:
            self._layout_transform_interlayer_cost[idx_pair_key] = []
        if len(self._layout_transform_interlayer_cost[idx_pair_key]) <= from_sch_idx:
            self._layout_transform_interlayer_cost[idx_pair_key].append([])
        self._layout_transform_interlayer_cost[idx_pair_key][from_sch_idx].append(
            layout_transform_time
        )

    def benchmark_layout_transform(
        self,
        min_exec_num=100,
        timeout=10,
        use_rpc=False,
        device_key=None,
        host="127.0.0.1",
        port=9190,
        n_parallel=1,
        build_func="default",
        layout_records=None,
        target_host=None,
        infer_layout=False,
        runner=None,
    ):
        """Benchmark all possible layout transformation in the graph,
        given a set of schedule candidates for each workload of target operator.

        Parameters
        ----------
        min_exec_num : int, optional
            Minimum number of execution. Final execution time is the average of
            all execution time.

        timeout : int, optional
            Time out for each execution.

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

        build_func: str or callable, optional
            'default': call default builder. This works for normal target (llvm, cuda)

            'ndk': use Android NDK to create shared library. Use this for android target.

            callable: customized build function for other backends (e.g. VTA).
                      See autotvm/measure/measure_methods.py::default_build_func for example.

        layout_records : str or iterator of (MeasureInput, MeasureResult). optional
            Collection of layout_transform benchmarking records.
            If is str, then it should be the filename of a records log file.
                   Each row of this file is an encoded record pair.
            Otherwise, it is an iterator.

            If this argument is set, graph tuner will first check whether layout_transform
            workload already exists in records and skip benchmarking if possible.

        target_host : str, optional
            str or :any:`tvm.target.Target` optional
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm intepreter is used.

        infer_layout : bool, optional
            Whether to infer layout transformation time if it doesn't exist in records, instead
            of benchmarking on target device.

            This might bring performance loss comparing to benchmarking layout transformation.
        runner : Runner, optional
            Accept a user-supplied runner
        """
        self._logger.info("Start to benchmark layout transformation...")
        self._target, target_host = Target.canon_target_and_host(self._target, target_host)

        if layout_records is None and infer_layout:
            raise RuntimeError("Requires some records to infer layout transformation time.")

        if isinstance(layout_records, str):
            layout_records = load_from_file(layout_records)
            if not layout_records and infer_layout:
                raise RuntimeError("Records must be non-empty to infer layout transformation time.")

        if isinstance(layout_records, str):
            layout_records = load_from_file(layout_records)
        num_flops, total_time = 0, 0
        if layout_records is not None:
            for record in layout_records:
                ltf_wkl = record[0].task.workload
                self._layout_transform_perf_records[ltf_wkl] = record
                input_shape = ltf_wkl[1][1]
                flops = np.prod(input_shape)
                num_flops += flops
                total_time += record[1].costs[0]
        avg_time = total_time / num_flops if num_flops > 0 else 0

        args_list = []

        def _fetch_args_callback(from_node_idx, to_node_idx, from_sch_idx, to_sch_idx, args):
            """Callback function to fetch layout transform args"""
            _, in_layout, out_layout = args
            if in_layout != out_layout:
                args_list.append(args)

        self._iterate_layout_transform(_fetch_args_callback)

        def _log_to_list(record_list):
            """Callback to log result to a list."""

            def _callback(_, inputs, results):
                """Callback implementation"""
                record_list.append((inputs[0], results[0]))

            return _callback

        builder = autotvm.LocalBuilder(n_parallel=n_parallel, build_func=build_func)
        if use_rpc:
            if device_key is None:
                raise RuntimeError("device_key need to be set to use rpc tracker mode.")
            runner = autotvm.measure.RPCRunner(
                device_key,
                host,
                port,
                n_parallel=n_parallel,
                number=min_exec_num,
                repeat=1,
                timeout=timeout,
            )
        elif not runner:
            runner = autotvm.LocalRunner(number=min_exec_num, repeat=1, timeout=timeout)
        measure_option = autotvm.measure_option(builder=builder, runner=runner)
        for args in args_list:
            data, in_layout, out_layout = args
            ltf_workload = autotvm.task.args_to_workload(args, "layout_transform")
            if ltf_workload in self._layout_transform_perf_records:
                continue

            if infer_layout:
                input_shape = ltf_workload[1][1]
                flops = 1
                for i in input_shape:
                    flops *= i

                # Rule out invalid layout transformations
                out = topi.layout_transform(data, in_layout, out_layout)
                out_flops = 1
                for i in topi.utils.get_const_tuple(out.shape):
                    out_flops *= i

                if flops != out_flops:
                    inferred_time = INVALID_LAYOUT_TIME
                else:
                    inferred_time = flops * avg_time

                record_input = MeasureInput(target=self._target, task=None, config=None)
                record_output = MeasureResult(
                    costs=(inferred_time,), error_no=0, all_cost=-1, timestamp=-1
                )
                self._layout_transform_perf_records[ltf_workload] = (record_input, record_output)
                continue

            records = []
            task = autotvm.task.create("layout_transform", args=args, target=self._target)
            tuner = autotvm.tuner.GridSearchTuner(task)
            tuner.tune(n_trial=1, measure_option=measure_option, callbacks=[_log_to_list(records)])
            if not isinstance(records[0][1].costs[0], float):
                records[0] = (records[0][0], records[0][1]._replace(costs=(INVALID_LAYOUT_TIME,)))
            self._layout_transform_perf_records[ltf_workload] = records[0]

        self._iterate_layout_transform(self._create_matrix_callback)
        self._logger.info("Benchmarking layout transformation successful.")

    @property
    def layout_transform_perf_records(self):
        """Get layout transformation dictionary for input graph.

        Returns
        -------
        layout_transform_perf_records : dict of tuple to (MeasureInput, MeasureResult)
            Layout transformation dictionary for input graph.
        """
        return self._layout_transform_perf_records

    def get_optimal_records(self):
        """Convert optimal record dictionary to a list of records
        with ascending order of node index in graph.

        Returns
        -------
        sch_list : list of tuple
            List of records with ascending order of node index in graph.
        """
        ordered_index_list = sorted(self._optimal_record_dict.keys())
        ret = []
        for index in ordered_index_list:
            node_entry = self._node_list[index]
            if node_entry["op"] not in self._target_ops:
                continue
            ret.append(node_entry["record_candidates"][self._optimal_record_dict[index]])
        return ret

    def write_opt_sch2record_file(self, record_file="graph_opt_schedule.log"):
        """Write graph level optimal schedules into file.

        Parameters
        ----------
        record_file : str, optional
            Output schedule file.
        """
        with open(record_file, "a") as out_file:
            records = self.get_optimal_records()
            for record in records:
                out_file.write(encode(record[0], record[1]) + "\n")
        msg = "Writing optimal schedules to %s successfully." % record_file
        self._logger.info(msg)

    @abstractmethod
    def run(self, **kwargs):
        """Run graph tuning."""
