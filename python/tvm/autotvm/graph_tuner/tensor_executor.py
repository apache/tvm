"""API to execute a tensor benchmark job."""

import itertools
import logging
import enum
import numpy as np
import nnvm
import tvm

from abc import abstractmethod
from multiprocessing import Process, Value, Array
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr
from nnvm.testing.init import Xavier
from tvm.contrib import graph_runtime
from topi.nn.conv2d import _get_schedule_NCHWc, _get_alter_layout_schedule
from topi.nn.conv2d import Workload
from topi.x86.conv2d_avx_common import AVXConvCommonFwd
from topi.x86.conv2d_avx_1x1 import AVXConv1x1Fwd
from rpc import run_remote_module
from utils import get_factor

class RPCMode(enum.Enum):
    local = 0
    rpc_host = 1
    rpc_tracker = 2

class BaseTensorExecutor(object):
    def __init__(self, schedule_dict, target="llvm", dtype="float32", verbose=True,
                 rpc_mode=RPCMode.local, rpc_hosts=None, rpc_ports=None, export_lib_format=".o",
                 log_file="tensor_tuning.log", log_level=logging.DEBUG, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self._formatter)
        self._logger.addHandler(file_handler)
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._formatter)
            self._logger.addHandler(console_handler)
        self._logger.setLevel(log_level)

        self._sch_dict = schedule_dict
        self._target = target
        self._dtype = dtype
        self._rpc_mode = rpc_mode
        self._rpc_hosts = rpc_hosts
        self._rpc_ports = rpc_ports
        self._export_lib_format = export_lib_format

        if self._rpc_mode == RPCMode.rpc_host:
            if self._rpc_hosts is None:
                self._log_msg("Number of rpc hosts must be at least 1 for rpc host mode.")
            if self._rpc_ports is None:
                self._log_msg("Number of rpc ports must be at least 1 for rpc host mode.")
            if len(self._rpc_hosts) != len(self._rpc_ports):
                self._log_msg("Number of rpc hosts and ports should be equal. "
                              "Got %d vs %d" % (len(self._rpc_hosts), len(self._rpc_ports)))
        # TODO: support RPC tracker.
        elif self._rpc_mode == RPCMode.rpc_tracker:
            self._log_msg("RPC tracker not supported.")

    @property
    def get_schedule_dict(self):
        return self._sch_dict

    @abstractmethod
    def _get_op_symbol(self):
        pass

    def _get_layout_related_fields(self):
        self._log_msg("Not Implemented.")

    def _workload2params(self, workload, schedule):
        self._log_msg("Not Implemented.")

    def _workload2ishapes(self, workload, schedule):
        self._log_msg("Not Implemented.")

    def _load_schedule(self, schedule):
        self._log_msg("Not Implemented.")

    def _create_search_space(self, workload):
        self._log_msg("Not Implemented. You need to either pass a "
                      "search_space_list when calling tensor_exec "
                      "or override this method.")

    def _log_err_msg(self, msg):
        self._logger.error(msg)
        raise RuntimeError(msg)

    def _get_real_exec_number(self, trial_exec_time):
        """Get the actual number of running times
        given the average execution time.
        """
        real_exec_number = -1
        if 1 < trial_exec_time <= 10:
            real_exec_number = 300
        elif 0.2 < trial_exec_time <= 1:
            real_exec_number = 1000
        elif trial_exec_time <= 0.2:
            real_exec_number = 10000
        return real_exec_number

    def _search_space2schedules(self, search_space_dict):
        if "schedule_template_name" not in search_space_dict:
            self._log_err_msg("schedule_template_name key not found in seach space dictionary."
                              "Please add name of the namedtuple representing "
                              "schedule template.")
        sch_template = search_space_dict["sch_template_name"]
        sch_fields_list = sch_template._fields
        search_candidates_list = []
        sch_list = []
        for sch_field in sch_fields_list:
            search_candidates_list.append(search_space_dict[sch_field])
        for search_entry in itertools.product(*search_candidates_list):
            sch_list.append(sch_template._make(search_entry))
        return sch_list

    def _atomic_exec(self, net, input_shapes, sch=None, random_low=0, random_high=1,
                     min_exec_number=10, rpc_mode=RPCMode.local, rpc_host="localhost",
                     rpc_port="9090", extra_compile_params=None):
        if extra_compile_params is None:
            extra_compile_params = {}
        input_data_dict = {}
        for name, shape in input_shapes.items():
            input_data_dict[name] = np.random.uniform(random_low, random_high,
                                                      size=shape).astype(self._dtype)
        # Random initialize weights
        g = nnvm.graph.create(net)
        g = graph_attr.set_shape_inputs(g, input_shapes)
        g = g.apply("InferShape")
        shape = g.json_attr("shape")
        index = g.index
        shape_dict = {x: shape[index.entry_id(x)] for x in index.input_names}
        initializer = Xavier()
        params = {}
        for k, v in shape_dict.items():
            if k == "data":
                continue
            init_value = np.zeros(v).astype(self._dtype)
            initializer(k, init_value)
            params[k] = tvm.nd.array(init_value, ctx=tvm.cpu())

        if sch is not None:
            self._load_schedule(sch)
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(net, self._target, shape=input_shapes,
                                                     params=params,
                                                     **extra_compile_params)

        if rpc_mode == RPCMode.local:
            ctx = tvm.context(self._target)
            module = graph_runtime.create(graph, lib, ctx)
            module.set_input(**params)
            module.set_input(**input_data_dict)
            time_eval = module.module.time_evaluator("run", ctx, number=min_exec_number)
            exec_time = time_eval().mean
            # Run more times to get stable data if necessary
            real_exec_number = self._get_real_exec_number(exec_time * 1000)
            if real_exec_number > min_exec_number:
                time_eval = module.module.time_evaluator("run", ctx, number=real_exec_number)
                exec_time = time_eval().mean
        elif rpc_mode == RPCMode.rpc_host:
            session = tvm.rpc.connect(rpc_host, rpc_port)
            exec_time = run_remote_module(session, graph, lib, params, input_data_dict,
                                          remote_dev_type=self._target, run_times=min_exec_number,
                                          export_lib_format=self._export_lib_format)
            # Run more times to get stable data if necessary
            real_exec_number = self._get_real_exec_number(exec_time * 1000)
            if real_exec_number > min_exec_number:
                exec_time = run_remote_module(session, graph, lib, params, input_data_dict,
                                              remote_dev_type=self._target, run_times=exec_time,
                                              export_lib_format=self._export_lib_format)
        else:
            self._log_err_msg("RPC tracker not supported.")

        return exec_time * 1000

    def _process_wrapper(self, *args, **kwargs):
        ret_val = kwargs["ret_val"]
        exec_kwargs = dict(kwargs)
        del exec_kwargs["ret_val"]
        exec_time = self._atomic_exec(*args, **exec_kwargs)
        ret_val.value = exec_time

    def parameter_execute(self, param_list, input_shape_list, sch_list=None,
                          extra_compile_params=None, random_low=0, random_high=1,
                          log_info=True):
        def _launch_atomic_job(proc_param, proc_input_shapes, proc_ret_val, **kwargs):
            input_var_list = [sym.Variable(name=input_name) for input_name in proc_input_shapes.keys()]
            op_sym = self._get_op_symbol()
            net = op_sym(*input_var_list, **proc_param)
            args = [net, input_shapes]
            kwargs["ret_val"] = proc_ret_val
            new_proc = Process(target=self._process_wrapper, args=args, kwargs=kwargs)
            new_proc.start()
            new_proc.join()

        if len(param_list) != len(input_shape_list):
            self._log_err_msg("Length of parameter list and input shape list must be equal: "
                              "%d vs %d." % (len(param_list), len(input_shape_list)))

        ret_list = []
        if self._rpc_mode == RPCMode.local:
            ret_val = Value("f", 0.0)
            for i, item in enumerate(zip(param_list, input_shape_list)):
                param, input_shapes = item
                sch = sch_list[i] if sch_list else None
                _launch_atomic_job(param, input_shapes, ret_val, sch=sch,
                                   random_low=random_low, random_high=random_high,
                                   extra_compile_params=extra_compile_params)
                ret_list.append(ret_val.value)
        elif self._rpc_mode == RPCMode.rpc_host:
            def _launch_group_jobs(proc_param_list, proc_input_shape_list, proc_sch_list, proc_ret_array, **kwargs):
                for i, item in enumerate(zip(proc_param_list, proc_input_shape_list,
                                             proc_sch_list)):
                    proc_param, proc_input_shapes, proc_sch = item
                    atomic_ret_value = Value("f", 0.0)
                    _launch_atomic_job(proc_param, proc_input_shapes, atomic_ret_value, **kwargs)
                    proc_ret_array[i] = atomic_ret_value.value

            proc_pool = []
            num_jobs = len(param_list)
            num_proc = len(self._rpc_hosts)
            num_jobs_per_proc = (num_jobs - num_jobs % num_proc) // num_proc
            proc_ret_list = []

            end_idx = 0
            for i, item in enumerate(zip(self._rpc_hosts, self._rpc_ports)):
                sch = sch_list[i] if sch_list else None
                rpc_host, rpc_port = item
                start_idx = end_idx
                end_idx = min(num_jobs, start_idx + num_jobs_per_proc)
                ret_array = Array("f", range(end_idx - start_idx))
                proc_ret_list.append(ret_array)
                current_param_list = [param_list[j] for j in range(start_idx ,end_idx)]
                current_input_shape_list = [input_shape_list[j] for j in range(start_idx ,end_idx)]
                current_sch_list = [sch_list[j] if sch_list else None for j in range(start_idx ,end_idx)]
                args = [current_param_list, current_input_shape_list, current_sch_list,
                        ret_array]
                kwargs = {"rpc_host": rpc_host, "rpc_port": rpc_port,
                          "random_low": random_low, "random_high": random_high,
                          "extra_compile_params": extra_compile_params}
                proc = Process(target=_launch_group_jobs, args=args, kwargs=kwargs)
                proc_pool.append(proc)
                proc.start()

            for proc in proc_pool:
                proc.join()

            for proc_ret_array in proc_ret_list:
                for val in proc_ret_array:
                    ret_list.append(val)
        else:
            self._log_err_msg("RPC tracker not supported.")

        if log_info:
            for i, item in enumerate(zip(param_list, input_shape_list, ret_list)):
                params, input_shapes, exec_time = item
                self._logger.info("Execution time for operator %s with input_shapes=%s, "
                                  "parameters=%s, schedule=%s: %f ms."
                                  % (self._get_op_symbol().__name__, str(input_shapes),
                                     str(params), str(sch_list[i] if sch_list else None),
                                     exec_time))
        return ret_list

    def workload_execute(self, wkl_list, search_space_list=None,
                         extra_compile_params=None, random_low=0, random_high=1,
                         force_search=False):
        if search_space_list is None:
            search_space_list = []
            for wkl in wkl_list:
                search_space_list.append(self._create_search_space(wkl))

        for wkl, search_space_dict in zip(wkl_list, search_space_list):
            if wkl in self._sch_dict and not force_search:
                continue
            param_list, input_shape_list= [], []
            sch_list = self._search_space2schedules(search_space_dict)
            sch_template = search_space_dict["schedule_template_name"]
            for sch in sch_list:
                param_list.append(self._workload2params(wkl, sch))
                input_shape_list.append(self._workload2ishapes(wkl, sch))
            exec_time_list = self.parameter_execute(param_list, input_shape_list,
                                                    sch_list=sch_list,
                                                    extra_compile_params=extra_compile_params,
                                                    random_low=random_low, random_high=random_high,
                                                    log_info=False)

            # Order schedule by execution time and only preserve
            # the fastest schedule in schedules generating the same
            # data layout.
            sch_record_dict = {}
            for sch, exec_time in zip(sch_list, exec_time_list):
                self._logger.info("Execution time for operator %s with workload=%s, "
                                  "schedule=%s: %f ms." % (self._get_op_symbol().__name__,
                                                           str(wkl), sch, exec_time))
                layout_related_fields = self._get_layout_related_fields()
                sch_record_key = str((getattr(sch_template, field)
                                      for field in layout_related_fields))
                if sch_record_key not in sch_record_dict \
                        or exec_time < sch_record_dict[sch_record_key][1]:
                    sch_record_dict[sch_record_key] = (sch, exec_time)

            self._sch_dict[wkl] = []
            for sch, exec_time in sch_record_dict.values():
                self._sch_dict[wkl].append({"schedule": sch, "time": exec_time})
            self._sch_dict[wkl] = sorted(self._sch_dict[wkl], key=lambda item: item["time"])


class Conv2dAVXExecutor(BaseTensorExecutor):
    def _get_op_symbol(self):
        return sym.contrib.conv2d_NCHWc

    def _workload2params(self, workload, schedule):
        ic_bn = schedule.ic_bn
        oc_bn = schedule.oc_bn
        is_unit_kernel = workload.hkernel == 1 and workload.wkernel == 1
        data_layout = "NCHW%dc" % ic_bn
        kernel_layout = "OIHW%di%do" % (ic_bn, oc_bn) \
            if not is_unit_kernel else "OI%di%doHW" % (ic_bn, oc_bn)
        out_layout = "NCHW%dc" % oc_bn
        param_dict = {"channels": workload.out_filter,
                      "kernel_size": (workload.hkernel, workload.wkernel),
                      "padding": (workload.hpad, workload.wpad),
                      "strides": (workload.hstride, workload.wstride),
                      "layout": data_layout, "out_layout": out_layout,
                      "kernel_layout": kernel_layout}
        return param_dict

    def _workload2ishapes(self, workload, schedule):
        batch_size = 1
        in_channel = workload.in_filter
        in_height = workload.height
        in_width = workload.width
        ic_bn = schedule.ic_bn
        data_shape = (batch_size, in_channel // ic_bn, in_height,
                      in_width, ic_bn)
        return data_shape

    def _load_schedule(self, schedule):
        @_get_schedule_NCHWc.register("cpu", override=True)
        def _get_schedule_NCHWc_x86(wkl, layout, out_layout):
            return schedule

        @_get_alter_layout_schedule.register("cpu", override=True)
        def _get_alter_layout_schedule_x86(wkl):
            return schedule

    def _create_search_space(self, workload):
        ih, iw = workload.height, workload.width
        ic, oc = workload.in_filter, workload.out_filter
        hk, wk = workload.hkernel, workload.wkernel
        hp, wp = workload.hpad, workload.wpad
        hs, ws = workload.hstride, workload.wstride
        oh = (ih - hk + 2 * hp) // hs + 1
        ow = (iw - wk + 2 * wp) // ws + 1
        ic_bn = get_factor(ic)
        oc_bn = get_factor(oc)
        ow_bn = get_factor(ow)
        ow_bn_max = 64
        tmp = []
        for ow_bn_candidate in ow_bn:
            if ow_bn_candidate <= ow_bn_max:
               tmp.append(ow_bn_candidate)
        ow_bn = tmp
        if len(ow_bn) > 2:
            ow_bn.remove(1)
        oh_bn = [1, 2] if oh > 1 else [1]
        unroll_kw = [True, False]
        is_unit_kernel = hk == 1 and wk == 1
        search_space_dict = {}
        if is_unit_kernel:
            search_space_dict["schedule_template_name"] = AVXConv1x1Fwd
            search_space_dict["oh_factor"] = oh_bn
            search_space_dict["ow_factor"] = ow_bn
        else:
            search_space_dict["schedule_template_name"] = AVXConvCommonFwd
            search_space_dict["reg_n"] = ow_bn
            search_space_dict["unroll_kw"] = unroll_kw
        search_space_dict["ic_bn"] = ic_bn
        search_space_dict["oc_bn"] = oc_bn
        return search_space_dict

    def _get_layout_related_fields(self):
        return "ic_bn", "oc_bn"


if __name__ == "__main__":
    executor = Conv2dAVXExecutor({}, "llvm -mcpu=core-avx2", verbose=True)
    param_list = [{"channels": 64, "kernel_size":(3, 3),
                   "layout": "NCHW8c", "out_layout": "NCHW8c",
                   "kernel_layout": "OIHW8i8o"}]
    input_shape_list = [{"data": (1, 24, 70, 70, 8)}]
    executor.parameter_execute(param_list, input_shape_list)

