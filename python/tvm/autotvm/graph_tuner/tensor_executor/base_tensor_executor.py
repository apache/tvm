"""API to execute an operator benchmark job."""

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
from ..utils import log_msg
from ..rpc import run_remote_module

class RPCMode(enum.Enum):
    local = 0
    rpc_host = 1
    rpc_tracker = 2

class BaseTensorExecutor(object):
    """Base class to execute operator benchmark job.

    Two methods for benchmarking are supported:
    1. parameter_execute. This method allows to pass a list of operator parameters and input shapes.
       It will return a list of execution time.
    2. workload_execute. This method allows to pass a list of workloads and schedule search space.

    _get_op_sym method is required to be overridden. It should return an NNVM operator symbol.
    Some methods need to be overridden before workload_execute can be called.
    """
    def __init__(self, schedule_dict=None, target="llvm", input_dtype="float32",
                 weight_dtype="float32",min_exec_num=10, verbose=True,
                 rpc_mode=RPCMode.local.value, rpc_hosts=None, rpc_ports=None, export_lib_format=".o",
                 file_logger=None, console_logger=None, log_file="tensor_tuning.log",
                 log_level=logging.DEBUG):
        """Create a tensor executor instance. RPC hosts mode is supported.(Create RPC host server on
        target devices and connect from host device).

        TODO Support RPC tracker.

        Parameters
        ----------
        schedule_dict : dict of namedtuple to list of dict, optional
            Schedule candidates for all workloads. Key is workload and value is a
            list of dictionary, which in format {"schedule": sch, "time": execution_time}.
            Time value is in millisecond.

            This argument is only used by workload_execute method when force_search is not set.
            When this dictionary is not empty, workload_execute will check whether the input
            workloads already exist in schedule_dict and skip them if existing.
            This will help reduce benchmarking time.

        target : str, optional
            The build target.

        input_dtype : str or list of str, optional
            Input data type. If it is a list, the order of element should be the same
            as inputs of operator symbol.

        weight_dtype : str, optional
            Weight data type.

        min_exec_num : int, optional
            Minimum number of execution. Final execution time is the average of
            all execution time.

        verbose : boolean, optional
            Whether to log to console.

        rpc_mode : int, optional
            RPC mode. 0 represents local mode. 1 represents rpc host mode.
            2 represents rpc tracker mode. Currently only 0 and 1 are supported.

        rpc_hosts : list of str, optional
            List of rpc hosts for rpc host mode.

        rpc_ports : list of int, optional
            List of rpc ports for rpc host mode.

        export_lib_format : str, optional
            Remote lib format. Currently ".o", ".so"
            and ".tar" are supported.

        file_logger : object, optional
            File logger object. If it is not provided, a new file logger will be created.

        console_logger : object, optional
            Console logger object. If it is not provided, a new console logger will be created.

        log_file : str, optional
            Log file.

        log_level : int, optional
            Log level.
        """
        self._verbose = verbose
        self._formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        if file_logger is None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._formatter)
            self._file_logger = logging.getLogger(__name__ + "_file_logger")
            self._file_logger.addHandler(file_handler)
            self._file_logger.setLevel(log_level)
        else:
            self._file_logger = file_logger
        if self._verbose:
            if console_logger is None:
                self._console_logger = logging.getLogger(__name__ + "_console_logger")
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(self._formatter)
                self._console_logger.addHandler(console_handler)
                self._console_logger.setLevel(log_level)
            else:
                self._console_logger = console_logger
            self._console_logger.propagate = False

        self._sch_dict = schedule_dict if schedule_dict else {}
        self._target = target
        self._input_dtype = input_dtype
        self._weight_dtype = weight_dtype
        self._min_exec_num = min_exec_num
        self._rpc_mode = rpc_mode
        self._rpc_hosts = rpc_hosts
        self._rpc_ports = rpc_ports
        self._export_lib_format = export_lib_format

        if self._rpc_mode == RPCMode.rpc_host.value:
            if self._rpc_hosts is None:
                log_msg("Number of rpc hosts must be at least 1 for rpc host mode.",
                        self._file_logger, self._console_logger, "error", self._verbose)
            if self._rpc_ports is None:
                log_msg("Number of rpc ports must be at least 1 for rpc host mode.",
                        self._file_logger, self._console_logger, "error", self._verbose)
            if len(self._rpc_hosts) != len(self._rpc_ports):
                log_msg("Number of rpc hosts and ports should be equal. "
                        "Got %d vs %d" % (len(self._rpc_hosts), len(self._rpc_ports)),
                        self._file_logger, self._console_logger, "error", self._verbose)
        # TODO: support RPC tracker.
        elif self._rpc_mode == RPCMode.rpc_tracker.value:
            log_msg("RPC tracker not supported.", self._file_logger,
                    self._console_logger, "error", self._verbose)

    @property
    def get_schedule_dict(self):
        """Get schedule dictionary.

        Returns
        -------
        sch_dict : dict of namedtuple to list of dict, optional
            Schedule dictionary.
        """
        return self._sch_dict

    @abstractmethod
    def _get_op_symbol(self):
        """NNVM operator symbol.

        It should return NNVM symbol of operator to be benchmark.
        """
        pass

    def _get_layout_related_fields(self):
        """Get fields of schedule template which relates to
        data layout. For example, for conv2d AVX schedule template
        AVXConvCommonFwd(ic_bn, oc_bn, reg_n, unroll_kw), this
        method should return "ic_bn" and "oc_bn".

        Override of this method is required if workload_execute is used.
        """
        log_msg("Not Implemented.", self._file_logger, self._console_logger,
                "error", self._verbose)

    def _workload2params(self, workload, schedule):
        """Generate operator parameters given a workload and schedule.

        It should return a dictionary containing parameters and need to be
        overridden if workload_execute is used.
        """
        log_msg("Not Implemented.", self._file_logger, self._console_logger,
                "error", self._verbose)

    def _workload2ishapes(self, workload, schedule):
        """Generate input shapes given a workload and schedule.

        It should return a dictionary containing input shapes and need to be
        overridden if workload_execute is used.
        """
        log_msg("Not Implemented.", self._file_logger, self._console_logger,
                "error", self._verbose)

    def _load_schedule(self, schedule):
        """Load schedule to enable it during compilation.

        This method should be overridden if workload_execute is used.
        """
        log_msg("Not Implemented.", self._file_logger, self._console_logger,
                "error", self._verbose)

    def _create_search_space(self, workload):
        """Generate search space given a workload.

        It should return a dictionary from str to list of number.
        The key represents field of schedule template and value is
        a list of possible values. An entry of schedule_template_name
        to schedule template object should also be included.
        """
        log_msg("Not Implemented. You need to either pass a "
                "search_space_list when calling workload_execute "
                "or override this method.", self._file_logger,
                self._console_logger, "error", self._verbose)

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
        """Convert search space dictionary to schedules.

        It uses Cartesian product to iterate search space.
        """
        if "schedule_template_name" not in search_space_dict:
            log_msg("schedule_template_name key not found in seach space dictionary."
                    "Please add name of the namedtuple representing "
                    "schedule template.", self._file_logger, self._console_logger,
                    "error", self._verbose)
        sch_template = search_space_dict["schedule_template_name"]
        sch_fields_list = sch_template._fields
        search_candidates_list = []
        sch_list = []
        for sch_field in sch_fields_list:
            search_candidates_list.append(search_space_dict[sch_field])
        for search_entry in itertools.product(*search_candidates_list):
            sch_list.append(sch_template._make(search_entry))
        return sch_list

    def _atomic_exec(self, net, input_shapes, sch=None, random_low=0, random_high=1,
                     rpc_mode=RPCMode.local.value, rpc_host="localhost",
                     rpc_port="9090", extra_compile_params=None):
        if extra_compile_params is None:
            extra_compile_params = {}
        input_data_dict = {}
        in_dtype_list = self._input_dtype
        if not isinstance(self._input_dtype, list):
            in_dtype_list = [self._input_dtype for _ in range(len(input_shapes))]
        for item, dtype in zip(input_shapes.items(), in_dtype_list):
            name, shape = item
            input_data_dict[name] = np.random.uniform(random_low, random_high,
                                                      size=shape).astype(dtype)
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
            init_value = np.zeros(v).astype(self._weight_dtype)
            initializer(k, init_value)
            params[k] = tvm.nd.array(init_value, ctx=tvm.cpu())

        if sch is not None:
            self._load_schedule(sch)
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(net, self._target, shape=input_shapes,
                                                     params=params,
                                                     **extra_compile_params)

        if rpc_mode == RPCMode.local.value:
            ctx = tvm.context(self._target)
            module = graph_runtime.create(graph, lib, ctx)
            module.set_input(**params)
            module.set_input(**input_data_dict)
            time_eval = module.module.time_evaluator("run", ctx, number=self._min_exec_num)
            exec_time = time_eval().mean
            # Run more times to get stable data if necessary
            real_exec_number = self._get_real_exec_number(exec_time * 1000)
            if real_exec_number > self._min_exec_num:
                time_eval = module.module.time_evaluator("run", ctx, number=real_exec_number)
                exec_time = time_eval().mean
        elif rpc_mode == RPCMode.rpc_host.value:
            session = tvm.rpc.connect(rpc_host, rpc_port)
            exec_time = run_remote_module(session, graph, lib, params, input_data_dict,
                                          remote_dev_type=self._target, run_times=self._min_exec_num,
                                          export_lib_format=self._export_lib_format)
            # Run more times to get stable data if necessary
            real_exec_number = self._get_real_exec_number(exec_time * 1000)
            if real_exec_number > self._min_exec_num:
                exec_time = run_remote_module(session, graph, lib, params, input_data_dict,
                                              remote_dev_type=self._target, run_times=real_exec_number,
                                              export_lib_format=self._export_lib_format)
        else:
           log_msg("RPC tracker not supported.", self._file_logger, self._console_logger,
                   "error", self._verbose)

        exec_time *= 1000
        if self._verbose:
            self._console_logger.info("Execution time for operator %s with input_shapes=%s, "
                              "parameters=%s, schedule=%s: %f ms."
                              % (self._get_op_symbol().__name__, str(input_shapes),
                                 str(net.list_attr()), str(sch), exec_time))
        return exec_time

    def _process_wrapper(self, *args, **kwargs):
        ret_val = kwargs["ret_val"]
        exec_kwargs = dict(kwargs)
        del exec_kwargs["ret_val"]
        exec_time = self._atomic_exec(*args, **exec_kwargs)
        ret_val.value = exec_time

    def parameter_execute(self, param_list, input_shape_list, sch_list=None,
                          extra_compile_params=None, random_low=0, random_high=1):
        def _launch_atomic_job(proc_param, proc_input_shapes, proc_ret_val, **kwargs):
            input_var_list = [sym.Variable(name=input_name) for input_name in proc_input_shapes.keys()]
            op_sym = self._get_op_symbol()
            net = op_sym(*input_var_list, **proc_param)
            args = [net, proc_input_shapes]
            kwargs["ret_val"] = proc_ret_val
            new_proc = Process(target=self._process_wrapper, args=args, kwargs=kwargs)
            new_proc.start()
            new_proc.join()

        if len(param_list) != len(input_shape_list):
            log_msg("Length of parameter list and input shape list must be equal: "
                    "%d vs %d." % (len(param_list), len(input_shape_list)),
                    self._file_logger, self._console_logger, "error", self._verbose)

        ret_list = []
        if self._rpc_mode == RPCMode.local.value:
            ret_val = Value("f", 0.0)
            for i, item in enumerate(zip(param_list, input_shape_list)):
                param, input_shapes = item
                sch = sch_list[i] if sch_list else None
                _launch_atomic_job(param, input_shapes, ret_val, sch=sch,
                                   random_low=random_low, random_high=random_high,
                                   extra_compile_params=extra_compile_params)
                ret_list.append(ret_val.value)
        elif self._rpc_mode == RPCMode.rpc_host.value:
            def _launch_group_jobs(proc_param_list, proc_input_shape_list, proc_sch_list, proc_ret_array, **kwargs):
                for i, item in enumerate(zip(proc_param_list, proc_input_shape_list,
                                             proc_sch_list)):
                    proc_param, proc_input_shapes, proc_sch = item
                    atomic_ret_value = Value("f", 0.0)
                    kwargs["sch"] = proc_sch_list[i]
                    _launch_atomic_job(proc_param, proc_input_shapes, atomic_ret_value, **kwargs)
                    proc_ret_array[i] = atomic_ret_value.value

            proc_pool = []
            num_jobs = len(param_list)
            num_proc = len(self._rpc_hosts)
            num_jobs_per_proc = (num_jobs - num_jobs % num_proc) // num_proc
            proc_ret_list = []

            end_idx = 0
            for i, item in enumerate(zip(self._rpc_hosts, self._rpc_ports)):
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
                kwargs = {"rpc_mode": self._rpc_mode, "rpc_host": rpc_host, "rpc_port": rpc_port,
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
            log_msg("RPC tracker not supported.", self._file_logger, self._console_logger,
                    "error", self._verbose)

        for i, item in enumerate(zip(param_list, input_shape_list, ret_list)):
            params, input_shapes, exec_time = item
            self._file_logger.info("Execution time for operator %s with input_shapes=%s, "
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
                                                    random_low=random_low, random_high=random_high)

            # Order schedule by execution time and only preserve
            # the fastest schedule in schedules generating the same
            # data layout.
            sch_record_dict = {}
            for sch, exec_time in zip(sch_list, exec_time_list):
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
