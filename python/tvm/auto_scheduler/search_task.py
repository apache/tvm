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

""" The definiton of SearchTask """

import json

import os
import logging
import numpy as np

import tvm._ffi
from tvm.runtime import Object, ndarray

from tvm.driver.build_module import build
from tvm.target import Target
from .measure import LocalBuilder, LocalRunner
from .measure_record import load_best_record
from .workload_registry import make_workload_key
from .compute_dag import ComputeDAG, LayoutRewriteOption
from .cost_model import XGBModel
from .search_policy import SketchPolicy
from .workload_registry import WORKLOAD_FUNC_REGISTRY, register_workload_tensors
from . import _ffi_api

# pylint: disable=invalid-name
logger = logging.getLogger("auto_scheduler")


@tvm._ffi.register_object("auto_scheduler.HardwareParams")
class HardwareParams(Object):
    """The parameters of target hardware used to guide the search policy.

    When a parameter isn't provided, it will instead use the
    current machine's default value if target is specified.
    TODO(jcf94): This is considered to be merged with the new Target specification:
    https://discuss.tvm.apache.org/t/rfc-tvm-target-specification/6844
    Parameters
    ----------
    num_cores : int, optional
        The number of device cores.
    vector_unit_bytes : int, optional
        The width of vector units in bytes.
    cache_line_bytes : int, optional
        The size of cache line in bytes.
    max_shared_memory_per_block : int, optional
        The max shared memory per block in bytes.
    max_local_memory_per_block : int, optional
        The max local memory per block in bytes.
    max_threads_per_block : int, optional
        The max number of threads per block.
    max_vthread_extent : int, optional
        The max vthread extent.
    warp_size : int, optional
        The thread numbers of a warp.
    target : str or Target, optional
        The compilation target. Used to determine default values if provided.
    target_host : str or Target, optional
        The compilation target host. Used to determine default values if provided.
    """

    def __init__(
        self,
        num_cores=None,
        vector_unit_bytes=None,
        cache_line_bytes=None,
        max_shared_memory_per_block=None,
        max_local_memory_per_block=None,
        max_threads_per_block=None,
        max_vthread_extent=None,
        warp_size=None,
        target=None,
        target_host=None,
    ):
        # If target is provided, get the default paramters for this machine.
        if target is not None:
            if isinstance(target, str):
                target = tvm.target.Target(target)
            if isinstance(target_host, str):
                target_host = tvm.target.Target(target_host)
            default_params = _ffi_api.GetDefaultHardwareParams(target, target_host)

            if num_cores is None:
                num_cores = default_params.num_cores
            if vector_unit_bytes is None:
                vector_unit_bytes = default_params.vector_unit_bytes
            if cache_line_bytes is None:
                cache_line_bytes = default_params.cache_line_bytes
            if max_shared_memory_per_block is None:
                max_shared_memory_per_block = default_params.max_shared_memory_per_block
            if max_local_memory_per_block is None:
                max_local_memory_per_block = default_params.max_local_memory_per_block
            if max_threads_per_block is None:
                max_threads_per_block = default_params.max_threads_per_block
            if max_vthread_extent is None:
                max_vthread_extent = default_params.max_vthread_extent
            if warp_size is None:
                warp_size = default_params.warp_size

        self.__init_handle_by_constructor__(
            _ffi_api.HardwareParams,
            num_cores,
            vector_unit_bytes,
            cache_line_bytes,
            max_shared_memory_per_block,
            max_local_memory_per_block,
            max_threads_per_block,
            max_vthread_extent,
            warp_size,
        )

    def __str__(self):
        """Pretty printing for hardware parameter configuration."""
        format_str = (
            "HardwareParams:\n"
            f"  num_cores: {self.num_cores}\n"
            f"  vector_unit_bytes: {self.vector_unit_bytes}\n"
            f"  cache_line_bytes: {self.cache_line_bytes}\n"
            f"  max_shared_memory_per_block: {self.max_shared_memory_per_block}\n"
            f"  max_local_memory_per_block: {self.max_local_memory_per_block}\n"
            f"  max_threads_per_block: {self.max_threads_per_block}\n"
            f"  max_vthread_extent: {self.max_vthread_extent}\n"
            f"  warp_size: {self.warp_size}\n"
        )
        return format_str


@tvm._ffi.register_object("auto_scheduler.TuningOptions")
class TuningOptions(Object):
    """This controls the options of performance tuning.

    Parameters
    ----------
    num_measure_trials: int = 0
        The number of measurement trials.
        The search policy measures `num_measure_trials` schedules in total and returns the best one
        among them.
        With `num_measure_trials` == 0, the policy will do the schedule search but won't involve
        measurement. This can be used to get a runnable schedule quickly without auto-tuning.
    early_stopping: Optional[int]
        Stop the tuning early if getting no improvement after n measurements.
    num_measures_per_round: int = 64
        The number of schedules to be measured at each search round.
        The whole schedule search process will try a total number of `num_measure_trials` in several
        rounds.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during schedule search.
    builder: Union[ProgramBuilder, str] = 'local'
        ProgramBuilder which builds the program.
    runner: Union[ProgramRunner, str] = 'local'
        ProgramRunner which runs the program and measures time costs.
    measure_callbacks: Optional[List[MeasureCallback]]
        Callback functions called after each measurement.
        Candidates:
        - auto_scheduler.RecordToFile
    """

    def __init__(
        self,
        num_measure_trials=0,
        early_stopping=None,
        num_measures_per_round=64,
        verbose=1,
        builder="local",
        runner="local",
        measure_callbacks=None,
    ):
        if isinstance(builder, str):
            if builder == "local":
                builder = LocalBuilder()
            else:
                raise ValueError("Invalid builder: " + builder)
        elif not isinstance(builder, tvm.auto_scheduler.measure.ProgramBuilder):
            raise ValueError(
                "Invalid builder: "
                + builder
                + " . TuningOptions expects a ProgramBuilder or string."
            )

        if isinstance(runner, str):
            if runner == "local":
                runner = LocalRunner()
            else:
                raise ValueError("Invalid runner: " + runner)
        elif not isinstance(runner, tvm.auto_scheduler.measure.ProgramRunner):
            raise ValueError(
                "Invalid runner: " + runner + " . TuningOptions expects a ProgramRunner or string."
            )

        self.__init_handle_by_constructor__(
            _ffi_api.TuningOptions,
            num_measure_trials,
            early_stopping or -1,
            num_measures_per_round,
            verbose,
            builder,
            runner,
            measure_callbacks,
        )


# The map stores special registered buffer for measurement.
# This can be used for sparse workloads when we cannot use random tensors for measurment.
# {
#     "workload_key_0": {
#         "task_input_0": Tensor(...),
#         "task_input_1": Tensor(...)
#     },
#     "workload_key_1": {
#         "task_input_2": Tensor(...),
#         "task_input_3": Tensor(...)
#     },
#     ...
# }
TASK_INPUT_BUFFER_TABLE = {}


def _save_buffer_to_file(buffer_name, buffer_data):
    """Save the current Tensor buffer to a numpy file.

    File name will be: {buffer_name}.{buffer_shape}_{buffer_data_type}.npy
    """
    np_data = buffer_data.numpy()

    buffer_name += "."
    for i in np_data.shape:
        buffer_name += "%d_" % (i)
    buffer_name += "%s" % (np_data.dtype)
    buffer_name += ".npy"

    np_data.tofile(buffer_name, " ")


def _try_load_buffer_from_file(buffer_name):
    """Try to load buffer from a numpy file, if not found, return None.

    File name has a same format as `_save_buffer_to_file`.
    """
    filelist = os.listdir()

    for file in filelist:
        if file.startswith(buffer_name + "."):
            meta_info = file.split(".")[-2].split("_")
            shape = [int(i) for i in meta_info[:-1]]
            dtype = meta_info[-1]
            buffer_data = np.fromfile(file, dtype=dtype, sep=" ")
            buffer_data = buffer_data.reshape(shape)
            return ndarray.array(buffer_data)

    return None


def register_task_input_buffer(
    workload_key,
    input_name,
    input_data,
    overwrite=False,
    save_to_file=False,
):
    """Register special buffer for measurement.

    Parameters
    ----------
    workload_key : str
        The workload key of the SearchTask.

    input_name : str
        The name of input buffer.

    input_data : tvm.nd.NDArray
        The input Tensor data.

    overwrite : bool = False
        Whether to overwrite the data if a name has already registered.

    save_to_file : bool = False
        Whether to save the data to a local file as well. This can be reused to resume the last
        tuning process.

    Returns
    -------
    tvm.nd.NDArray
        The actual registered Tensor data of this input_name. With `overwrite` set to False, will
        return the original one if the name has already registered before.
    """
    global TASK_INPUT_BUFFER_TABLE

    if workload_key not in TASK_INPUT_BUFFER_TABLE:
        TASK_INPUT_BUFFER_TABLE[workload_key] = {}
    input_table = TASK_INPUT_BUFFER_TABLE[workload_key]

    if not overwrite:
        if input_name not in input_table.keys():
            # Try to load buffer data from local file
            tensor_from_file = _try_load_buffer_from_file(input_name)
            if tensor_from_file:
                input_table[input_name] = tensor_from_file
        elif input_name in input_table.keys():
            raise RuntimeError(
                "Tensor %s exists in TASK_INPUT_BUFFER_TABLE, %s"
                % (input_name, "set overwrite to True or this Tensor will not be registered")
            )

    input_table[input_name] = input_data
    if save_to_file:
        _save_buffer_to_file(input_name, input_data)
    return input_data


def get_task_input_buffer(workload_key, input_name):
    """Get special buffer for measurement.

    The buffers are registered by `register_task_input_buffer`.

    Parameters
    ----------
    workload_key : str
        The workload key of the SearchTask.

    input_name : str
        The name of input buffer.

    Returns
    -------
    tvm.nd.NDArray
        The registered input buffer.
    """
    global TASK_INPUT_BUFFER_TABLE

    if workload_key not in TASK_INPUT_BUFFER_TABLE:
        TASK_INPUT_BUFFER_TABLE[workload_key] = {}
    input_table = TASK_INPUT_BUFFER_TABLE[workload_key]

    if input_name not in input_table:
        # Try to load buffer data from local file
        tensor_from_file = _try_load_buffer_from_file(input_name)
        if tensor_from_file:
            input_table[input_name] = tensor_from_file

    # Then check for the default table, the input names extracted from a relay model will be
    # stored here for we're not able to get the workload_key at that time
    if input_name not in input_table:
        input_table = TASK_INPUT_BUFFER_TABLE["default"]

    if input_name in input_table:
        return input_table[input_name]

    raise ValueError(
        "%s not found in TASK_INPUT_BUFFER_TABLE, " % (input_name)
        + "should provide with `SearchTask(..., task_inputs={...})`"
    )


@tvm._ffi.register_object("auto_scheduler.SearchTask")
class SearchTask(Object):
    """The computation information and hardware parameters for a schedule search task.

    Parameters
    ----------
    func : Union[Function, str]
        The function that returns the compute declaration Tensors.
        Can be the a function or the function name.
    args : Union[Tuple[Any, ...], List[Any]]
        The args of the function.
    compute_dag : ComputeDAG
        The ComputeDAG for the corresponding compute declaration.
    workload_key : str
        The workload key for the corresponding compute declaration.
    target : any target-like object, see Target.canon_target
        The target device of this search task.
    target_host : None or any target-like object, see Target.canon_target
        The target host device of this search task.
    hardware_params : Optional[HardwareParams]
        Hardware parameters used in this search task.
    layout_rewrite_option : Optional[LayoutRewriteOption]
        The layout rewrite option used for measuring programs. If None, the default value will be
        set depending on the specified target.
        Auto_scheduler will find a better schedule for the specified layout rewrite option.
        The NO_REWRITE and INSERT_TRANSFORM_STAGE are expected to be used when tuning a standalone
        op, and the REWRITE_FOR_PRE_TRANSFORMED is expected to be used when tuning ops inside a
        network.
    task_inputs : Union[Dict[str, tvm.nd.NDArray], List[str]]
        A dict maps the input names to input tensors or a list of input names.
        Some special Tensor used as inputs in program measuring. Usually we do not need to care
        about it, but for special workloads like Sparse computation the Sparse Tensor input are
        meaningful that we cannot use random input directly.
    task_inputs_overwrite : bool = False
        Whether to overwrite the data if a name has already in the global table.
    task_inputs_save_to_file : bool = False
        Whether to save the data to a local file as well. This can be reused to resume the last
        tuning process.
    desc: str = ""
        The description string of this task.

    Examples
    --------
    .. code-block:: python

      # We support two ways to create a search task

      # Way 1: create a task by a workload generation function.
      # The `workload_func` is a function decorated by @auto_scheduler.register_workload
      task = SearchTask(func=workload_func, args=args, target=target)

      # Way 2: create a task by a workload_key.
      # The `workload_key` is a string, which can be either a hash key or a json-serialized
      # tuple(func, args).
      task = SearchTask(workload_key=workload_key, target=target)
    """

    def __init__(
        self,
        func=None,
        args=None,
        compute_dag=None,
        workload_key=None,
        target=None,
        target_host=None,
        hardware_params=None,
        layout_rewrite_option=None,
        task_inputs=None,
        task_inputs_overwrite=False,
        task_inputs_save_to_file=False,
        desc="",
    ):
        assert (
            func is not None or workload_key is not None
        ), "Either a workload generation function or a workload key should be provided"

        if func is not None:
            workload_key = make_workload_key(func, args)
        if compute_dag is None:
            compute_dag = ComputeDAG(workload_key)

        assert target is not None, "Must specify a target."

        target, target_host = Target.canon_target_and_host(target, target_host)

        if layout_rewrite_option is None:
            layout_rewrite_option = LayoutRewriteOption.get_target_default(target)

        task_input_names = []
        if isinstance(task_inputs, list):
            task_input_names = task_inputs
        elif isinstance(task_inputs, dict):
            for input_name in task_inputs:
                register_task_input_buffer(
                    workload_key,
                    input_name,
                    task_inputs[input_name],
                    task_inputs_overwrite,
                    task_inputs_save_to_file,
                )
                task_input_names.append(input_name)
        elif task_inputs is not None:
            raise ValueError("task_inputs should be a dict or a list.")

        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,
            compute_dag,
            workload_key,
            target,
            target_host,
            hardware_params,
            layout_rewrite_option,
            task_input_names,
            desc,
        )

    def tune(self, tuning_options, search_policy=None):
        """Run auto scheduling search for a task

        Parameters
        ----------
        tuning_options : TuningOptions
            Tuning and measurement options.
        search_policy : Optional[SearchPolicy]
            The search policy to be used for schedule search.
        """
        if search_policy is None:
            cost_model = XGBModel()
            search_policy = SketchPolicy(self, cost_model)

        _ffi_api.AutoSchedule(search_policy, tuning_options)

    def apply_best(self, log_file, include_compatible=False, layout_rewrite_option=None):
        """Apply the history best from a log file and return the schedule.

        Parameters
        ----------
        log_file : str
           The name of the log file.
        include_compatible: bool
            When set to True, all compatible records in the log file will be considered.
        layout_rewrite_option : Optional[LayoutRewriteOption]
           The layout rewrite option.


        Returns
        -------
            A `te.Schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
        """
        inp, _ = load_best_record(
            log_file, self.workload_key, include_compatible=include_compatible
        )
        if inp is None:
            raise RuntimeError(
                "Cannot find any valid schedule for %s in file %s" % (self.workload_key, log_file)
            )

        sch, args = self.compute_dag.apply_steps_from_state(
            inp.state, layout_rewrite_option or self.layout_rewrite_option
        )
        return sch, args

    def print_best(self, log_file, print_mode="schedule"):
        """Print the best schedule as python schedule API code or CUDA source code.

        Parameters
        ----------
        log_file : str
           The name of the log file
        print_mode: str
           if "schedule", print the best schedule as python schedule API code.
           if "cuda", print the best schedule as CUDA source code.

        Returns
        -------
        code: str
            The best schedule code in python API or CUDA source code
        """
        inp, _ = load_best_record(log_file, self.workload_key)
        if inp is None:
            raise RuntimeError(
                "Cannot find any valid schedule for %s in file %s" % (self.workload_key, log_file)
            )

        if print_mode == "schedule":
            return self.compute_dag.print_python_code_from_state(inp.state)
        if print_mode == "cuda":
            assert self.target.kind.name == "cuda"
            sch, args = self.compute_dag.apply_steps_from_state(inp.state)
            func = build(sch, args, "cuda")
            return func.imported_modules[0].get_source()
        raise ValueError("Invalid print_mode: %s" % print_mode)

    def __getstate__(self):
        self.target, self.target_host = Target.canon_target_and_host(self.target, self.target_host)
        return {
            "compute_dag": self.compute_dag,
            "workload_key": self.workload_key,
            "target": self.target,
            "target_host": self.target_host,
            "hardware_params": self.hardware_params,
            "layout_rewrite_option": self.layout_rewrite_option,
            "task_input_names": self.task_input_names,
            "desc": self.desc,
        }

    def __setstate__(self, state):
        # Register the workload if needed
        try:
            workload = json.loads(state["workload_key"])
        except Exception:  # pylint: disable=broad-except
            raise RuntimeError("Invalid workload key %s" % state["workload_key"])

        # workload[0] is either the compute function name or the ComputeDAG hash.
        # The compute functions are already registered when importing TVM, so here
        # we only register the ComputeDAG workloads. If the same workload has
        # already been registered, the later registration overrides the prvious one.
        if workload[0] not in WORKLOAD_FUNC_REGISTRY:
            register_workload_tensors(state["workload_key"], state["compute_dag"].tensors)

        state["target"], state["target_host"] = Target.canon_target_and_host(
            state["target"], state["target_host"]
        )
        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,
            state["compute_dag"],
            state["workload_key"],
            state["target"],
            state["target"].host,
            state["hardware_params"],
            state["layout_rewrite_option"],
            state["task_input_names"],
            state["desc"],
        )


def create_task(func, args, target, target_host=None, hardware_params=None):
    """THIS API IS DEPRECATED.

    Create a search task.

    Parameters
    ----------
    func : Union[Function, str]
        The function that returns the compute declaration Tensors.
        Can be the a function or the function name.
    args : Union[Tuple[Any, ...], List[Any]]
        The args of the function.
    target : Union[tvm.target.Target, str]
        The target device of this search task.
    target_host : Optional[Union[tvm.target.Target, str]]
        The target host device of this search task.
    hardware_params : Optional[HardwareParams]
        Hardware parameters used in this search task.

    Returns
    -------
        SearchTask: the created task
    """
    raise ValueError(
        'The API "auto_scheduler.create_task" is deprecated.'
        "See https://github.com/apache/tvm/pull/7028 for the upgrade guide"
    )


def auto_schedule(task, search_policy=None, tuning_options=TuningOptions()):
    """THIS API IS DEPRECATED.

    Run auto scheduling search for a task.

    Parameters
    ----------
    task : SearchTask
        The SearchTask for the computation declaration.
    search_policy : Optional[SearchPolicy]
        The search policy to be used for schedule search.
    tuning_options : Optional[TuningOptions]
        Tuning and measurement options.

    Returns
    -------
        A `te.Schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
    """
    raise ValueError(
        'The API "auto_scheduler.create_task" is deprecated.'
        "See https://github.com/apache/tvm/pull/7028 for the upgrade guide."
    )
