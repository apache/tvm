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

import tvm._ffi
from tvm.runtime import Object

from tvm.driver.build_module import build
from tvm.target import Target
from .measure import LocalBuilder, LocalRunner
from .measure_record import load_best_record
from .workload_registry import make_workload_key
from .compute_dag import ComputeDAG, LayoutRewriteOption
from .cost_model import XGBModel
from .search_policy import SketchPolicy
from .workload_registry import register_workload_tensors
from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.HardwareParams")
class HardwareParams(Object):
    """The parameters of target hardware used to guide the search policy
    TODO(jcf94): This is considered to be merged with the new Target specification:
    https://discuss.tvm.apache.org/t/rfc-tvm-target-specification/6844
    Parameters
    ----------
    num_cores : int
        The number of device cores.
    vector_unit_bytes : int
        The width of vector units in bytes.
    cache_line_bytes : int
        The size of cache line in bytes.
    max_shared_memory_per_block : int
        The max shared memory per block in bytes.
    max_local_memory_per_block : int
        The max local memory per block in bytes.
    max_threads_per_block : int
        The max number of threads per block.
    max_vthread_extent : int
        The max vthread extent.
    warp_size : int
        The thread numbers of a warp.
    """

    def __init__(
        self,
        num_cores,
        vector_unit_bytes,
        cache_line_bytes,
        max_shared_memory_per_block,
        max_local_memory_per_block,
        max_threads_per_block,
        max_vthread_extent,
        warp_size,
    ):
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
    target : tvm.target.Target
        The target device of this search task.
    target_host : Optional[tvm.target.Target]
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
    ):
        assert (
            func is not None or workload_key is not None
        ), "Either a workload generation function or a workload key should be provided"

        if func is not None:
            workload_key = make_workload_key(func, args)
        if compute_dag is None:
            compute_dag = ComputeDAG(workload_key)

        assert target is not None, "Must specify a target."
        if isinstance(target, str):
            target = Target(target)
        if isinstance(target_host, str):
            target_host = Target(target_host)

        if layout_rewrite_option is None:
            layout_rewrite_option = LayoutRewriteOption.get_target_default(target)

        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,
            compute_dag,
            workload_key,
            target,
            target_host,
            hardware_params,
            layout_rewrite_option,
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

    def apply_best(self, log_file, layout_rewrite_option=None):
        """Apply the history best from a log file and return the schedule.

        Parameters
        ----------
        log_file : str
           The name of the log file.
        layout_rewrite_option : Optional[LayoutRewriteOption]
           The layout rewrite option.


        Returns
        -------
            A `te.Schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
        """
        inp, _ = load_best_record(log_file, self.workload_key)
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
        return {
            "compute_dag": self.compute_dag,
            "workload_key": self.workload_key,
            "target": self.target,
            "target_host": self.target_host,
            "hardware_params": self.hardware_params,
            "layout_rewrite_option": self.layout_rewrite_option,
        }

    def __setstate__(self, state):
        # Register the workload if needed
        try:
            workload = json.loads(state["workload_key"])
        except Exception:  # pylint: disable=broad-except
            raise RuntimeError("Invalid workload key %s" % state["workload_key"])

        # The workload from a compute DAG does not have arguments and is not registered
        # by default so we register it here. If the workload has already been registered,
        # the later registration overrides the prvious one.
        if len(workload) == 1:
            register_workload_tensors(workload[0], state["compute_dag"].tensors)

        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,
            state["compute_dag"],
            state["workload_key"],
            state["target"],
            state["target_host"],
            state["hardware_params"],
            state["layout_rewrite_option"],
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
