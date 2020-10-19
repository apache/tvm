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

"""
The user interface and tuning options of the TVM auto-scheduler.
"""

import tvm._ffi
from tvm.runtime import Object
from tvm.target import Target
from .measure import LocalBuilder, LocalRunner
from .workload_registry import make_workload_key
from .compute_dag import ComputeDAG
from .cost_model import XGBModel
from .search_policy import SketchPolicy
from .search_task import SearchTask
from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.HardwareParams")
class HardwareParams(Object):
    """The parameters of target hardware used to guide the search policy

    TODO(jcf94): This is considered to be merged with the new Target specification:
    https://discuss.tvm.ai/t/rfc-tvm-target-specification/6844

    Parameters
    ----------
    num_cores : int
        The number of device cores.
    vector_unit_bytes : int
        The width of vector units in bytes.
    cache_line_bytes : int
        The size of cache line in bytes.
    """

    def __init__(self, num_cores, vector_unit_bytes, cache_line_bytes):
        self.__init_handle_by_constructor__(
            _ffi_api.HardwareParams, num_cores, vector_unit_bytes, cache_line_bytes
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


def create_task(func, args, target, target_host=None, hardware_params=None):
    """Create a search task

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
    workload_key = make_workload_key(func, args)
    dag = ComputeDAG(workload_key)
    if isinstance(target, str):
        target = Target(target)
    if isinstance(target_host, str):
        target_host = Target(target_host)
    return SearchTask(dag, workload_key, target, target_host, hardware_params)


def auto_schedule(task, search_policy=None, tuning_options=TuningOptions()):
    """Run auto scheduling search for a task

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
    if not isinstance(task, SearchTask):
        raise ValueError(
            "Invalid task: " + task + " . `auto_scheduler.auto_schedule` expects a SearchTask."
        )

    if search_policy is None:
        cost_model = XGBModel()
        search_policy = SketchPolicy(task, cost_model)

    sch, tensors = _ffi_api.AutoSchedule(search_policy, tuning_options)
    return sch, tensors
