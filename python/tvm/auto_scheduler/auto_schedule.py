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
User interface for TVM Auto-scheduler.

The basic schedule search process for TVM Auto-scheduler is designed to be:
`Program sampling` -> `Performance Tuning`.

In `Program sampling`, we use some predefined precise or heuristic rules to generate several
initial schedules. Based on these initial starting points, we perform `Performance Tuning` which
uses cost model based evolutionary search to select schedules with the best performance.

Candidate schedules are measured against the specific hardware target.
"""

import tvm._ffi
from tvm.runtime import Object
from .measure import LocalBuilder, LocalRunner
from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.HardwareParams")
class HardwareParams(Object):
    """ The parameters of target hardware used to guide the search policy

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
        self.__init_handle_by_constructor__(_ffi_api.HardwareParams, num_cores,
                                            vector_unit_bytes, cache_line_bytes)


@tvm._ffi.register_object("auto_scheduler.SearchTask")
class SearchTask(Object):
    """ The computation information and hardware parameters for a schedule search task.

    Parameters
    ----------
    dag : ComputeDAG
        The ComputeDAG for the corresponding compute declaration.
    workload_key : str
        The workload key for the corresponding compute declaration.
    target : tvm.target.Target
        The target device of this search task.
    target_host : Optional[tvm.target.Target]
        The target host device of this search task.
    hardware_params : Optional[HardwareParams]
        Hardware parameters used in this search task.
    """
    def __init__(self, dag, workload_key, target, target_host=None,
                 hardware_params=None):
        self.__init_handle_by_constructor__(_ffi_api.SearchTask, dag,
                                            workload_key, target, target_host,
                                            hardware_params)


@tvm._ffi.register_object("auto_scheduler.SearchPolicy")
class SearchPolicy(Object):
    """ The base class of search policies. """


@tvm._ffi.register_object("auto_scheduler.EmptyPolicy")
class EmptyPolicy(SearchPolicy):
    """ This is an example empty search policy which will always generate
    the init state of ComputeDAG.
    """
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.EmptyPolicy)


@tvm._ffi.register_object("auto_scheduler.TuningOptions")
class TuningOptions(Object):
    """ This controls the options of performance tuning.

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
    pre_search_callbacks: Optional[List[SearchCallback]]
      Callback functions called before the search process.
      Candidates:
        - auto_scheduler.PreloadMeasuredStates
        - auto_scheduler.PreloadCustomSketchRule
        TODO(jcf94): Add these implementation in later PRs.
    """
    def __init__(self, num_measure_trials=0, early_stopping=None, num_measures_per_round=64,
                 verbose=1, builder='local', runner='local', measure_callbacks=None,
                 pre_search_callbacks=None):
        if isinstance(builder, str):
            if builder == 'local':
                builder = LocalBuilder()
            else:
                raise ValueError("Invalid builder: " + builder)
        elif not isinstance(builder, tvm.auto_scheduler.measure.ProgramBuilder):
            raise ValueError("Invalid builder: " + builder +
                             " . TuningOptions expects a ProgramBuilder or string.")

        if isinstance(runner, str):
            if runner == 'local':
                runner = LocalRunner()
            else:
                raise ValueError("Invalid runner: " + runner)
        elif not isinstance(runner, tvm.auto_scheduler.measure.ProgramRunner):
            raise ValueError("Invalid runner: " + runner +
                             " . TuningOptions expects a ProgramRunner or string.")

        self.__init_handle_by_constructor__(
            _ffi_api.TuningOptions, num_measure_trials, early_stopping if early_stopping else -1,
            num_measures_per_round, verbose, builder, runner, measure_callbacks,
            pre_search_callbacks)


def auto_schedule(task, search_policy='default', tuning_options=None):
    """ Do auto scheduling for a computation declaration.

    Parameters
    ----------
    task : SearchTask
        The SearchTask for the computation declaration.
    search_policy : Union[SearchPolicy, str] = 'default'
        The search policy to be used for schedule search.
    tuning_options : Optional[TuningOptions]
        Tuning and measurement options.

    Returns
    -------
        A `te.schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
    """
    if not isinstance(task, SearchTask):
        raise ValueError("Invalid task: " + task +
                         " . `auto_scheduler.auto_schedule` expects a SearchTask.")

    if isinstance(search_policy, str):
        if search_policy == 'default':
            # TODO(jcf94): This is an example policy for minimum system, will be upgrated to
            # formal search policy later.
            search_policy = EmptyPolicy()
        else:
            raise ValueError("Invalid search policy: " + search_policy)
    elif not isinstance(search_policy, SearchPolicy):
        raise ValueError("Invalid search policy: " + search_policy +
                         " . `auto_scheduler.auto_schedule` expects a SearchPolicy or a string.")

    sch, tensors = _ffi_api.AutoSchedule(task, search_policy,
                                         tuning_options if tuning_options else TuningOptions())
    return sch, tensors
