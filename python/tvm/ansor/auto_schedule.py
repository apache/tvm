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
User interface for Ansor auto-scheduler.

The basic schedule search process for Ansor is designed to be:
`Program sampling` -> `Performance Tuning`.

In `Program sampling`, we use some predefined precise or heuristic rules to generate several
initial schedules. Based on these initial starting points, we perform `Performance Tuning` which
uses cost model based evolutionary search to select schedules with the best performance.

Candidate schedules are measured against the specific hardware target.
"""

import tvm._ffi
from tvm.runtime import Object
from .compute_dag import ComputeDAG
from .measure import LocalBuilder, LocalRunner
from . import _ffi_api


@tvm._ffi.register_object("ansor.HardwareParams")
class HardwareParams(Object):
    """ The parameters of target hardware used to guide the search process of SearchPolicy.

    TODO(jcf94): This is considered to be merged with the new Target:
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


@tvm._ffi.register_object("ansor.SearchTask")
class SearchTask(Object):
    """ The computation information and hardware parameters for a specific schedule search task.

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


@tvm._ffi.register_object("ansor.SearchPolicy")
class SearchPolicy(Object):
    """ The base class of search policies. """


@tvm._ffi.register_object("ansor.EmptyPolicy")
class EmptyPolicy(SearchPolicy):
    """ This is an example empty search policy which will always generate
    the init state of ComputeDAG.
    """
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.EmptyPolicy)


@tvm._ffi.register_object("ansor.TuningOptions")
class TuningOptions(Object):
    """ This controls the options of performance tuning.

    Parameters
    ----------
    num_measure_trials: int = 0
      The number of measurement trials.
      The search policy measures `num_measure_trials` schedules in total and returns the best one
      among them.
      With `num_measure_trials` == 0, the policy will do the schedule search but won't involve
      measurement.
      This can be used to get a runnable schedule quickly without auto-tuning.
    early_stopping: int = -1
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
        - ansor.LogToFile
    pre_search_callbacks: Optional[List[SearchCallback]]
      Callback functions called before the search process.
      Candidates:
        - ansor.PreloadMeasuredStates
        - ansor.PreloadCustomSketchRule
        TODO(jcf94): Add these implementation in later PRs.
    """
    def __init__(self, num_measure_trials=0, early_stopping=-1, num_measures_per_round=64,
                 verbose=1, builder='local', runner='local', measure_callbacks=None,
                 pre_search_callbacks=None):
        if isinstance(builder, str):
            if builder == 'local':
                builder = LocalBuilder()
            else:
                raise ValueError("Invalid builder: " + builder)
        elif not isinstance(builder, tvm.ansor.measure.ProgramBuilder):
            raise ValueError("Invalid builder: " + builder +
                             " . TuningOptions expects a ProgramBuilder or string.")

        if isinstance(runner, str):
            if runner == 'local':
                runner = LocalRunner()
            else:
                raise ValueError("Invalid runner: " + runner)
        elif not isinstance(runner, tvm.ansor.measure.ProgramRunner):
            raise ValueError("Invalid runner: " + runner +
                             " . TuningOptions expects a ProgramRunner or string.")

        measure_callbacks = measure_callbacks if measure_callbacks else []
        pre_search_callbacks = pre_search_callbacks if pre_search_callbacks else []

        self.__init_handle_by_constructor__(
            _ffi_api.TuningOptions, num_measure_trials, early_stopping, num_measures_per_round,
            verbose, builder, runner, measure_callbacks, pre_search_callbacks)


def auto_schedule(task, target, target_host=None, search_policy='default',
                  hardware_params=None, tuning_options=None):
    """ Do auto scheduling for a computation declaration.

    The task parameter can be a `string` as workload_key, or directly
    passing a `SearchTask` as input.

    Parameters
    ----------
    task : Union[SearchTask, str]
        The SearchTask or workload key for the computation declaration.
    target : tvm.target.Target
        The target device of this schedule search.
    target_host : Optional[tvm.target.Target]
        The target host device of this schedule search.
    search_policy : Union[SearchPolicy, str] = 'default'
        The search policy to be used for schedule search.
    hardware_params : Optional[HardwareParams]
        The hardware parameters of this schedule search.
    tuning_options : Optional[TuningOptions]
        Tuning and measurement options.

    Returns
    -------
        A `te.schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
    """
    if isinstance(search_policy, str):
        if search_policy == 'default':
            # TODO(jcf94): This is an example policy for minimum system, will be upgrated to
            # formal search policy later.
            search_policy = EmptyPolicy()
        else:
            raise ValueError("Invalid search policy: " + search_policy)

    tuning_options = tuning_options if tuning_options else TuningOptions()

    if isinstance(task, str):
        dag = ComputeDAG(task)
        task = SearchTask(dag, task, target, target_host, hardware_params)
    elif not isinstance(task, SearchTask):
        raise ValueError("Invalid task: " + task +
                         " . `ansor.auto_schedule` expects a `str` or `SearchTask`.")

    sch, tensors = _ffi_api.AutoSchedule(task, search_policy, tuning_options)
    return sch, tensors
