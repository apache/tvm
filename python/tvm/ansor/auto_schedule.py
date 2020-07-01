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

"""User interface for auto-scheduler"""

import tvm._ffi
from tvm.runtime import Object
from .measure import LocalBuilder, LocalRunner
from . import _ffi_api


@tvm._ffi.register_object("ansor.HardwareParams")
class HardwareParams(Object):
    """ The parameters of target hardware, this is used to guide the search process of
    SearchPolicy.

    Parameters
    ----------
    num_cores : int
        The number of device cores.
    vector_unit_bytes : int
        The width of vector units in bytes.
    cache_line_bytes : int
        The size of cache line in bytes.
    max_unroll_vec : int
        The max length of an axis to be unrolled or vectorized.
    max_innermost_split_factor : int
        The max split factor for the innermost tile.
    """
    def __init__(self, num_cores, vector_unit_bytes, cache_line_bytes,
                 max_unroll_vec, max_innermost_split_factor):
        self.__init_handle_by_constructor__(_ffi_api.HardwareParams, num_cores,
                                            vector_unit_bytes, cache_line_bytes,
                                            max_unroll_vec, max_innermost_split_factor)


@tvm._ffi.register_object("ansor.SearchTask")
class SearchTask(Object):
    """ The meta-information of a search task.

    Parameters
    ----------
    dag : ComputeDAG
        The ComputeDAG for target compute declaration.
    workload_key : str
        The workload key for target compute declaration.
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
    """ The base class for search policy  """


@tvm._ffi.register_object("ansor.EmptyPolicy")
class EmptyPolicy(SearchPolicy):
    """ This is an example empty search policy which will always generate
    the init state of target ComputeDAG.
    """
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.EmptyPolicy)


@tvm._ffi.register_object("ansor.TuneOption")
class TuneOption(Object):
    """ The options for tuning.

    Parameters
    ----------
    n_trials: int = 1
      The number of total schedule measure trials.
      Ansor takes `n_trials` state for measuring in total, and finally gets the best schedule
      among them.
      With `n_trials` == 1, Ansor will do the schedule search but don't involve measurement,
      this can be used if we want to quickly get a runnable schedule without performance tuning.
    early_stopping: int = -1
      Stops early the tuning if no improvement get after n measurements.
    num_measure_per_round: int = 64
      The number of programs to be measured at each search round.
      The whole schedule search process is designed to have several rounds to try a total
      `n_trials` schedules.
      We have: `num_search_rounds` = `n_trials` // `num_measure_per_round`
    verbose: int = 1
      Verbosity level. 0 means silent.
    builder: Union[Builder, str] = 'local'
      Builder which builds the program.
    runner: Union[Runner, str] = 'local'
      Runner which runs the program and measures time costs.
    measure_callbacks: Optional[List[MeasureCallback]]
      Callback functions called after each measure.
      Candidates:
        - ansor.LogToFile
    pre_search_callbacks: Optional[List[SearchCallback]]
      Callback functions called before the search process.
      Candidates:
        - ansor.PreloadMeasuredStates
        - ansor.PreloadCustomSketchRule
        TODO(jcf94): Add these implementation in later PRs.
    """
    def __init__(self, n_trials=1, early_stopping=-1, num_measure_per_round=64,
                 verbose=1, builder='local', runner='local', measure_callbacks=None,
                 pre_search_callbacks=None):
        if isinstance(builder, str):
            if builder == 'local':
                builder = LocalBuilder()
            else:
                raise ValueError("Invalid builder: " + builder)

        if isinstance(runner, str):
            if runner == 'local':
                runner = LocalRunner()
            else:
                raise ValueError("Invalid runner: " + runner)

        measure_callbacks = [] if measure_callbacks is None else measure_callbacks
        pre_search_callbacks = [] if pre_search_callbacks is None else pre_search_callbacks

        self.__init_handle_by_constructor__(
            _ffi_api.TuneOption, n_trials, early_stopping, num_measure_per_round,
            verbose, builder, runner, measure_callbacks, pre_search_callbacks)


def auto_schedule(task, target, target_host=None, search_policy='default',
                  hardware_params=None, tune_option=None):
    """ Do auto scheduling for a computation declaration.

    The task parameter can be a `string` as workload_key, or directly
    passing a `SearchTask` as input.

    Parameters
    ----------
    task : Union[SearchTask, str]
        The target search task or workload key.
    target : tvm.target.Target
        The target device of this schedule search.
    target_host : Optional[tvm.target.Target]
        The target host device of this schedule search.
    search_policy : Union[SearchPolicy, str] = 'default'
        The search policy to be used for schedule search.
    hardware_params : Optional[HardwareParams]
        The hardware parameters of this schedule search.
    tune_option : Optional[TuneOption]
        Tuning and measurement options.

    Returns
    -------
        A `te.schedule` and the target `te.Tensor`s to be used in `tvm.lower` or `tvm.build`
    """
    if isinstance(search_policy, str):
        if search_policy == 'default':
            # TODO(jcf94): This is an example policy for minimum system, will be upgrated to
            # formal search policy later.
            search_policy = EmptyPolicy()
        else:
            raise ValueError("Invalid search policy: " + search_policy)

    tune_option = TuneOption() if tune_option is None else tune_option

    if isinstance(task, str):
        sch, tensors = _ffi_api.AutoScheduleByWorkloadKey(
            task, target, target_host, search_policy, hardware_params, tune_option)
        return sch, tensors
    if isinstance(task, SearchTask):
        sch, tensors = _ffi_api.AutoScheduleBySearchTask(task, search_policy, tune_option)
        return sch, tensors
    raise ValueError("Invalid task: " + task + ". Expect a string or SearchTask")
