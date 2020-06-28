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

import random

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
    """ The meta-information of a search task

    Parameters
    ----------
    dag : ComputeDAG
        The ComputeDAG for target compute declaration.
    workload_key : str
        The workload key for target compute declaration.
    target : tvm.target.Target
        The target device of this search task.
    target_host : tvm.target.Target
        The target host device of this search task.
    hardware_params : HardwareParams
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
    """ The options for tuning

    Parameters
    ----------
    n_trials: int
      Number of total measurement trials
    early_stopping: int
      Stops early the tuning if no improvement after n measurements
    num_measure_per_round: int
      The number of programs to be measured at each iteration
    verbose: int
      Verbosity level. 0 means silent.
    builder: Builder
      Builder which builds the program
    runner: Runner
      Runner which runs the program and measure time costs
    measure_callbacks: List[MeasureCallback]
      Callback functions called after each measure
      Candidates:
        - ansor.LogToFile
    pre_search_callbacks: List[SearchCallback]
      Callback functions called before the search process
      Candidates:
        - ansor.PreloadMeasuredStates(will be added later)
        - ansor.PreloadCustomSketchRule(will be added later)
    """
    def __init__(self, n_trials=0, early_stopping=-1, num_measure_per_round=64,
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
                raise ValueError("Invalid builder: " + runner)

        if measure_callbacks is None:
            measure_callbacks = []

        if pre_search_callbacks is None:
            pre_search_callbacks = []

        self.__init_handle_by_constructor__(
            _ffi_api.TuneOption, n_trials, early_stopping, num_measure_per_round,
            verbose, builder, runner, measure_callbacks, pre_search_callbacks)


def auto_schedule(workload, target=None,
                  target_host=None, search_policy='default',
                  hardware_params=None, tune_option=None):
    """ Do auto scheduling for a computation declaration.

    The workload parameter can be a `string` as workload_key, or directly
    passing a `SearchTask` as input.

    Parameters
    ----------
    workload : Union[SearchTask, str]
        The target search task or workload key.
    target : Target
        The target device of this schedule search.
    target_host : Target = None
        The target host device of this schedule search.
    search_policy : Union[SearchPolicy, str]
        The search policy to be used for schedule search.
    hardware_params : HardwareParams
        The hardware parameters of this schedule search.
    tune_option : TuneOption
        Tuning and measurement options.

    Returns
    -------
        A `te.schedule` and the target `te.Tensor`s to be used in `tvm.lower` or `tvm.build`
    """
    if isinstance(search_policy, str):
        if search_policy == 'default':
            search_policy = EmptyPolicy()
        else:
            raise ValueError("Invalid search policy: " + search_policy)

    if tune_option is None:
        tune_option = TuneOption(n_trials=0)

    if isinstance(workload, str):
        sch, tensors = _ffi_api.AutoScheduleByWorkloadKey(
            workload, target, target_host, search_policy, hardware_params, tune_option)
        return sch, tensors
    if isinstance(workload, SearchTask):
        sch, tensors = _ffi_api.AutoScheduleBySearchTask(workload, search_policy, tune_option)
        return sch, tensors
    raise ValueError("Invalid workload: " + workload + ". Expect a string or SearchTask")
