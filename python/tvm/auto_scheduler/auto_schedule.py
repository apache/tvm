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

import random

import tvm._ffi
from tvm.runtime import Object
from .measure import LocalBuilder, LocalRunner
from .cost_model import RandomModel
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

    Parameters
    ----------
    task : SearchTask
        The SearchTask for the computation declaration.
    init_search_callbacks : Optional[List[SearchCallback]]
        Callback functions called before the search process.
    """
    def __init__(self, task, init_search_callbacks=None):
        self.__init_handle_by_constructor__(_ffi_api.EmptyPolicy, task, init_search_callbacks)


@tvm._ffi.register_object("auto_scheduler.SketchPolicy")
class SketchPolicy(SearchPolicy):
    """  The search policy that searches in a hierarchical search space defined by sketches.
    The policy randomly samples programs from the space defined by sketches
    and use evolutionary search to fine-tune them.

    Parameters
    ----------
    task : SearchTask
        The SearchTask for the computation declaration.
    schedule_cost_model : CostModel = RandomModel()
        The cost model to estimate the complete schedules.
    params : Optional[Dict[str, Any]]
        Parameters of the search policy.
        See `src/auto_scheduler/search_policy/sketch_search_policy.h` for the definitions.
        See `DEFAULT_PARAMS` below to find the default values.
    seed : Optional[int]
        Random seed.
    verbose : int = 1
        Verbosity level. 0 for silent, 1 to output information during schedule search.
    init_search_callbacks : Optional[List[SearchCallback]]
        Callback functions called before the search process, usually used to do extra
        initializations.
        Possible callbacks:
            - auto_scheduler.PreloadMeasuredStates
            - auto_scheduler.PreloadCustomSketchRule
            TODO(jcf94): Add these search callback implementations.
    """

    DEFAULT_PARAMS = {
        "eps_greedy": 0.05,

        'evolutionary_search_population': 2048,
        "evolutionary_search_use_measured_ratio": 0.2,

        'cpu_multi_level_tiling_structure': 'SSRSRS',
        'gpu_multi_level_tiling_structure': 'SSSRRSRS',

        'max_innermost_split_factor': 16,
        'max_vectorize_size': 16,

        'disable_change_compute_location': 0,
    }

    def __init__(self, task, schedule_cost_model=RandomModel(), params=None, seed=None, verbose=1,
                 init_search_callbacks=None):
        if params is None:
            params = SketchPolicy.DEFAULT_PARAMS
        else:
            for key, value in SketchPolicy.DEFAULT_PARAMS.items():
                if key not in params:
                    params[key] = value

        self.__init_handle_by_constructor__(
            _ffi_api.SketchPolicy, task, schedule_cost_model, params,
            seed or random.randint(1, 1 << 30), verbose, init_search_callbacks)

    def generate_sketches(self, print_for_debug=False):
        """ Generate the sketches, this is mainly used for debug.

        Parameters
        ----------
        print_for_debug : bool = False
            Whether print out the sketches for debug.

        Returns
        -------
        sketches : List[State]
            The generated sketches of this search task.
        """
        sketches = _ffi_api.SketchPolicyGenerateSketches(self)
        if print_for_debug:
            for i, s in enumerate(sketches):
                print("=" * 20 + " %d " % i + "=" * 20)
                print(s)
        return sketches

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
    """
    def __init__(self, num_measure_trials=0, early_stopping=None, num_measures_per_round=64,
                 verbose=1, builder='local', runner='local', measure_callbacks=None):
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
            _ffi_api.TuningOptions, num_measure_trials, early_stopping or -1,
            num_measures_per_round, verbose, builder, runner, measure_callbacks)


def auto_schedule(task, search_policy=None, tuning_options=TuningOptions()):
    """ Do auto scheduling for a computation declaration.

    Parameters
    ----------
    task : SearchTask
        The SearchTask for the computation declaration.
    search_policy : Optional[SearchPolicy]
        The search policy to be used for schedule search. Use EmptyPolicy as default, which always
        returns an empty schedule.
    tuning_options : Optional[TuningOptions]
        Tuning and measurement options.

    Returns
    -------
        A `te.schedule` and the a list of `te.Tensor` to be used in `tvm.lower` or `tvm.build`.
    """
    if not isinstance(task, SearchTask):
        raise ValueError("Invalid task: " + task +
                         " . `auto_scheduler.auto_schedule` expects a SearchTask.")

    sch, tensors = _ffi_api.AutoSchedule(search_policy or EmptyPolicy(task), tuning_options)
    return sch, tensors
