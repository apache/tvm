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

from tempfile import mkdtemp
from collections import OrderedDict
import pickle
import os
import shutil
import tvm._ffi
from tvm.runtime import Object, ndarray
from tvm.target import Target
from tvm.driver import build_module
from .measure import LocalBuilder, LocalRunner, RPCRunner
from .workload_registry import make_workload_key
from .compute_dag import ComputeDAG
from .cost_model import XGBModel
from .search_policy import SketchPolicy
from .search_task import SearchTask
from .utils import get_const_tuple
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
    max_registers_per_block : int
        The max number of register per block.
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
        max_registers_per_block,
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
            max_registers_per_block,
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
    working_dir: Optional [string]
        Temp working directory for binary buffers
        If it is None, a random temp path will be generated
    check_correctness: Optional [bool]
        Whether generate an empty CPU schedule to check correctness
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during schedule search.
    builder: Union[ProgramBuilder, str] = 'local'
        ProgramBuilder which builds the program.
    builder_n_parallel: int = -1
        How many parallel job builder will run
        For Metal/ROCM, builder_n_parallel is recommended to set to 1
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
        working_dir=None,
        check_correctness=False,
        verbose=1,
        builder="local",
        builder_n_parallel=-1,
        runner="local",
        measure_callbacks=None,
    ):
        if working_dir is None:
            self.temp_working_dir = mkdtemp()
        else:
            self.temp_working_dir = working_dir
        if isinstance(builder, str):
            if builder == "local":
                builder = LocalBuilder(n_parallel=builder_n_parallel)
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
                runner = LocalRunner(working_dir=self.temp_working_dir)
            else:
                raise ValueError("Invalid runner: " + runner)

        elif isinstance(runner, RPCRunner):
            rpc_kwargs = runner.kwargs
            rpc_kwargs["working_dir"] = self.temp_working_dir
            runner = RPCRunner(**rpc_kwargs)
        elif not isinstance(runner, tvm.auto_scheduler.measure.ProgramRunner):
            raise ValueError(
                "Invalid runner: " + runner + " . TuningOptions expects a ProgramRunner or string."
            )

        self.__init_handle_by_constructor__(
            _ffi_api.TuningOptions,
            num_measure_trials,
            early_stopping or -1,
            num_measures_per_round,
            self.temp_working_dir,
            check_correctness,
            verbose,
            builder,
            runner,
            measure_callbacks,
        )

    def register_buffer(self, name, buffer):
        """Register numpy buffer for a given tensor argument name for running

        Parameters
        ----------
        name : string
            name of the argument
        buffer : numpy.ndarray
            data for the tensor argument
        """
        buffer_path = os.path.join(self.temp_working_dir, "buffer.pkl")
        buf = OrderedDict()
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as finput:
                buf = pickle.load(finput)
        buf[name] = buffer
        with open(buffer_path, "wb") as foutput:
            pickle.dump(buf, foutput)

    def __del__(self):
        shutil.rmtree(self.temp_working_dir)


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

    if tuning_options.check_correctness:
        null_sch, args = task.compute_dag.apply_steps_from_state(task.compute_dag.get_init_state())
        cpu_func = build_module.build(
            null_sch, args, target=task.target_host, target_host=task.target_host
        )
        buffer_path = os.path.join(tuning_options.working_dir, "buffer.pkl")
        if os.path.exists(buffer_path) is True:
            with open(buffer_path, "rb") as finput:
                buffer = pickle.load(finput)
            if len(buffer) == len(args):
                # we skip check each arg shape here
                pass
            elif len(buffer) == len(args) - 1:
                # assume only one output
                # TODO(xxx): get the output information from
                # `task.compute_dag` to support multiple output
                cpu_args = [v for _, v in buffer.items()] + [
                    ndarray.empty(args[-1].shape, dtype=args[-1].dtype, ctx=tvm.cpu())
                ]
                cpu_func(*cpu_args)
                ### save cpu result
                answer = [x.asnumpy() for x in cpu_args]
                tuning_options.register_buffer(args[-1].name, answer[-1])
        else:
            cpu_args = [ndarray.empty(get_const_tuple(x.shape), x.dtype, tvm.cpu()) for x in args]
            random_fill = tvm.get_global_func("tvm.contrib.random.random_fill", True)
            assert random_fill, "Please make sure USE_RANDOM is ON in the config.cmake"
            for arg in cpu_args:
                random_fill(arg)
            cpu_func(*cpu_args)
            answer = [arg.asnumpy() for arg in cpu_args]
            # pylint: disable=C0200
            for i in range(len(answer)):
                tuning_options.register_buffer(args[i].name, answer[i])

    sch, tensors = _ffi_api.AutoSchedule(search_policy, tuning_options)
    return sch, tensors
