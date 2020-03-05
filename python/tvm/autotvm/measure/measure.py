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
# pylint: disable=pointless-string-statement,consider-using-enumerate,invalid-name
"""User facing API for specifying how to measure the generated code"""
import multiprocessing
import time
from collections import namedtuple

import numpy as np

from tvm import nd
from tvm import target as _target
from tvm.driver import build

from ..util import get_const_tuple
from .local_executor import LocalExecutor


class MeasureInput(namedtuple("MeasureInput", ["target", "task", "config"])):
    """
    Stores all the necessary inputs for a measurement.

    Parameters
    ----------
    target : tvm.target.Target
        The target device
    task : task.Task
        Task function
    config : ConfigEntity
        Specific configuration.
    """


class MeasureResult(namedtuple("MeasureResult", ["costs", "error_no", "all_cost", "timestamp"])):
    """
    Stores all the results of a measurement

    Parameters
    ----------
    costs: Array of float or Array of Exception
        If no error occurs during measurement, it is an array of measured running times.
        If an error occurs during measurement, it is an array of the exception objections.
    error_no: int
        Denote error type, defined by MeasureErrorNo
    all_cost: float
        All cost of this measure, including rpc, compilation, test runs
    timestamp: float
        The absolute time stamp when we finish measurement.
    """


class MeasureErrorNo(object):
    """Error type for MeasureResult"""
    NO_ERROR = 0              # no error
    INSTANTIATION_ERROR = 1   # actively detected error in instantiating a template with a config
    COMPILE_HOST = 2          # error when compiling code on host (e.g. tvm.build)
    COMPILE_DEVICE = 3        # error when compiling code on device (e.g. OpenCL JIT on the device)
    RUNTIME_DEVICE = 4        # error when run program on device
    WRONG_ANSWER = 5          # answer is wrong when compared to a golden output
    BUILD_TIMEOUT = 6         # timeout during compilation
    RUN_TIMEOUT = 7           # timeout during run
    UNKNOWN_ERROR = 8         # unknown error


class Builder(object):
    """
    Builder that builds programs in tuning

    Parameters
    ----------
    timeout: float, optional
        The timeout of a build task
    n_parallel: int, optional
        The number of tasks submitted in parallel
        By default it will use all cpu cores
    """
    def __init__(self, timeout=10, n_parallel=None):
        self.timeout = timeout
        self.n_parallel = n_parallel or multiprocessing.cpu_count()
        self.build_kwargs = {}
        self.task = None

    def set_task(self, task, build_kwargs=None):
        """
        Initialize for a new tuning task

        Parameters
        ----------
        task: Task
            The tuning task
        build_kwargs: dict, optional
            The additional kwargs for build function
        """
        self.task = task
        self.build_kwargs = build_kwargs

    def build(self, measure_inputs):
        """Build programs

        Parameters
        ----------
        measure_inputs: List of MeasureInput
            The measure input

        Returns
        -------
        build_results: List of BuildResult
            The build result.
        """
        raise NotImplementedError()


class Runner(object):
    """
    Runner that runs and measures the time cost of a generated program in tuning

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    check_correctness: bool, optional
        Whether check correctness after measurement. This will use llvm cpu target to
        call your template and get the reference output.
        This can work for TOPI templates, but may not work for your custom template.
    """
    def __init__(self,
                 timeout=5,
                 n_parallel=None,
                 number=4,
                 repeat=3,
                 min_repeat_ms=0,
                 cooldown_interval=0.1,
                 check_correctness=False):
        self.timeout = timeout
        self.n_parallel = n_parallel or multiprocessing.cpu_count()
        self.timeout = timeout

        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms

        self.ref_input = None
        self.ref_output = None
        self.check_correctness = check_correctness
        self.cooldown_interval = cooldown_interval

        self.executor = LocalExecutor()

        self.task = None

    def set_task(self, task):
        """
        Initialize for a new tuning task

        Parameters
        ----------
        task: Task
            The tuning task
        """
        self.task = task
        if self.check_correctness:
            # use llvm cpu to generate a reference input/output
            # this option works for tuning topi, but might not work for your custom ops
            with _target.create("llvm"):
                s, arg_bufs = task.instantiate(task.config_space.get(0))
            self.ref_input = [np.random.uniform(size=get_const_tuple(x.shape)).astype(x.dtype)
                              for x in arg_bufs]
            func = build(s, arg_bufs, "llvm")
            tvm_buf = [nd.array(x) for x in self.ref_input]
            func(*tvm_buf)
            self.ref_output = [x.asnumpy() for x in tvm_buf]

    def get_device_context(self):
        """
        Get device context for build and run.

        Returns
        -------
        ctx: TVMContext
            The TVM context of the target device.
        """
        raise NotImplementedError()

    def get_build_kwargs(self):
        """
        Get device specific build arguments (e.g. maximum shared memory size).

        Returns
        ----------
        kwargs: dict
            The additional keyword arguments
        """
        kwargs = {}
        if ('cuda' in self.task.target.keys or 'opencl' in self.task.target.keys
                or 'rocm' in self.task.target.keys):
            ctx = self.get_device_context()
            max_dims = ctx.max_thread_dimensions
            kwargs['check_gpu'] = {
                'max_shared_memory_per_block': ctx.max_shared_memory_per_block,
                'max_threads_per_block': ctx.max_threads_per_block,
                'max_thread_x': max_dims[0],
                'max_thread_y': max_dims[1],
                'max_thread_z': max_dims[2],
            }

            if 'cuda' in self.task.target.keys:
                kwargs["cuda_arch"] = "sm_" + "".join(ctx.compute_version.split('.'))

        return kwargs

    def get_run_args(self, measure_inputs, build_results):
        """
        Get arguments for running the built binaries.

        Parameters
        ----------
        measure_inputs: List of MeasureInput
            The raw measure input
        build_results: List of BuildResults
            The build results

        Returns
        -------
        args: list
            A list of running arguments. The first element in the list is the running function.
        """
        raise NotImplementedError()

    def run(self, measure_inputs, build_results):
        """
        Run amd measure built programs

        Parameters
        ----------
        measure_inputs: List of MeasureInput
            The raw measure input
        build_results: List of BuildResults
            The build results

        Returns
        -------
        measure_results: List of MeasureResult
            The final results of measurement
        """
        results = []

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(measure_inputs[i:i + self.n_parallel],
                                              build_results[i:i + self.n_parallel]):
                args = self.get_run_args(measure_inp, build_res)
                ret = self.executor.submit(*args)
                futures.append(ret)

            for future in futures:
                res = future.get()
                if isinstance(res, Exception):  # executor error or timeout
                    results.append(
                        MeasureResult((str(res), ), MeasureErrorNo.RUN_TIMEOUT, self.timeout,
                                      time.time()))
                else:
                    results.append(res)

        return results


def measure_option(builder, runner):
    """
    Set options for measure. To measure a config, we will build it and run it.
    So we have to set options for these two steps.
    They have their own options on timeout, parallel, etc.

    Parameters
    ----------
    builder: Builder
        Specify how to build programs
    runner: Runner
        Specify how to run programs

    Examples
    --------
    # example setting for using local devices
    >>> measure_option = autotvm.measure_option(
    >>>     builder=autotvm.LocalBuilder(),      # use all local cpu cores for compilation
    >>>     runner=autotvm.LocalRunner(          # measure them sequentially
    >>>         number=10,
    >>>         timeout=5)
    >>> )

    # example setting for using remote devices
    >>> measure_option = autotvm.measure_option(
    >>>    builder=autotvm.LocalBuilder(),  # use all local cpu cores for compilation
    >>>    runner=autotvm.RPCRunner(
    >>>        'rasp3b', 'locahost', 9190, # device key, host and port of the rpc tracker
    >>>        number=4,
    >>>        timeout=4) # timeout of a run on the device. RPC request waiting time is excluded.
    >>>)

    Note
    ----
    To make measurement results accurate, you should pick the correct value for the argument
    `number` and `repeat` in Runner(). Some devices need a certain minimum running time to
    "warm up," such as GPUs that need time to reach a performance power state.
    Using `min_repeat_ms` can dynamically adjusts `number`, so it is recommended.
    The typical value for NVIDIA GPU is 150 ms.
    """
    # pylint: disable=import-outside-toplevel
    from .measure_methods import LocalBuilder, LocalRunner

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

    opt = {
        'builder': builder,
        'runner': runner,
    }

    return opt


def create_measure_batch(task, option):
    """
    Get a standard measure_batch function.

    Parameters
    ----------
    task: tvm.autotvm.task.Task
        The tuning task
    option: dict
        The option for measuring generated code.
        You should use the return value of function :any:`measure_option` for this argument.

    Returns
    -------
    measure_batch: callable
        a callback function to measure a batch of configs
    """
    builder = option['builder']
    runner = option['runner']

    attach_objects = runner.set_task(task)

    # feed device related information from runner to builder
    # (e.g. max shared memory for validity checking)
    build_kwargs = runner.get_build_kwargs()
    builder.set_task(task, build_kwargs)

    def measure_batch(measure_inputs):
        build_results = builder.build(measure_inputs)
        results = runner.run(measure_inputs, build_results)
        return results

    measure_batch.n_parallel = builder.n_parallel
    measure_batch.attach_objects = attach_objects
    return measure_batch
