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
import enum
import multiprocessing
from collections import namedtuple


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

    def __repr__(self):
        error_no_str = (
            str(MeasureErrorNo(self.error_no))
            if isinstance(self.error_no, (MeasureErrorNo, int))
            else str(self.error_no)
        )
        return (
            f"{self.__class__.__name__}(costs={self.costs!r}, error_no={error_no_str}, "
            f"all_cost={self.all_cost}, timestamp={self.timestamp!r})"
        )


class MeasureErrorNo(enum.IntEnum):
    """Error type for MeasureResult"""

    NO_ERROR = 0  # no error
    INSTANTIATION_ERROR = 1  # actively detected error in instantiating a template with a config
    COMPILE_HOST = 2  # error when compiling code on host (e.g. tvm.build)
    COMPILE_DEVICE = 3  # error when compiling code on device (e.g. OpenCL JIT on the device)
    RUNTIME_DEVICE = 4  # error when run program on device
    WRONG_ANSWER = 5  # answer is wrong when compared to a golden output
    BUILD_TIMEOUT = 6  # timeout during compilation
    RUN_TIMEOUT = 7  # timeout during run
    UNKNOWN_ERROR = 8  # unknown error


class Builder(object):
    """Builder that builds programs in tuning

    Parameters
    ----------
    timeout: float, optional
        The timeout of a build task
    n_parallel: int, optional
        The number of tasks submitted in parallel
        By default it will use all cpu cores
    build_kwargs: dict, optional
        Keyword args given to the build function.
    """

    def __init__(self, timeout=10, n_parallel=None, build_kwargs=None):
        self.timeout = timeout
        self.n_parallel = n_parallel or multiprocessing.cpu_count()
        self.user_build_kwargs = build_kwargs if build_kwargs is not None else {}
        self.runner_build_kwargs = None
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
        self.build_kwargs = dict(build_kwargs.items()) if build_kwargs is not None else {}
        if any(k in self.build_kwargs for k in self.user_build_kwargs):
            logging.warn(
                "Overriding these runner-supplied kwargs with user-supplied:\n%s",
                "\n".join(
                    f" * {k}: from {build_kwargs[k]!r} to {self.user_build_kwargs[k]!r}"
                    for k in sorted([k for k in build_kwargs if k in self.user_build_kwargs])
                ),
            )
        for k, v in self.user_build_kwargs.items():
            self.build_kwargs[k] = v

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
    """Runner that runs and measures the time cost of a generated program in tuning

    Parameters
    ----------
    timeout: float, optional
        The timeout of a build task
    n_parallel: int, optional
        The number of tasks submitted in parallel
        By default it will use all cpu cores
    """

    def __init__(self, timeout=5, n_parallel=None):
        self.timeout = timeout
        self.n_parallel = n_parallel or multiprocessing.cpu_count()
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

    def get_build_kwargs(self):
        """
        Get device specific build arguments (e.g. maximum shared memory size)

        Returns
        ----------
        kwargs: dict
            The additional keyword arguments
        """
        raise NotImplementedError()

    def run(self, measure_inputs, build_results):
        """Run amd measure built programs

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
        raise NotImplementedError()


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
        if builder == "local":
            builder = LocalBuilder()
        else:
            raise ValueError("Invalid builder: " + builder)

    if isinstance(runner, str):
        if runner == "local":
            runner = LocalRunner()
        else:
            raise ValueError("Invalid runner: " + runner)

    opt = {
        "builder": builder,
        "runner": runner,
    }

    return opt


def create_measure_batch(task, option):
    """Get a standard measure_batch function.

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
    builder = option["builder"]
    runner = option["runner"]

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
