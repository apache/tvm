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
Distributed measurement infrastructure to measure the runtime costs of tensor programs.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.

We separate the measurement into two steps: build and run.
A builder builds the executable binary files and a runner runs the binary files to
get the measurement results. The flow of data structures is

                `ProgramBuilder`                 `ProgramRunner`
`MeasureInput` -----------------> `BuildResult` ----------------> `MeasureResult`

We implement these in python to utilize python's multiprocessing and error handling.
"""

import os
import time
import shutil
import traceback
import tempfile
import multiprocessing

import tvm._ffi
from tvm.runtime import Object, module, ndarray
from tvm.driver import build_module
from tvm.ir import transform
from tvm.contrib import tar, ndk

from . import _ffi_api
from .utils import get_const_tuple, NoDaemonPool, call_func_with_timeout

# The maximum length of error message
MAX_ERROR_MSG_LEN = 512

# We use fork and a global variable to copy arguments between processings.
# This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
GLOBAL_BUILD_ARGUMENTS = None

@tvm._ffi.register_object("auto_scheduler.MeasureCallback")
class MeasureCallback(Object):
    """ The base class of measurement callback functions. """


@tvm._ffi.register_object("auto_scheduler.MeasureInput")
class MeasureInput(Object):
    """ Store the input of a measurement.

    Parameters
    ----------
    task : SearchTask
        The SearchTask of this measure.
    state : State
        The State to be measured.
    """
    def __init__(self, task, state):
        self.__init_handle_by_constructor__(_ffi_api.MeasureInput, task, state.state_object)


@tvm._ffi.register_object("auto_scheduler.BuildResult")
class BuildResult(Object):
    """ Store the result of a build.

    Parameters
    ----------
    filename : Optional[str]
        The filename of built binary file.
    args : List[Tensor]
        The arguments.
    error_no : int
        The error code.
    error_msg : Optional[str]
        The error message if there is any error.
    time_cost : float
        The time cost of build.
    """
    def __init__(self, filename, args, error_no, error_msg, time_cost):
        filename = filename if filename else ""
        error_msg = error_msg if error_msg else ""

        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult, filename, args, error_no, error_msg, time_cost)


@tvm._ffi.register_object("auto_scheduler.MeasureResult")
class MeasureResult(Object):
    """ Store the results of a measurement.

    Parameters
    ----------
    costs : List[float]
        The time costs of execution.
    error_no : int
        The error code.
    error_msg : Optional[str]
        The error message if there is any error.
    all_cost : float
        The time cost of build and run.
    timestamp : float
        The time stamps of this measurement.
    """
    def __init__(self, costs, error_no, error_msg, all_cost, timestamp):
        error_msg = error_msg if error_msg else ""

        self.__init_handle_by_constructor__(
            _ffi_api.MeasureResult, costs, error_no,
            error_msg, all_cost, timestamp)


@tvm._ffi.register_object("auto_scheduler.ProgramBuilder")
class ProgramBuilder(Object):
    """ The base class of ProgramBuilders. """

    def build(self, measure_inputs, verbose=1):
        """ Build programs and return results.

        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        verbose: int = 1
            Verbosity level. 0 for silent, 1 to output information during program building.

        Returns
        -------
        res : List[BuildResult]
        """
        return _ffi_api.ProgramBuilderBuild(self, measure_inputs, verbose)


@tvm._ffi.register_object("auto_scheduler.ProgramRunner")
class ProgramRunner(Object):
    """ The base class of ProgramRunners. """

    def run(self, measure_inputs, build_results, verbose=1):
        """ Run measurement and return results.

        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        build_results : List[BuildResult]
            A List of BuildResult to be ran.
        verbose: int = 1
            Verbosity level. 0 for silent, 1 to output information during program running.

        Returns
        -------
        res : List[MeasureResult]
        """
        return _ffi_api.ProgramRunnerRun(self, measure_inputs, build_results, verbose)


@tvm._ffi.register_object("auto_scheduler.LocalBuilder")
class LocalBuilder(ProgramBuilder):
    """ LocalBuilder use local CPU cores to build programs in parallel.

    Parameters
    ----------
    timeout : int = 15
        The timeout limit (in second) for each build thread.
        This is used in a wrapper of the multiprocessing.Process.join().
    n_parallel : int = multiprocessing.cpu_count()
        Number of threads used to build in parallel.
    build_func : str = 'default'
        The name of registered build function.
    """

    def __init__(self,
                 timeout=15,
                 n_parallel=multiprocessing.cpu_count(),
                 build_func='default'):
        self.__init_handle_by_constructor__(
            _ffi_api.LocalBuilder, timeout, n_parallel, build_func)


@tvm._ffi.register_object("auto_scheduler.LocalRunner")
class LocalRunner(ProgramRunner):
    """ LocalRunner that uses local CPU/GPU to measures the time cost of programs.

    Parameters
    ----------
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    """

    def __init__(self,
                 timeout=10,
                 number=3,
                 repeat=1,
                 min_repeat_ms=0,
                 cooldown_interval=0.0):
        self.__init_handle_by_constructor__(
            _ffi_api.LocalRunner, timeout, number, repeat, min_repeat_ms, cooldown_interval)


class MeasureErrorNo(object):
    """ Error type for MeasureResult. """
    NO_ERROR = 0              # No error
    INSTANTIATION_ERROR = 1   # Errors happen when apply transform steps from init state
                              # Errors happen when compiling code on host (e.g. tvm.build)
    COMPILE_HOST = 2
    COMPILE_DEVICE = 3        # Errors happen when compiling code on device
                              # (e.g. OpenCL JIT on the device)
    RUNTIME_DEVICE = 4        # Errors happen when run program on device
    WRONG_ANSWER = 5          # Answer is wrong when compared to a reference output
    BUILD_TIMEOUT = 6         # Timeout during compilation
    RUN_TIMEOUT = 7           # Timeout during run
    UNKNOWN_ERROR = 8         # Unknown error


def make_error_msg():
    """ Get the error message from traceback. """
    error_msg = str(traceback.format_exc())
    if len(error_msg) > MAX_ERROR_MSG_LEN:
        error_msg = error_msg[:MAX_ERROR_MSG_LEN//2] + \
            "\n...\n" + error_msg[-MAX_ERROR_MSG_LEN//2:]
    return error_msg


def local_build_worker(index):
    """
    Build function of LocalBuilder to be ran in the Builder thread pool.

    Parameters
    ----------
    index : int
        The MeasureInput index to be processed by the current Builder thread.

    Returns
    -------
    res : BuildResult
        The build result of this Builder thread.
    """
    global GLOBAL_BUILD_ARGUMENTS

    # We use fork and a global variable to copy arguments between processings.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    if not GLOBAL_BUILD_ARGUMENTS:
        raise ValueError("GLOBAL_BUILD_ARGUMENTS not found")
    measure_inputs, build_func, timeout, verbose = GLOBAL_BUILD_ARGUMENTS
    assert isinstance(build_func, str)

    if build_func == 'default':
        build_func = tar.tar
    elif build_func == 'ndk':
        build_func = ndk.create_shared
    else:
        raise ValueError("Invalid build_func" + build_func)

    def timed_func():
        tic = time.time()
        inp = measure_inputs[index]
        task = inp.task

        error_no = MeasureErrorNo.NO_ERROR
        error_msg = None
        args = []

        try:
            sch, args = task.compute_dag.apply_steps_from_state(
                inp.state)
        # pylint: disable=broad-except
        except Exception:
            error_no = MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = make_error_msg()

        if error_no == 0:
            dirname = tempfile.mkdtemp()
            filename = os.path.join(
                dirname, "tmp_func." + build_func.output_format)

            try:
                with transform.PassContext():  # todo(lmzheng): port the unroll pass
                    func = build_module.build(
                        sch, args, target=task.target, target_host=task.target_host)
                func.export_library(filename, build_func)
            # pylint: disable=broad-except
            except Exception:
                error_no = MeasureErrorNo.COMPILE_HOST
                error_msg = make_error_msg()
        else:
            filename = ""

        if verbose >= 1:
            if error_no == MeasureErrorNo.NO_ERROR:
                print(".", end="")
            else:
                print(".E", end="")  # Build error
        return filename, args, error_no, error_msg, time.time() - tic

    res = call_func_with_timeout(timeout, timed_func)
    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print(".T", end="")  # Build timeout
        res = None, [], MeasureErrorNo.BUILD_TIMEOUT, None, timeout

    return res


@tvm._ffi.register_func("auto_scheduler.local_builder.build")
def local_builder_build(inputs, timeout, n_parallel, build_func='default', verbose=1):
    """
    Build function of LocalBuilder to build the MeasureInputs to runnable modules.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be built.
    timeout : int
        The timeout limit (in second) for each build thread.
        This is used in a wrapper of the multiprocessing.Process.join().
    n_parallel : int
        Number of threads used to build in parallel.
    build_func : str = 'default'
        The name of build function to process the built module.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program building.

    Returns
    -------
    res : List[BuildResult]
        The build results of these MeasureInputs.
    """
    # We use fork and a global variable to copy arguments between processings.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    global GLOBAL_BUILD_ARGUMENTS

    GLOBAL_BUILD_ARGUMENTS = (inputs, build_func, timeout, verbose)

    pool = NoDaemonPool(n_parallel)
    tuple_res = pool.map(local_build_worker, range(len(inputs)))
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(BuildResult(*res))

    return results

@tvm._ffi.register_func("auto_scheduler.local_runner.run")
def local_run(inputs, build_results, timeout, number, repeat, min_repeat_ms, cooldown_interval,
              verbose=1):
    """
    Run function of LocalRunner to test the performance of the input BuildResults.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be measured.
    build_results : List[BuildResult]
        The BuildResults to be measured.
    timeout : int
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program measuring.

    Returns
    -------
    res : List[MeasureResult]
        The measure results of these MeasureInputs.
    """
    max_float = 1e10  # We use 1e10 instead of sys.float_info.max for better readability in log

    def timed_func(inp, build_res):
        tic = time.time()
        error_no = 0
        error_msg = None
        try:
            func = module.load_module(build_res.filename)
            ctx = ndarray.context(str(inp.task.target), 0)
            time_f = func.time_evaluator(
                func.entry_name, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
        # pylint: disable=broad-except
        except Exception:
            costs = (max_float,)
            error_no = MeasureErrorNo.COMPILE_DEVICE
            error_msg = make_error_msg()

        if error_no == 0:
            try:
                args = [ndarray.empty(get_const_tuple(x.shape), x.dtype, ctx) for x in
                        build_res.args]
                ctx.sync()
                costs = time_f(*args).results
            # pylint: disable=broad-except
            except Exception:
                costs = (max_float,)
                error_no = MeasureErrorNo.RUNTIME_DEVICE
                error_msg = make_error_msg()

        shutil.rmtree(os.path.dirname(build_res.filename))
        toc = time.time()
        time.sleep(cooldown_interval)

        if verbose >= 1:
            if error_no == MeasureErrorNo.NO_ERROR:
                print("*", end="")
            else:
                print("*E", end="")  # Run error
        return costs, error_no, error_msg, toc - tic + build_res.time_cost, toc

    measure_results = []
    assert len(inputs) == len(build_results), \
        "Measure input size should be equal to build results"
    for inp, build_res in zip(inputs, build_results):
        if build_res.error_no != 0:
            res = (max_float,), build_res.error_no, build_res.error_msg, build_res.time_cost, \
                time.time()
        else:
            res = call_func_with_timeout(
                timeout, timed_func, args=(inp, build_res))
            if isinstance(res, TimeoutError):
                if verbose >= 1:
                    print("*T", end="")  # Run timeout
                res = (max_float,), MeasureErrorNo.RUN_TIMEOUT, None, \
                    build_res.time_cost + timeout, time.time()
        measure_results.append(MeasureResult(*res))

    if verbose >= 1:
        print("")

    return measure_results
