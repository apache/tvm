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

"""Distributed measurement infrastructure to measure the runtime costs of tensor programs

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.

We implement these in python to utilize python's multiprocessing and error handling
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

@tvm._ffi.register_object("ansor.MeasureCallback")
class MeasureCallback(Object):
    """Base class for measurement callback function"""


@tvm._ffi.register_object("ansor.MeasureInput")
class MeasureInput(Object):
    """
    Parameters
    ----------
    task : SearchTask
        The target SearchTask.
    state : State
        The current State to be measured.
    """

    def __init__(self, task, state):
        self.__init_handle_by_constructor__(_ffi_api.MeasureInput, task, state.state_object)


@tvm._ffi.register_object("ansor.BuildResult")
class BuildResult(Object):
    """ Store the input of a build.

    Parameters
    ----------
    filename : Str
        The filename of built binary file.
    args : List[Tensor]
        The arguments.
    error_no : Int
        The error code.
    error_msg : Str
        The error message if there is any error.
    time_cost : Float
        The time cost of build.
    """

    def __init__(self, filename, args, error_no, error_msg, time_cost):
        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult, filename if filename else "", args, error_no,
            error_msg if error_msg else "", time_cost)


@tvm._ffi.register_object("ansor.MeasureResult")
class MeasureResult(Object):
    """
    Parameters
    ----------
    costs : List[Float]
        The time costs of execution.
    error_no : Int
        The error code.
    error_msg : Str
        The error message if there is any error.
    all_cost : Float
        The time cost of build and run.
    timestamp : Float
        The time stamps of this measurement.
    """

    def __init__(self, costs, error_no, error_msg, all_cost, timestamp):
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureResult, costs, error_no,
            error_msg if error_msg else "", all_cost, timestamp)


@tvm._ffi.register_object("ansor.Builder")
class Builder(Object):
    """ Base class of Builder """
    def build(self, measure_inputs, verbose=1):
        """
        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        verbost : Int
            Verbosity level. (0 means silent)

        Returns
        -------
        res : List[BuildResult]
        """
        return _ffi_api.BuilderBuild(self, measure_inputs, verbose)


@tvm._ffi.register_object("ansor.Runner")
class Runner(Object):
    """ Base class of Runner """
    def run(self, measure_inputs, build_results, verbose=1):
        """
        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        build_results : List[BuildResult]
            A List of BuildResult to be ran.

        Returns
        -------
        res : List[MeasureResult]
        """
        return _ffi_api.RunnerRun(self, measure_inputs, build_results, verbose)


@tvm._ffi.register_object("ansor.LocalBuilder")
class LocalBuilder(Builder):
    """ LocalBuilder use local CPU cores to build programs in parallel.

    Parameters
    ----------
    timeout : Int
        The timeout limit for each build.
    n_parallel : Int
        Number of threads used to build in parallel.
    build_func : Str
        The name of registered build function.
    """

    def __init__(self,
                 timeout=15,
                 n_parallel=multiprocessing.cpu_count(),
                 build_func='default'):
        self.__init_handle_by_constructor__(
            _ffi_api.LocalBuilder, timeout, n_parallel, build_func)


@tvm._ffi.register_object("ansor.LocalRunner")
class LocalRunner(Runner):
    """ LocalRunner that uses local CPU/GPU to measures the time cost of programs.

    Parameters
    ----------
    timeout : Int
        The timeout limit for each run.
    number : Int
        Number of measure times.
    repeat : Int
        Number of repeat times in each measure.
    min_repeat_ms : Int
        The minimum duration of one repeat in milliseconds.
    cooldown_interval : Float
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
    """Error type for MeasureResult"""
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
    """ Get the error message from traceback """
    error_msg = str(traceback.format_exc())
    if len(error_msg) > MAX_ERROR_MSG_LEN:
        error_msg = error_msg[:MAX_ERROR_MSG_LEN//2] + \
            "\n...\n" + error_msg[-MAX_ERROR_MSG_LEN//2:]
    return error_msg


GLOBAL_BUILD_ARGUMENTS = None
GLOBAL_RUN_ARGUMENTS = None


def local_build_worker(index):
    """ Local builder function """
    # We use fork to copy arguments from a global variable.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
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
        # pylint: disable=W0703
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
            # pylint: disable=W0703
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


@tvm._ffi.register_func("ansor.local_builder.build")
def local_builder_build(inputs, timeout, n_parallel, build_func, verbose):
    """ Local builder build function """
    # We use fork to copy arguments from a global variable.
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

@tvm._ffi.register_func("ansor.local_runner.run")
def local_run(inputs, build_results, timeout, number, repeat, min_repeat_ms, cooldown_interval,
              verbose):
    """ Local runner run function """
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
        # pylint: disable=W0703
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
            # pylint: disable=W0703
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
