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
from typing import List
import os
import time
import shutil
import logging
import traceback
import tempfile
import multiprocessing

import tvm._ffi
from tvm.runtime import Object, module, ndarray
from tvm.driver import build_module
from tvm.target import build_config
from ..contrib import tar, ndk
from .utils import get_const_tuple, NoDaemonPool, call_func_with_timeout, request_remote, check_remote
from .compute_dag import LayoutRewriteLevel
from . import _ffi_api

logger = logging.getLogger('ansor')


@tvm._ffi.register_object("ansor.MeasureCallback")
class MeasureCallback(Object):
    pass

@tvm._ffi.register_object("ansor.MeasureInput")
class MeasureInput(Object):
    """
    Parameters
    ----------
    task : SearchTask
    state : State
    """

    def __init__(self, task, state):
        self.__init_handle_by_constructor__(_ffi_api.MeasureInput, task, state)


@tvm._ffi.register_object("ansor.BuildResult")
class BuildResult(Object):
    """
    Parameters
    ----------
    filename : Str
    args : List[Tensor]
    error_no : Int
    error_msg : Str
    time_cost : Float
    """

    def __init__(self, filename, args, error_no, error_msg, time_cost):
        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult, filename, args, error_no,
            error_msg if error_msg else "", time_cost)


@tvm._ffi.register_object("ansor.MeasureResult")
class MeasureResult(Object):
    """
    Parameters
    ----------
    costs : List[Float]
    error_no : Int
    error_msg : Str
    all_cost : Float
    timestamp : Float
    """

    def __init__(self, costs, error_no, error_msg, all_cost, timestamp):
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureResult, costs, error_no,
            error_msg if error_msg else "", all_cost, timestamp)


@tvm._ffi.register_object("ansor.Builder")
class Builder(Object):
    def build(self, measure_inputs, verbose=0):
        """
        Parameters
        ----------
        measure_inputs : List[MeasureInput]
        verbost : Int

        Returns
        -------
        res : List[BuildResult]
        """
        return _ffi_api.BuilderBuild(self, measure_inputs, verbose)


@tvm._ffi.register_object("ansor.Runner")
class Runner(Object):
    def run(self, measure_inputs, build_results, verbose=0):
        """
        Parameters
        ----------
        measure_inputs : List[MeasureInput]
        build_results : List[BuildResult]

        Returns
        -------
        res : List[MeasureResult]
        """
        return _ffi_api.RunnerRun(self, measure_inputs, build_results, verbose)


@tvm._ffi.register_object("ansor.LocalBuilder")
class LocalBuilder(Builder):
    """
    Parameters
    ----------
    timeout : Int
    n_parallel : Int
    build_func : Str
    """

    def __init__(self,
                 timeout=15,
                 n_parallel=multiprocessing.cpu_count(),
                 build_func='default'):
        self.__init_handle_by_constructor__(
            _ffi_api.LocalBuilder, timeout, n_parallel, build_func)


@tvm._ffi.register_object("ansor.LocalRunner")
class LocalRunner(Runner):
    """
    Parameters
    ----------
    timeout : Int
    number : Int
    repeat : Int
    min_repeat_ms : Int
    cooldown_interval : Float
    """

    def __init__(self,
                 timeout=10,
                 number=3,
                 repeat=1,
                 min_repeat_ms=0,
                 cooldown_interval=0.0):
        self.__init_handle_by_constructor__(
            _ffi_api.LocalRunner, timeout, number, repeat, min_repeat_ms, cooldown_interval)


MAX_ERROR_MSG_LEN = 512


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
    error_msg = str(traceback.format_exc())
    if len(error_msg) > MAX_ERROR_MSG_LEN:
        error_msg = error_msg[:MAX_ERROR_MSG_LEN//2] + \
            "\n...\n" + error_msg[-MAX_ERROR_MSG_LEN//2:]
    return error_msg


global global_build_arguments
global global_run_arguments


def local_build_worker(index):
    # We use fork to copy arguments from a global variable.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    measure_inputs, build_func, timeout, verbose = global_build_arguments
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
                inp.state, LayoutRewriteLevel.BOTH_REWRITE)
        except Exception:
            error_no = MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = make_error_msg()

        if error_no == 0:
            dirname = tempfile.mkdtemp()
            filename = os.path.join(
                dirname, "tmp_func." + build_func.output_format)

            try:
                with build_config(unroll_max_extent=task.hardware_params.max_unroll_vec):
                    func = build_module.build(
                        sch, args, target=task.target, target_host=task.target_host)
                func.export_library(filename, build_func)
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
def local_builder_build(inputs: List[MeasureInput], timeout: float, n_parallel: int, build_func: str, verbose: int):
    # We use fork to copy arguments from a global variable.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    global global_build_arguments
    global_build_arguments = (inputs, build_func, timeout, verbose)

    pool = NoDaemonPool(n_parallel)
    tuple_res = pool.map(local_build_worker, range(len(inputs)))
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(BuildResult(*res))

    return results


@tvm._ffi.register_func("ansor.rpc_runner.run")
def rpc_runner_run(inputs: List[MeasureInput], build_results: List[BuildResult],
                   key: str, host: str, port: int, priority: int, timeout: float,
                   n_parallel: int, number: int, repeat: int, min_repeat_ms: int,
                   cooldown_interval: float, verbose: int):
    global global_run_arguments
    global_run_arguments = (inputs, build_results, key, host, port, priority, timeout, number,
                            repeat, min_repeat_ms, cooldown_interval, verbose)

    assert len(inputs) == len(build_results), \
        "Measure input size should be equal to build results"
    pool = NoDaemonPool(n_parallel)
    tuple_res = pool.map(rpc_run_worker, range(len(build_results)))
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(MeasureResult(*res))

    if verbose >= 1:
        print("")

    return results


def rpc_run_worker(index):
    inputs, build_results, key, host, port, priority, timeout, number, \
        repeat, min_repeat_ms, cooldown_interval, verbose = global_run_arguments

    MAX_FLOAT = 1e10  # We use 1e10 instead of sys.float_info.max for better readability in log
    inp = inputs[index]
    build_res = build_results[index]

    if build_res.error_no != MeasureErrorNo.NO_ERROR:
        return (MAX_FLOAT,), build_res.error_no, build_res.error_msg, build_res.time_cost, time.time()

    def timed_func():
        tic = time.time()
        error_no = 0
        error_msg = None
        try:
            # upload built module
            remote = request_remote(key, host, port, priority, timeout)
            remote.upload(build_res.filename)
            func = remote.load_module(os.path.split(build_res.filename)[1])
            ctx = remote.context(str(inp.task.target), 0)
            time_f = func.time_evaluator(
                func.entry_name, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
        except Exception:
            costs = (MAX_FLOAT,)
            error_no = MeasureErrorNo.COMPILE_DEVICE
            error_msg = make_error_msg()

        if error_no == 0:
            try:
                args = [ndarray.non_empty(get_const_tuple(x.shape), x.dtype, ctx) for x in
                        build_res.args]
                ctx.sync()

                costs = time_f(*args).results
                # clean up remote files
                remote.remove(build_res.filename)
                remote.remove(os.path.splitext(build_res.filename)[0] + '.so')
                remote.remove('')
            except Exception:
                costs = (MAX_FLOAT,)
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

    res = call_func_with_timeout(timeout, timed_func)

    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print("*T", end="")  # Run timeout
        res = (MAX_FLOAT,), MeasureErrorNo.RUN_TIMEOUT, None, build_res.time_cost + \
            timeout, time.time()
    return res


@tvm._ffi.register_func("ansor.local_runner.run")
def local_run(inputs: List[MeasureInput], build_results: List[BuildResult],
              timeout: float, number: int, repeat: int, min_repeat_ms: int,
              cooldown_interval: float, verbose: int):
    MAX_FLOAT = 1e10  # We use 1e10 instead of sys.float_info.max for better readability in log

    def timed_func(inp, build_res):
        tic = time.time()
        error_no = 0
        error_msg = None
        try:
            func = module.load_module(build_res.filename)
            ctx = ndarray.context(str(inp.task.target), 0)
            time_f = func.time_evaluator(
                func.entry_name, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
        except Exception:
            costs = (MAX_FLOAT,)
            error_no = MeasureErrorNo.COMPILE_DEVICE
            error_msg = make_error_msg()

        if error_no == 0:
            try:
                args = [ndarray.non_empty(get_const_tuple(x.shape), x.dtype, ctx) for x in
                        build_res.args]
                ctx.sync()

                costs = time_f(*args).results
            except Exception:
                costs = (MAX_FLOAT,)
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
            res = (
                MAX_FLOAT,), build_res.error_no, build_res.error_msg, build_res.time_cost, time.time()
        else:
            res = call_func_with_timeout(
                timeout, timed_func, args=(inp, build_res))
            if isinstance(res, TimeoutError):
                if verbose >= 1:
                    print("*T", end="")  # Run timeout
                res = (
                    MAX_FLOAT,), MeasureErrorNo.RUN_TIMEOUT, None, build_res.time_cost + timeout, time.time()
        measure_results.append(MeasureResult(*res))

    if verbose >= 1:
        print("")

    return measure_results
