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
# pylint: disable=invalid-name,too-many-function-args,too-many-nested-blocks
"""
Functions that run on executor for measurement.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.
"""

import contextlib
import logging
import os
import shutil
import tempfile
import threading
import time
import typing
from collections import namedtuple
from random import getrandbits
import warnings

import tvm._ffi
import tvm.ir.transform
from tvm import nd
from tvm import rpc as _rpc
from tvm.autotvm.env import AutotvmGlobalScope, reset_global_scope
from tvm.contrib import ndk, nvcc, stackvm, tar
from tvm.contrib.popen_pool import PopenPoolExecutor
from tvm.driver import build
from tvm.error import TVMError
from tvm.target import Target

from ..env import AutotvmGlobalScope
from ..task.space import InstantiationError
from ..utils import get_const_tuple
from .measure import Builder, MeasureErrorNo, MeasureResult, Runner

logger = logging.getLogger("autotvm")


class BuildResult(namedtuple("BuildResult", ("filename", "arg_info", "error", "time_cost"))):
    """
    Stores all the necessary inputs for a measurement.

    Parameters
    ----------
    filename : str
        The filename of generated library
    arg_info : Tuple
        The shape and dtype information of tvm tensor arguments
    error : Exception
        The error happens during compilation.
    time_cost : float
        The time cost of building
    """


class LocalBuilder(Builder):
    """Run compilation on local machine

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    build_kwargs: dict
        If supplied, additional kwargs passed to build_func. Overrides any build_kwargs supplied
        by the Runner.
    build_func: callable or str
        If is 'default', use default build function
        If is 'ndk', use function for android ndk
        If id 'stackvm', use function for stackvm
        If is callable, use it as custom build function, expect lib_format field.
    do_fork: bool
        If False, do not fork when building. Requires n_parallel=1.
    """

    def __init__(
        self, timeout=10, n_parallel=None, build_kwargs=None, build_func="default", do_fork=False
    ):
        super(LocalBuilder, self).__init__(timeout, n_parallel, build_kwargs)

        if isinstance(build_func, str):
            if build_func == "default":
                build_func = tar.tar
            elif build_func == "ndk":
                build_func = ndk.create_shared
            elif build_func == "stackvm":
                build_func = stackvm.build
            else:
                raise ValueError("Invalid build_func" + build_func)
        self.build_func = _WrappedBuildFunc(build_func)
        if not do_fork:
            assert n_parallel in (
                None,
                1,
            ), f"if do_fork=False, need n_parallel=None or 1; got {n_parallel}"
        self.executor = PopenPoolExecutor(
            timeout=timeout, initializer=reset_global_scope, initargs=(AutotvmGlobalScope.current,)
        )
        self.tmp_dir = tempfile.mkdtemp()

    def build(self, measure_inputs):
        results = []

        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i : i + self.n_parallel]:
                ret = self.executor.submit(self.build_func, inp, self.tmp_dir, **self.build_kwargs)
                futures.append(ret)

            for future in futures:
                try:
                    res = future.result()
                    if res.error is not None:
                        # instantiation error
                        if isinstance(res.error, InstantiationError):
                            res = MeasureResult(
                                (res.error,),
                                MeasureErrorNo.INSTANTIATION_ERROR,
                                res.time_cost,
                                time.time(),
                            )

                        else:
                            if "InstantiationError" in str(res.error):
                                msg = str(res.error)
                                try:
                                    msg = msg.split("\n")[-2].split(": ")[1]
                                except Exception:  # pylint: disable=broad-except
                                    pass
                                res = MeasureResult(
                                    (InstantiationError(msg),),
                                    MeasureErrorNo.INSTANTIATION_ERROR,
                                    res.time_cost,
                                    time.time(),
                                )

                            else:  # tvm error
                                res = MeasureResult(
                                    (res.error,),
                                    MeasureErrorNo.COMPILE_HOST,
                                    res.time_cost,
                                    time.time(),
                                )
                except TimeoutError as ex:
                    res = MeasureResult(
                        (ex,), MeasureErrorNo.BUILD_TIMEOUT, self.timeout, time.time()
                    )
                except ChildProcessError as ex:
                    res = MeasureResult(
                        (ex,),
                        MeasureErrorNo.RUNTIME_DEVICE,
                        self.timeout,
                        time.time(),
                    )

                results.append(res)

        return results


class RPCRunner(Runner):
    """Run generated code on remove devices.
    This function will ask a RPC Tracker to get device for measurement.

    Parameters
    ----------
    timeout: float
        The timeout of a RPCRunner measurement task
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    key: str
        The key of the device registered in the tracker
    host: str
        The host address of RPC Tracker
    port: int
        The port of RPC Tracker
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
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    module_loader : ModuleLoader
        If given, a context manager that loads the module to be timed into the remote runtime.
        If not given, default_module_loader is used.
    """

    def __init__(
        self,
        key,
        host,
        port,
        priority=1,
        timeout=10,
        n_parallel=None,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        enable_cpu_cache_flush=False,
        module_loader=None,
    ):
        super(RPCRunner, self).__init__(timeout, n_parallel)

        self.key = key
        self.host = host
        self.port = port
        self.priority = priority
        self.timeout = timeout

        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms
        self._ref_input = None

        self.enable_cpu_cache_flush = enable_cpu_cache_flush
        self.cooldown_interval = cooldown_interval
        self.module_loader = module_loader

        self.executor = PopenPoolExecutor(
            timeout=timeout * (self.n_parallel + 1),
            initializer=reset_global_scope,
            initargs=(AutotvmGlobalScope.current,),
        )

    @property
    def ref_input(self):
        """
        Fixed input for tuning special operators, e.g., sparse operators
        requiring indices as input.
        """
        return self._ref_input

    @ref_input.setter
    def ref_input(self, val):
        if val is not None:
            warnings.warn(
                "You are specifying fixed input for tuning the operator. "
                "Be sure your input always fits the operator. Some "
                "operators may conduct layout transformation during tuning, "
                "thus can lead to unexpected behaviors. ",
                RuntimeWarning,
            )
        self._ref_input = val

    def set_task(self, task):
        self.task = task

        if check_remote(task.target, self.key, self.host, self.port):
            logger.info("Get devices for measurement successfully!")
        else:
            raise RuntimeError(
                "Cannot get remote devices from the tracker. "
                "Please check the status of tracker by "
                "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                "and make sure you have free devices on the queue status."
            )

    def get_build_kwargs(self):
        kwargs = {}
        if (
            "cuda" in self.task.target.keys
            or "opencl" in self.task.target.keys
            or "rocm" in self.task.target.keys
            or "vulkan" in self.task.target.keys
        ):
            remote = request_remote(self.key, self.host, self.port)
            dev = remote.device(str(self.task.target), 0)
            max_dims = dev.max_thread_dimensions
            kwargs["check_gpu"] = {
                "max_shared_memory_per_block": dev.max_shared_memory_per_block,
                "max_threads_per_block": dev.max_threads_per_block,
                "max_thread_x": max_dims[0],
                "max_thread_y": max_dims[1],
                "max_thread_z": max_dims[2],
            }

            if "cuda" in self.task.target.keys:
                kwargs["cuda_arch"] = "sm_" + "".join(dev.compute_version.split("."))

        return kwargs

    def run(self, measure_inputs, build_results):
        results = []
        remote_kwargs = dict(
            device_key=self.key,
            host=self.host,
            port=self.port,
            priority=self.priority,
            timeout=self.timeout,
        )

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(
                measure_inputs[i : i + self.n_parallel], build_results[i : i + self.n_parallel]
            ):
                module_loader = (
                    self.module_loader
                    if self.module_loader is not None
                    else default_module_loader()
                )
                ret = self.executor.submit(
                    run_through_rpc,
                    measure_inp,
                    build_res,
                    self.number,
                    self.repeat,
                    self.min_repeat_ms,
                    self.cooldown_interval,
                    remote_kwargs,
                    self.ref_input,
                    self.enable_cpu_cache_flush,
                    module_loader,
                )
                futures.append(ret)

            for future in futures:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as ex:  # pylint: disable=broad-except
                    results.append(
                        MeasureResult(
                            (str(ex),), MeasureErrorNo.RUN_TIMEOUT, self.timeout, time.time()
                        )
                    )

        return results


class LocalRunner(RPCRunner):
    """Run generated code on local devices.

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
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
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    Note
    ----
    This is a "fake" local mode. We start a silent rpc tracker and rpc server
    for the user. In this way we reuse timeout/isolation mechanism in RPC infrastructure.
    """

    def __init__(
        self,
        timeout=10,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        enable_cpu_cache_flush=False,
        module_loader=None,
    ):
        super(LocalRunner, self).__init__(
            "",
            None,
            None,
            0,
            timeout=timeout,
            n_parallel=1,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            cooldown_interval=cooldown_interval,
            enable_cpu_cache_flush=enable_cpu_cache_flush,
            module_loader=module_loader,
        )
        self.tracker = None
        self.server = None

    def set_task(self, task):
        # pylint: disable=import-outside-toplevel
        from ...rpc.server import Server
        from ...rpc.tracker import Tracker

        self.task = task
        tracker = Tracker(port=9000, port_end=10000, silent=True)
        device_key = "$local$device$%d" % tracker.port
        server = Server(
            port=9000,
            port_end=10000,
            key=device_key,
            silent=True,
            tracker_addr=("127.0.0.1", tracker.port),
        )
        self.key = device_key
        self.host = "127.0.0.1"
        self.port = tracker.port

        super(LocalRunner, self).set_task(task)
        return server, tracker


def _build_func_common(measure_input, check_gpu=None, cuda_arch=None, build_option=None):
    """Common part for building a configuration"""
    target, task, config = measure_input
    target, task.target_host = Target.check_and_update_host_consist(target, task.target_host)

    with target:
        s, args = task.instantiate(config)

        # check invalidity of template and code hash consistency
        if not config.valid():
            raise InstantiationError(config.errors)

        opts = build_option or {}
        if check_gpu:  # Add verify pass to filter out invalid configs in advance.
            opts["tir.add_lower_pass"] = [(2, gpu_verify_pass(**check_gpu))]
        if cuda_arch:
            set_cuda_target_arch(cuda_arch)

        # if target is vta, we need to use vta build
        if (
            hasattr(measure_input.target, "device_name")
            and measure_input.target.device_name == "vta"
        ):
            # pylint: disable=import-outside-toplevel
            import vta

            func = vta.build(s, args, target_host=task.target_host)
        else:
            with tvm.ir.transform.PassContext(config=opts):
                func = build(s, args, target_host=task.target_host)
    return func, tuple((get_const_tuple(x.shape), x.dtype) for x in args)


class _WrappedBuildFunc:
    """
    Wrap build_func to a function that can be used in measure.

    Note: this is a class instead of a closure so that it can be pickled when
    using multiprocessing.

    Parameters
    ----------
    build_func : The compilation function
        We expect fcompile to contain an attr "output_format".

    Returns
    -------
    wrapped_build_func : callable
        The wrapped build function
    """

    def __init__(self, build_func):
        if not hasattr(build_func, "output_format"):
            raise AttributeError("Expect build_func to have the attribute output_format.")
        self.build_func = build_func

    def __call__(self, measure_input, tmp_dir, **kwargs):
        """
        Wrapped build func.

        Parameters
        ----------
        measure_input: MeasureInput
            The input of measurement

        tmp_dir: str
            The path of temporary directory to export generated library
        """
        tic = time.time()
        try:
            filename = os.path.join(
                tmp_dir, "tmp_func_%0x.%s" % (getrandbits(64), self.build_func.output_format)
            )
            # TODO(tvm-team) consider linline _build_func_common
            func, arg_info = _build_func_common(measure_input, **kwargs)
            if self.build_func.output_format == ".model-library-format":
                # Late import to preserve autoTVM with USE_MICRO OFF
                try:
                    from tvm import micro  # pylint: disable=import-outside-toplevel
                except ImportError:
                    raise ImportError("Requires USE_MICRO")

                micro.export_model_library_format(func, filename)
            else:
                func.export_library(filename, self.build_func)
        except Exception as e:  # pylint: disable=broad-except
            return BuildResult(None, None, e, time.time() - tic)
        return BuildResult(filename, arg_info, None, time.time() - tic)


ModuleLoader = typing.Callable[
    [dict, dict], typing.ContextManager[typing.Tuple[tvm.rpc.RPCSession, tvm.runtime.Module]]
]


def run_through_rpc(
    measure_input,
    build_result,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    remote_kwargs,
    ref_input,
    enable_cpu_cache_flush=False,
    module_loader=None,
):
    """Run a generated library through rpc

    Parameters
    ----------
    measure_input: MeasureInput
        The raw measure input
    build_result: BuildResult
        The result returned from Builder. This contains the path to the generated library.
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float
        The cool down interval between two measurements
    remote_kwargs: dict
        Passed to module_loader(). Ultimately, keyword args to request_remote().
    ref_input: List of np.ndarray
        The reference input used for tuning. Empty for randomly filled input.
    enable_cpu_cache_flush: bool
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    module_loader: ModuleLoader
        A function that returns a ContextManager used to establish and teardown the remote session.
    """
    if isinstance(build_result, MeasureResult):
        return build_result

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR
    try:
        # upload built module
        with module_loader(remote_kwargs, build_result) as (remote, mod):
            dev = remote.device(str(measure_input.target), 0)

            # Limitation:
            # We can not get PackFunction directly in the remote mode as it is wrapped
            # under the std::function. We could lift the restriction later once we fold
            # the PackedFunc as an object. Currently, we pass function name to work
            # around it.
            f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
            time_f = mod.time_evaluator(
                mod.entry_name,
                dev,
                number=number,
                repeat=repeat,
                min_repeat_ms=min_repeat_ms,
                f_preproc=f_prepare,
            )

            if ref_input:
                args = [nd.array(x, device=dev) for x in ref_input]
            else:
                try:
                    random_fill = remote.get_function("tvm.contrib.random.random_fill")
                except AttributeError:
                    raise AttributeError(
                        "Please make sure USE_RANDOM is ON in the config.cmake "
                        "on the remote devices"
                    )
                args = [nd.empty(x[0], x[1], dev) for x in build_result.arg_info]
                if "scatter" not in measure_input.task.name:
                    # the index tensor of scatter op cannot be randomly initialized
                    for arg in args:
                        random_fill(arg)
                dev.sync()

            costs = time_f(*args).results

        if len(costs) > 2:  # remove largest and smallest value to reduce variance
            costs = list(costs)
            costs.sort()
            costs = tuple(costs[1:-1])
    except TVMError as exc:
        msg = str(exc)
        if "Stack trace returned" in msg:
            msg = msg[: msg.index("Stack trace returned")]
        if "CUDA Source" in msg:
            msg = msg[: msg.index("CUDA Source")]
        costs = (RuntimeError(msg[:1024]),)
        errno = MeasureErrorNo.RUNTIME_DEVICE
    tstamp = time.time()
    time.sleep(cooldown_interval)
    return MeasureResult(costs, errno, tstamp - tic + build_result.time_cost, tstamp)


class DefaultModuleLoader:
    """See default_module_loader(). A pickleable emulation of the original function closure."""

    def __init__(self, pre_load_function=None) -> None:
        self.pre_load_function = pre_load_function

    @contextlib.contextmanager
    def __call__(self, remote_kwargs, build_result):
        remote = request_remote(**remote_kwargs)
        if self.pre_load_function is not None:
            self.pre_load_function(remote, build_result)

        remote.upload(build_result.filename)
        try:
            yield remote, remote.load_module(os.path.split(build_result.filename)[1])

        finally:
            # clean up remote files
            remote.remove(build_result.filename)
            remote.remove(os.path.splitext(build_result.filename)[0] + ".so")
            remote.remove("")


def default_module_loader(pre_load_function=None):
    """Returns a default function that can be passed as module_loader to run_through_rpc.

    Parameters
    ----------
    pre_load_function : Optional[Function[tvm.rpc.Session, tvm.runtime.Module]]
        Invoked after a session is established and before the default code-loading RPC calls are
        issued. Allows performing pre-upload actions, e.g. resetting the remote runtime environment.

    Returns
    -------
    DefaultModuleLoader :
        A callable that can be passed as module_loader to run_through_rpc.
    """

    # This was a function with a closure before but that couldn't be pickled!
    # We need pickle to work for using python's multiprocessing on some platforms.
    return DefaultModuleLoader(pre_load_function)


def request_remote(device_key, host=None, port=None, priority=1, timeout=60):
    """Request a remote session

    Parameters
    ----------
    device_key: string
        The device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this session (units: second)

    Returns
    ------
    session: RPCSession
    """
    # connect to the tracker
    host = host or os.environ["TVM_TRACKER_HOST"]
    port = port or int(os.environ["TVM_TRACKER_PORT"])

    tracker = _rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout)
    return remote


def check_remote(target, device_key, host=None, port=None, priority=100, timeout=10):
    """
    Check the availability of a remote device

    Parameters
    ----------
    target: Target
        The wanted compilation target
    device_key: string
        device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this check (units: seconds).

    Returns
    -------
    available: bool
        True if can find available device
    """

    def _check():
        remote = request_remote(device_key, host, port, priority)
        dev = remote.device(str(target))
        while not dev.exist:  # wait until we get an available device
            pass

    t = threading.Thread(
        target=_check,
    )
    t.start()
    t.join(timeout)
    return not t.is_alive()


@tvm._ffi.register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    curr_cuda_target_arch = AutotvmGlobalScope.current.cuda_target_arch
    # e.g., target arch could be [
    #   "-gencode", "arch=compute_52,code=sm_52",
    #   "-gencode", "arch=compute_70,code=sm_70"
    # ]
    target = "fatbin" if isinstance(curr_cuda_target_arch, list) else "ptx"
    ptx = nvcc.compile_cuda(code, target=target, arch=AutotvmGlobalScope.current.cuda_target_arch)
    return ptx


def set_cuda_target_arch(arch):
    """set target architecture of nvcc compiler

    Parameters
    ----------
    arch: str or list
        The argument of nvcc -arch. (e.g. "sm_51", "sm_62")
        it can also be a count of gencode arguments pass to nvcc command line,
        e.g., ["-gencode", "arch=compute_52,code=sm_52", "-gencode", "arch=compute_70,code=sm_70"]
    """
    AutotvmGlobalScope.current.cuda_target_arch = arch


def gpu_verify_pass(**kwargs):
    """Verify the validity of a gpu kernel.
    This pass will check memory usage and number of threads per block.
    """

    def verify_pass(f, *_):
        valid = tvm.tir.analysis.verify_gpu_code(f, kwargs)
        if not valid:
            raise InstantiationError("Skipped because of invalid gpu kernel")
        return f

    return tvm.tir.transform.prim_func_pass(verify_pass, opt_level=0)
