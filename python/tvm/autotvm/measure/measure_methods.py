# pylint: disable=invalid-name,too-many-function-args,too-many-nested-blocks
"""
Functions that run on executor for measurement.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.
"""

import logging
import shutil
import os
import threading
import time
from random import getrandbits
from collections import namedtuple
import tempfile

import numpy as np

from ... import ir_pass, build, build_config, nd, TVMError, register_func, \
    rpc as _rpc, target as _target
from ...contrib import nvcc, ndk

from ..util import get_const_tuple
from ..env import AutotvmGlobalScope
from ..task.space import InstantiationError

from .measure import MeasureResult, MeasureErrorNo, Builder, Runner
from .local_executor import LocalExecutor

logger = logging.getLogger('autotvm')

class BuildResult(namedtuple("BuildResult", ('filename', 'arg_info', 'error', 'time_cost'))):
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
    build_func: callable or str
        If is 'default', use default build function
        If is 'ndk', use function for android ndk
        If is callable, use it as custom build function
    """
    def __init__(self, timeout=10, n_parallel=None, build_func='default'):
        super(LocalBuilder, self).__init__(timeout, n_parallel)

        if isinstance(build_func, str):
            if build_func == 'default':
                build_func = default_build_func
            elif build_func == 'ndk':
                build_func = android_ndk_build_func
            else:
                raise ValueError("Invalid build_func" + build_func)

        self.build_func = build_func
        self.executor = LocalExecutor(timeout=timeout)
        self.tmp_dir = tempfile.mkdtemp()

    def build(self, measure_inputs):
        results = []

        shutil.rmtree(self.tmp_dir)
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i:i + self.n_parallel]:
                ret = self.executor.submit(self.build_func,
                                           inp,
                                           self.tmp_dir,
                                           **self.build_kwargs)
                futures.append(ret)

            for future in futures:
                res = future.get()

                if isinstance(res, Exception):
                    # timeout or fleet error, return MeasureResult directly
                    results.append(MeasureResult((res,), MeasureErrorNo.BUILD_TIMEOUT,
                                                 self.timeout, time.time()))
                elif res.error is not None:
                    # instantiation error
                    if isinstance(res.error, InstantiationError):
                        results.append(MeasureResult((res.error,),
                                                     MeasureErrorNo.INSTANTIATION_ERROR,
                                                     res.time_cost, time.time()))
                    else:
                        if "InstantiationError" in str(res.error):
                            msg = str(res.error)
                            try:
                                msg = msg.split('\n')[-2].split(": ")[1]
                            except Exception:  # pylint: disable=broad-except
                                pass
                            results.append(MeasureResult((InstantiationError(msg),),
                                                         MeasureErrorNo.INSTANTIATION_ERROR,
                                                         res.time_cost, time.time()))
                        else:  # tvm error
                            results.append(MeasureResult((res.error,),
                                                         MeasureErrorNo.COMPILE_HOST,
                                                         res.time_cost, time.time()))
                else:
                    # return BuildResult
                    results.append(res)

        return results


class RPCRunner(Runner):
    """Run generated code on remove devices.
    This function will ask a RPC Tracker to get device for measurement.

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    key: str
        The key of the device registered in the tracker
    host: str
        The host address of RPC Tracker
    port: int
        The port of RPC Tracker
    number : int, optional
        Number of times to do measurement for tasking average
    repeat : int, optional
        Number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up. The returned result contains `repeat` costs,
    min_repeat_ms : float, optional
        Minimum duration of a timer measurement in milliseconds.
        When the run time of a measurement trial falls below this time, the
        `number` parameter will be automatically increased.
        Set this to improve the accuracy of perf measurement, e.g., when timers
        are not precise enough to capture short-running tasks. This parameter is
        also critical when devices need a certain minimum running time to "warm
        up," such as GPUs that need time to reach a performance power state.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    check_correctness: bool, optional
        Whether check correctness after measurement. This will use llvm cpu target to
        call your template and get the reference output.
        This can work for TOPI templates, but may not work for your custom template.
    """
    def __init__(self,
                 key, host, port, priority=1,
                 timeout=10, n_parallel=None,
                 number=4, repeat=3, min_repeat_ms=0, cooldown_interval=0.1,
                 check_correctness=False):
        super(RPCRunner, self).__init__(timeout, n_parallel)

        self.key = key
        self.host = host
        self.port = port
        self.priority = priority
        self.timeout = timeout

        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms
        self.cur_number = number

        self.ref_input = None
        self.ref_output = None
        self.check_correctness = check_correctness
        self.cooldown_interval = cooldown_interval

        self.executor = LocalExecutor()

    def set_task(self, task):
        self.task = task
        self.cur_number = self.number

        if check_remote(task.target, self.key, self.host, self.port):
            logger.info("Get devices for measurement successfully!")
        else:
            raise RuntimeError("Cannot get remote devices from the tracker. "
                               "Please check the status of tracker by "
                               "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                               "and make sure you have free devices on the queue status.")

        if self.check_correctness:
            # use llvm cpu to generate a reference input/output
            # this option works for tuning topi, but might not work for you custom op
            with _target.create("llvm"):
                s, arg_bufs = task.instantiate(task.config_space.get(0))
            self.ref_input = [np.random.uniform(size=get_const_tuple(x.shape)).astype(x.dtype)
                              for x in arg_bufs]
            func = build(s, arg_bufs, "llvm")
            tvm_buf = [nd.array(x) for x in self.ref_input]
            func(*tvm_buf)
            self.ref_output = [x.asnumpy() for x in tvm_buf]

    def get_build_kwargs(self):
        kwargs = {}
        if 'cuda' in self.task.target.keys or 'opencl' in self.task.target.keys:
            remote = request_remote(self.key, self.host, self.port)
            ctx = remote.context(str(self.task.target), 0)
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

    def run(self, measure_inputs, build_results):
        results = []
        remote_args = (self.key, self.host, self.port, self.priority, self.timeout)

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(measure_inputs[i:i+self.n_parallel],
                                              build_results[i:i+self.n_parallel]):
                ret = self.executor.submit(run_through_rpc,
                                           measure_inp,
                                           build_res,
                                           self.cur_number,
                                           self.repeat,
                                           self.cooldown_interval,
                                           remote_args,
                                           self.ref_input,
                                           self.ref_output)
                futures.append(ret)

            for future in futures:
                res = future.get()
                if isinstance(res, Exception):   # executor error or timeout
                    results.append(MeasureResult((str(res),), MeasureErrorNo.RUN_TIMEOUT,
                                                 self.timeout, time.time()))
                else:
                    results.append(res)

        # If some runs were too fast, do remeasure for them
        # to meet the requirement of `min_repeat_ms`
        remeasure = np.zeros((len(measure_inputs),), dtype=np.bool)
        pre_number = next_number = self.cur_number
        min_repeat_duration = self.min_repeat_ms / 1000.0
        for i, res in enumerate(results):
            if res.error_no == MeasureErrorNo.NO_ERROR:
                if np.mean(res.costs) * pre_number <= min_repeat_duration:
                    next_number = max(next_number,
                                      int(np.ceil(min_repeat_duration / np.mean(res.costs))))
                    remeasure[i] = True

        if pre_number != next_number:
            self.cur_number = next_number
            msg = "increasing number to %d" % self.cur_number
            logger.info(msg)

            re_measure_inputs = [x for i, x in enumerate(measure_inputs) if remeasure[i]]
            re_build_results = [x for i, x in enumerate(build_results) if remeasure[i]]
            re_res = self.run(re_measure_inputs, re_build_results)
            ct = 0
            for i, rerun in enumerate(remeasure):
                if rerun:
                    results[i] = re_res[ct]
                    ct += 1

        return results

class LocalRunner(RPCRunner):
    """Run generated code on local devices.

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    number : int, optional
        Number of times to do measurement for tasking average
    repeat : int, optional
        Number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up. The returned result contains `repeat` costs,
        each of which is the average of `number` test run.
    min_repeat_ms : float, optional
        Minimum duration of a timer measurement in milliseconds.
        When the run time of a measurement trial falls below this time, the
        `number` parameter will be automatically increased.
        Set this to improve the accuracy of perf measurement, e.g., when timers
        are not precise enough to capture short-running tasks. This parameter is
        also critical when devices need a certain minimum running time to "warm
        up," such as GPUs that need time to reach a performance power state.
    cooldown_interval: float, optional
        The cool down interval between two measurements.
    check_correctness: bool, optional
        Whether check correctness after measurement. This will use llvm cpu target to
        call your template and get the reference output.
        This can work for TOPI templates, but may not work for your custom template.

    Note
    ----
    This is a "fake" local mode. We start a silent rpc tracker and rpc server
    for the user. In this way we reuse timeout/isolation mechanism in RPC infrastructure.
    """
    def __init__(self,
                 timeout=10,
                 number=4, repeat=3, min_repeat_ms=0, cooldown_interval=0.1,
                 check_correctness=False):
        super(LocalRunner, self).__init__('', None, None, 0,
                                          timeout=timeout, n_parallel=1,
                                          number=number, repeat=repeat,
                                          min_repeat_ms=min_repeat_ms,
                                          cooldown_interval=cooldown_interval,
                                          check_correctness=check_correctness)
        self.tracker = None
        self.server = None

    def set_task(self, task):
        self.task = task

        from ...rpc.tracker import Tracker
        from ...rpc.server import Server

        tracker = Tracker('localhost', port=9000, port_end=10000, silent=True)
        device_key = '$local$device$%d' % tracker.port
        server = Server('localhost', port=9000, port_end=10000,
                        key=device_key,
                        use_popen=True, silent=True,
                        tracker_addr=(tracker.host, tracker.port))
        self.key = device_key
        self.host = tracker.host
        self.port = tracker.port

        super(LocalRunner, self).set_task(task)
        return server, tracker


def _build_func_common(measure_input, check_gpu=None, cuda_arch=None, build_option=None):
    """Common part for building a configuration"""
    target, task, config = measure_input

    with target:
        s, args = task.instantiate(config)

        # check invalidity of template and code hash consistency
        if not config.valid():
            raise InstantiationError(config.errors)

        opts = build_option or {}
        if check_gpu:  # Add verify pass to filter out invalid configs in advance.
            opts["add_lower_pass"] = [(2, gpu_verify_pass(**check_gpu))]
        if cuda_arch:
            set_cuda_target_arch(cuda_arch)

        with build_config(**opts):
            func = build(s, args, target_host=task.target_host)
    return func, tuple((get_const_tuple(x.shape), x.dtype) for x in args)


def default_build_func(measure_input, tmp_dir, **kwargs):
    """
    Default build func. This can work for cuda, opencl, llvm backend

    Parameters
    ----------
    measure_input: MeasureInput
        The input of measurement
    tmp_dir: str
        The path of temporary directory to export generated library
    """
    tic = time.time()
    try:
        filename = os.path.join(tmp_dir, "tmp_func_%0x.tar" % getrandbits(64))
        func, arg_info = _build_func_common(measure_input, **kwargs)
        func.export_library(filename)
    except Exception as e:  # pylint: disable=broad-except
        return BuildResult(None, None, e, time.time() - tic)
    return BuildResult(filename, arg_info, None, time.time() - tic)


def android_ndk_build_func(measure_input, tmp_dir, **kwargs):
    """
    Build function for android device using ndk.

    Parameters
    ----------
    measure_input: MeasureInput
        The input of measurement
    tmp_dir: str
        The path of temporary directory to export generated library
    """
    tic = time.time()
    try:
        filename = os.path.join(tmp_dir, "tmp_func_%0x.so" % getrandbits(64))
        func, arg_info = _build_func_common(measure_input, **kwargs)
        func.export_library(filename, ndk.create_shared)
    except Exception as e:  # pylint: disable=broad-except
        return BuildResult(None, None, e, time.time() - tic)
    return BuildResult(filename, arg_info, None, time.time() - tic)


def run_through_rpc(measure_input, build_result,
                    number, repeat, cooldown_interval,
                    remote_args, ref_input=None, ref_output=None):
    """Run a generated library through rpc

    Parameters
    ----------
    measure_input: MeasureInput
        The raw measure input
    build_result: BuildResult
        The result returned from Builder. This contains the path to the generated library.
    number : int, optional
        Number of times to do measurement for tasking average
    repeat : int, optional
        Number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up. The returned result contains `repeat` costs,
        each of which is the average of `number` test run.
    cooldown_interval: float
        The cool down interval between two measurements
    remote_args: Tuple
        The argument for request_remote
    ref_input: List of np.ndarray
        The reference input used for checking correctness
    ref_output: List of np.ndarray
        The reference output used for checking correctness
    """
    if isinstance(build_result, MeasureResult):
        return build_result

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR
    try:
        # upload built module
        remote = request_remote(*remote_args)
        remote.upload(build_result.filename)
        func = remote.load_module(os.path.split(build_result.filename)[1])
        ctx = remote.context(str(measure_input.target), 0)
        time_f = func.time_evaluator(
            func.entry_name, ctx, number=number, repeat=repeat)

        # set input
        if ref_input:
            args = [nd.array(x, ctx=ctx) for x in ref_input]
        else:
            args = [nd.empty(x[0], dtype=x[1], ctx=ctx) for x in build_result.arg_info]

        costs = time_f(*args).results
        if len(costs) > 2:  # remove largest and smallest value to reduce variance
            costs = list(costs)
            costs.sort()
            costs = tuple(costs[1:-1])

        # check correctness of output
        if ref_output:
            for expected, real in zip(ref_output, args):
                if not np.allclose(expected, real.asnumpy(), rtol=1e-4):
                    logger.warning("Wrong Answer!")
                    errno = MeasureErrorNo.WRONG_ANSWER
    except TVMError as exc:
        msg = str(exc)
        if "Stack trace returned" in msg:
            msg = msg[:msg.index("Stack trace returned")]
        if "CUDA Source" in msg:
            msg = msg[:msg.index("CUDA Source")]
        costs = (RuntimeError(msg[:1024]),)
        errno = MeasureErrorNo.RUNTIME_DEVICE
    tstamp = time.time()
    time.sleep(cooldown_interval)
    return MeasureResult(costs, errno, tstamp - tic + build_result.time_cost, tstamp)


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
    host = host or os.environ['TVM_TRACKER_HOST']
    port = port or int(os.environ['TVM_TRACKER_PORT'])

    tracker = _rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority,
                             session_timeout=timeout)
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
        remote.context(str(target))
    t = threading.Thread(target=_check,)
    t.start()
    t.join(timeout)
    return not t.is_alive()


@register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    ptx = nvcc.compile_cuda(code, target="ptx", arch=AutotvmGlobalScope.current.cuda_target_arch)
    return ptx


def set_cuda_target_arch(arch):
    """set target architecture of nvcc compiler

    Parameters
    ----------
    arch: str
        The argument of nvcc -arch. (e.g. "sm_51", "sm_62")
    """
    AutotvmGlobalScope.current.cuda_target_arch = arch


def gpu_verify_pass(**kwargs):
    """Verify the validity of a gpu kernel.
    This pass will check memory usage and number of threads per block.
    """
    def verify_pass(stmt):
        valid = ir_pass.VerifyGPUCode(stmt, kwargs)
        if not valid:
            raise InstantiationError("Skipped because of invalid gpu kernel")
        return stmt
    return verify_pass
