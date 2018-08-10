# pylint: disable=consider-using-enumerate,invalid-name,too-many-function-args
"""
Functions that run on executor for measurement.
These functions are responsible for building tvm module, uploading it to
remote devices, recording the running time costs and checking the correctness of output
"""

import logging
import os
import time
from random import getrandbits
import threading

import numpy as np

from ... import ir_pass, build, build_config, nd, context, TVMError, register_func, \
    target as _target, rpc as _rpc
from ...contrib import nvcc, util, ndk

from ..util import get_const_tuple
from ..env import AutotvmGlobalScope
from ..task.space import InstantiationError

from .measure import MeasureResult, MeasureErrorNo
from .local_executor import LocalExecutor

logger = logging.getLogger('autotvm')

class HashMismatchError(ValueError):
    """Raised when the code hash of a submitted config doesn't match that on the
       measure side """
    pass


def request_remote(device_key, tracker_addr=None, priority=1, timeout=60):
    """request a remote session

    Parameters
    ----------
    device_key: string
        device key of registered device in tracker
    tracker_addr: Tuple(string, int), optional
        The address of rpc tracker in (host, port) format.
        If is none, will use environment variable "TVM_TRACKER_HOST"
        and "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this session (units: seconds)

    Returns
    ------
    session: RPCSession
    """
    # connect to the tracker
    if tracker_addr:
        host = tracker_addr[0] or os.environ['TVM_TRACKER_HOST']
        port = tracker_addr[1] or int(os.environ['TVM_TRACKER_PORT'])
    else:
        host = os.environ['TVM_TRACKER_HOST']
        port = int(os.environ['TVM_TRACKER_PORT'])

    tracker = _rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority,
                             session_timeout=timeout)
    return remote

def check_remote(target, device_key, tracker_addr=None, priority=2, timeout=10):
    """
    Check the availability of a remote device

    Parameters
    ----------
    target: Target
        The wanted compilation target
    device_key: string
        device key of registered device in tracker
    tracker_addr: Tuple(string, int), optional
        The address of rpc tracker in (host, port) format.
        If is none, will use environment variable "TVM_TRACKER_HOST"
        and "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this check (units: seconds).
        If time is out, a RuntimerError will be raised.
    """
    def _check():
        remote = request_remote(device_key, tracker_addr, priority)
        remote.context(str(target))
    t = threading.Thread(target=_check,)
    t.start()
    t.join(timeout)
    return not t.is_alive()

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
    from ..database import filter_inputs

    measure_func = option['measure_func']
    number, repeat = option['number'], option['repeat']
    timeout, n_parallel, do_fork = option['timeout'], option['n_parallel'], option['do_fork']
    build_func = option['build_func']
    check_correctness = option['check_correctness']
    replay_db = option['replay_db']

    executor = LocalExecutor(timeout=timeout, do_fork=do_fork)

    # convert convenient string to function object
    attach_objects = None
    if measure_func == 'local':
        # start temporary rpc tracker and rpc server for the user
        from ...rpc.tracker import Tracker
        from ...rpc.server import Server

        tracker = Tracker('localhost', port=9000, port_end=10000, silent=True)
        device_key = '$local$device$%d' % tracker.port
        server = Server('localhost', port=9000, port_end=10000,
                        key=device_key,
                        use_popen=True, silent=True,
                        tracker_addr=(tracker.host, tracker.port))

        measure_func = rpc(device_key, tracker.host, tracker.port)
        attach_objects = (server, tracker)

    build_kwargs = {}
    if build_func == 'default':
        build_func = default_build_func
    if build_func == 'ndk':
        build_func = default_build_func
        build_kwargs['use_ndk'] = True

    # check the availability of remote devices
    if hasattr(measure_func, 'rpc_info'):
        rpc_info = measure_func.rpc_info
        if check_remote(task.target, rpc_info['key'], (rpc_info['host'], rpc_info['port'])):
            logger.info("Get devices for measurement successfully!")
        else:
            raise RuntimeError("Cannot get remote devices from the tracker. "
                               "Please check the status of tracker by "
                               "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                               "and make sure you have free devices on the queue status.")

    # add device info of cuda and opencl target
    if ('cuda' in task.target.keys or 'opencl' in task.target.keys) \
            and hasattr(measure_func, 'rpc_info'):
        rpc_info = measure_func.rpc_info
        add_gpu_target_info(task.target, rpc_info["key"], (rpc_info["host"], rpc_info["port"]),
                            build_kwargs)

    if check_correctness:
        # use llvm cpu to generate a reference input/output
        # this option works for tuning topi, but might not work for you custom op
        with _target.create("llvm"):
            s, arg_bufs = task.instantiate(task.config_space.get(0))
        ref_input = [np.random.uniform(size=get_const_tuple(x.shape)).astype(x.dtype)
                     for x in arg_bufs]
        func = build(s, arg_bufs, "llvm")
        tvm_buf = [nd.array(x) for x in ref_input]
        func(*tvm_buf)
        ref_output = [x.asnumpy() for x in tvm_buf]
    else:
        ref_input = ref_output = None

    def measure_batch(measure_inputs):
        """measure the time cost for a batch of configs in real machines"""
        if replay_db is not None:
            partial_results, measure_inputs = \
                filter_inputs(replay_db, measure_inputs, retry=False)

        # launch measure jobs in parallel
        pack_size = getattr(measure_func, "pack_size", 1)  # measure `pack_size` inputs in one job
        futures = []
        for i in range(0, len(measure_inputs), pack_size):
            input_pack = measure_inputs[i:i + pack_size]
            ret = executor.submit(
                measure_func,
                input_pack,
                build_func,
                build_kwargs,
                number,
                repeat,
                ref_input,
                ref_output)
            futures.append(ret)

        # transform results
        results = []
        for future in futures:
            result = future.get()
            if isinstance(result, Exception):
                tstamp = time.time()
                results.extend([MeasureResult((result,), MeasureErrorNo.FLEET_ERROR,
                                              timeout, tstamp)] * pack_size)
            else:
                results.extend(result)

        if replay_db is not None:
            result_idx = 0
            for i in range(len(partial_results)):
                if partial_results[i] is None:
                    partial_results[i] = results[result_idx]
                    result_idx += 1
            return partial_results
        return results

    measure_batch.n_parallel = n_parallel
    # attach server and tracker object to avoid them of being garbage-collected
    measure_batch.attach_objects = attach_objects
    return measure_batch


def rpc(key,
        host=None,
        port=None,
        priority=1,
        session_timeout=60,
        pack_size=1):
    """
    Create a standard measure_func which uses RPC Tracker for measurement.
    This measure_func will request a device from the RPC Tracker and
    upload the built binary library to that device for measurement.

    Parameters
    ----------
    key: str
        The registered key of the device in tracker. The tuner will request devices for
        measurement by this key.
    host: str, optional
        The hostname of RPC Tracker. If not set, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port of RPC Tracker. If not set, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        Priority of this task, used by scheduler in tracker
    session_timeout: int, optional
        Timeout of rpc session
    pack_size: int, optional
        The number of configs measure in one RPC session.
        Usually this can be set to 1. If your device has high overhead to establish a
        rpc connection, set this higher.
    """
    def fmeasure(input_pack, build_func, build_kwargs, number, repeat, ref_input, ref_output):
        """Do measurement for a list of inputs inside a same RPC session.

        Parameters
        ----------
        input_pack: List of MeasureInput
            The inputs of measurement
        build_func: callable
            Function for building the code. see :any:`default_build_func` for example
        build_kwargs: dict
            Extra arguments for build_func
        number : int, optional
            Number of times to do the measurement for average
        repeat : int, optional
            Number of times to repeat the measurement.
            In total, the generated code will be run (1 + number x repeat) times,
            where the first one is warm up. The returned result contains `repeat` costs,
            each of which is the average of `number` test run.
        ref_input: List of numpy array
            Reference input for correctness check
        ref_output: List of numpy array
            Reference output for correctness check

        Returns
        -------
        results: List of MeasureResult
            The results for input_pack
        """
        remote = request_remote(key, (host, port), priority, session_timeout)

        res = _measure_common(input_pack, build_func, build_kwargs, number, repeat,
                              ref_input, ref_output,
                              remote)
        return res

    fmeasure.pack_size = pack_size
    fmeasure.rpc_info = {"key": key, "host": host, "port": port}
    return fmeasure


def _measure_common(input_pack, build_func, build_kwargs, number, repeat,
                    ref_input=None, ref_output=None, remote=None):
    """Measure the time cost for a pack of inputs.

    (Note: A pack is a list of inputs which will be measured inside a same RPC session)

    Parameters
    ----------
    input_pack : list of MeasureInput
        The inputs we need to evaluate
    build_func : function takes MeasureInput returns tuple of (time_func, ctx, args)
        The build function used to build each input.
    build_kwargs: Dict
        The extra keyword arguments to build_func
    number : int, optional
        Number of times to do the measurement for average
    repeat : int, optional
        Number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up. The returned result contains `repeat` costs,
        each of which is the average of `number` test run.
    ref_input: Array of np.ndarray, optional
        Reference input for checking correctness
    ref_output: Array of np.ndarray, optional
        Reference output for checking correctness
    remote: RPCSession, optional
        The remote RPC session

    Returns
    -------
    res_pack : Array of MeasureResult
        The list of results of measurement.
    """
    res_pack = []
    tmp_dir = util.tempdir() if remote else None

    for inp in input_pack:
        tic = time.time()

        # build function
        try:
            func, arg_bufs, filename = build_func(inp, tmp_dir, **build_kwargs)
        except TVMError as exc:
            tstamp = time.time()
            msg = str(exc)
            if "Stack trace returned" in msg:
                msg = msg[:msg.index("Stack trace returned")]
            if "InstantiationError" in msg:
                try:
                    msg = msg.split('\n')[-2].split(": ")[1]
                except Exception:  # pylint: disable=broad-except
                    pass
                res_pack.append(MeasureResult((InstantiationError(msg),),
                                              MeasureErrorNo.INSTANTIATION_ERROR,
                                              tstamp - tic, tstamp))
            else:
                res_pack.append(MeasureResult((RuntimeError(msg),),
                                              MeasureErrorNo.COMPILE_HOST,
                                              tstamp - tic, tstamp))
            continue
        except InstantiationError as e:
            tstamp = time.time()
            res_pack.append(MeasureResult((InstantiationError(str(e)),),
                                          MeasureErrorNo.INSTANTIATION_ERROR,
                                          tstamp - tic, tstamp))
            continue

        # upload built module
        if remote:
            remote.upload(tmp_dir.relpath(filename))
            func = remote.load_module(filename)
            ctx = remote.context(str(inp.target), 0)
            time_f = func.time_evaluator(
                func.entry_name, ctx, number=number, repeat=repeat)
        else:
            ctx = context(str(inp.target), 0)
            time_f = func.time_evaluator(
                func.entry_name, ctx, number=number, repeat=repeat)

        # measure time
        errno = MeasureErrorNo.NO_ERROR
        try:
            if ref_input:
                args = [nd.array(x, ctx=ctx) for x in ref_input]
            else:
                args = [nd.empty(get_const_tuple(x.shape), dtype=x.dtype, ctx=ctx)
                        for x in arg_bufs]
            costs = time_f(*args).results
            if len(costs) > 2:  # remove largest and smallest value to reduce variance
                costs = list(costs)
                costs.sort()
                costs = tuple(costs[1:-1])
            if ref_output:
                for expected, real in zip(ref_output, args):
                    if not np.allclose(expected, real.asnumpy(), rtol=1e-4):
                        logger.warning("Wrong Answer!")
                        errno = MeasureErrorNo.WRONG_ANSWER
        except TVMError as exc:
            msg = str(exc)
            if "Stack trace returned" in msg:
                msg = msg[:msg.index("Stack trace returned")]
            costs = (RuntimeError(msg),)
            errno = MeasureErrorNo.RUNTIME_DEVICE
        tstamp = time.time()
        res_pack.append(MeasureResult(costs, errno, tstamp - tic, tstamp))
    return res_pack


def default_build_func(inp, tmp_dir=None, **kwargs):
    """Build function module. Exception will be raised when any error occurs

    Parameters
    ----------
    inp: MeasureInput
       The input of this measurement
    tmp_dir: tvm.contrib.util.TempDirectory, optional
       The temporary directory for exporting built binary library.
       If is not None (in RPC mode), the library in this directory will be uploaded to
       remote devices.
    kwargs: Dict, optional
        Other extra arguments

    Returns
    -------
    func: Function
        TVM built function. Typically this is the return value of tvm.build.
    args: Array of Buffer or Tensor
        The argument list for the function. Typically this is the second argument of tvm.build.
    filename: str
        The filename of the output build library
    """
    # build function
    with inp.target:
        s, args = inp.task.instantiate(inp.config)

        # check invalidity of template and code hash consistency
        if not inp.config.valid():
            raise InstantiationError(inp.config.errors)
        code_hash = getattr(s, 'code_hash', None)
        if inp.config.code_hash != code_hash:
            raise HashMismatchError('got {0:s}, expected {1:s}'
                                    .format(str(inp.config.code_hash), str(code_hash)))

        opts = {}
        if "check_gpu" in kwargs:  # Add verify pass to filter out invalid configs in advance.
            opts["add_lower_pass"] = [(2, gpu_verify_pass(**kwargs['check_gpu']))]
        if 'cuda_arch' in kwargs:
            set_cuda_target_arch(kwargs['cuda_arch'])

        with build_config(**opts):
            func = build(s, args, target_host=inp.task.target_host)

    # export library to temp directory
    if tmp_dir:
        if kwargs.get('use_ndk', False):  # for Android NDK
            filename = "tmp_func_%0x.so" % getrandbits(64)
            func.export_library(tmp_dir.relpath(filename), ndk.create_shared)
        else:
            filename = "tmp_func_%0x.tar" % getrandbits(64)
            func.export_library(tmp_dir.relpath(filename))
    else:
        filename = None

    return func, args, filename


def add_gpu_target_info(target, device_key, rpc_tracker_addr, kwargs):
    """Add device info for gpu target.
    The info will be used to check the validity of generated code."""
    remote = request_remote(device_key, rpc_tracker_addr)
    ctx = remote.context(str(target), 0)
    max_dims = ctx.max_thread_dimensions
    kwargs['check_gpu'] = {
        'max_shared_memory_per_block': ctx.max_shared_memory_per_block,
        'max_threads_per_block': ctx.max_threads_per_block,
        'max_thread_x': max_dims[0],
        'max_thread_y': max_dims[1],
        'max_thread_z': max_dims[2],
    }

    if 'cuda' in target.keys:
        kwargs["cuda_arch"] = "sm_" + "".join(ctx.compute_version.split('.'))

def set_cuda_target_arch(arch):
    """set target architecture of nvcc compiler"""
    AutotvmGlobalScope.current.cuda_target_arch = arch


@register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    ptx = nvcc.compile_cuda(code, target="ptx", arch=AutotvmGlobalScope.current.cuda_target_arch)
    return ptx


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
