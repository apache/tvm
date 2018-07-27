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

import numpy as np

from ...contrib import nvcc, util
from ... import rpc, ir_pass, build, build_config, nd, context, TVMError, register_func

from ..util import get_const_tuple
from ..env import AutotvmGlobalScope
from .measure import MeasureResult, MeasureErrorNo
from ..task.space import InstantiationError


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
        The address of rpc tracker in (host, port) format
    priority: int, optional
        priority of this request, larger is more prior
    timeout: float, optional
        timeout of this session (units: seconds)

    Returns
    ------
    session: RPCSession
    """
    # connect to the tracker
    if tracker_addr:
        host = tracker_addr[0]
        port = tracker_addr[1]
    else:
        host = os.environ['TVM_TRACKER_HOST']
        port = int(os.environ['TVM_TRACKER_PORT'])

    tracker = rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority,
                             session_timeout=timeout)
    return remote


def _measure_pack(fbuild, input_pack, ref_input, ref_output):
    """Do measure for a pack of inputs.
    This function mainly does error handling and correctness check.

    (Note: A pack is a list of inputs which will be measured inside a same RPC session)

    Parameters
    ----------
    fbuild : function takes MeasureInput returns tuple of (time_func, ctx, args)
        The build function used to build each input.
    input_pack : list of MeasureInput
        The inputs we need to evaluate
    ref_input: Array of np.ndarray
        Reference input for checking correctness
    ref_output: Array of np.ndarray
        Reference output for checking correctness

    Returns
    -------
    res_pack : array of MeasureResult
        The list of execution result of measurement.
    """
    res_pack = []
    for inp in input_pack:
        tic = time.time()
        try:
            time_f, ctx, arg_bufs = fbuild(inp)
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
            res_pack.append(MeasureResult((e,),
                                          MeasureErrorNo.INSTANTIATION_ERROR,
                                          tstamp - tic, tstamp))
            continue

        # measure time
        errno = MeasureErrorNo.NO_ERROR
        try:
            if ref_input:
                args = [nd.array(x, ctx) for x in ref_input]
            else:
                args = [nd.empty(get_const_tuple(x.shape), dtype=x.dtype,
                                 ctx=ctx) for x in arg_bufs]
            costs = time_f(*args).results
            if len(costs) > 2:  # remove largest and smallest value to reduce variance
                costs = list(costs)
                costs.sort()
                costs = tuple(costs[1:-1])
            if ref_output:
                for expected, real in zip(ref_output, args):
                    if not np.allclose(expected, real.asnumpy(), rtol=1e-4):
                        logging.warning("Wrong Answer!")
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
       The temporary directory for output built binary library.
       In RPC mode, we will upload this library to remote device
    kwargs: Dict
       Other arguments

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

        if "check_gpu" in kwargs:
            values = kwargs['check_gpu']
            # Add gpu verify pass to filter out invalid configs in advance.
            # This can accelerate the tuning process
            check_keys = ['max_shared_memory_per_block', 'max_threads_per_block',
                          'max_thread_x', 'max_thread_y', 'max_thread_z']
            opts["add_lower_pass"] = [
                (2, gpu_verify_pass(**{key: values[key] for key in check_keys}))]
        if 'cuda_arch' in kwargs:
            set_cuda_target_arch(kwargs['cuda_arch'])

        with build_config(**opts):
            func = build(s, args, target_host=inp.task.target_host)

    # export library to temp directory
    if tmp_dir:
        if kwargs.get('use_ndk', False):  # for Android NDK
            from ...contrib import ndk
            filename = "tmp_func_%0x.so" % getrandbits(64)
            func.export_library(tmp_dir.relpath(filename), ndk.create_shared)
        else:
            filename = "tmp_func_%0x.tar" % getrandbits(64)
            func.export_library(tmp_dir.relpath(filename))

    return func, args, filename


def measure_rpc(input_pack,
                rpc_device_key,
                number,
                repeat=1,
                rpc_tracker_addr=None,
                rpc_priority=1,
                rpc_timeout=60,
                build_func='default',
                **kwargs):
    """Measure the time cost on a device by rpc

    Parameters
    ----------
    input_pack : list of MeasureInput
        The inputs we need to evaluate
    rpc_device_key: str
        The device key of registered devices in tracker
    number : int
        Number of times to get the running measurement
    repeat : int, optional
        How many times we want to repeat the measurement.

    rpc_tracker_addr: Tuple(string, int), optional
        The address of rpc tracker in (host, port) format
        If is none, will use environment variable
    rpc_priority: int, optional
        priority of this task, used by scheduler in tracker
    rpc_timeout: int, optional
        timeout of the rpc session

    build_func: str or callable, optional
        'default': call default build_func. This works for normal target (llvm, cuda)

        'ndk': use Android NDK to create shared library. Use this for android target

        callable; customized build function for other backends (e.g. VTA)

    kwargs: dict, optional
        Additional key word arguments

    Returns
    -------
    res_pack : Array of MeasureResult
        The list of execution results of measurement.
    """
    def _fbuild(inp):
        """ Local build function."""
        tmp_dir = util.tempdir()

        if build_func == 'default':
            func, args, filename = default_build_func(inp, tmp_dir, **kwargs)
        elif build_func == 'ndk':
            kwargs['use_ndk'] = True
            func, args, filename = default_build_func(inp, tmp_dir, **kwargs)
        else:
            func, args, filename = build_func(inp, tmp_dir, **kwargs)

        remote = request_remote(rpc_device_key, rpc_tracker_addr, rpc_priority, rpc_timeout)
        remote.upload(tmp_dir.relpath(filename))
        func = remote.load_module(filename)
        ctx = remote.context(str(inp.target), 0)
        time_f = func.time_evaluator(
            func.entry_name, ctx, number=number, repeat=repeat)
        return time_f, ctx, args

    ret = _measure_pack(_fbuild, input_pack,
                        kwargs.get("ref_input", None), kwargs.get("ref_output", None))
    return ret

def measure_local(input_pack,
                  number,
                  repeat=1,
                  build_func='default',
                  **kwargs):
    """Measure the time cost on a local machine.

    Parameters
    ----------
    input_pack : list of MeasureInput
        The inputs we need to evaluate
    number : int
        Number of times to get the running measurement
    repeat : int, optional
        How many times we want to repeat the measurement.

    build_func: str or callable, optional
        'default': call default build_func. This works for normal target (llvm, cuda)

        'ndk': use Android NDK to create shared library. Use this for android target

        callable; customized build function for other backends (e.g. VTA)

    kwargs: dict, optional
        Additional key word arguments

    Returns
    -------
    res_pack : Array of MeasureResult
        The list of execution results of measurement.
    """
    def _fbuild(inp):
        """ Local build function """
        tmp_dir = util.tempdir()

        if build_func == 'default':
            func, args, filename = default_build_func(inp, tmp_dir, **kwargs)
        else:
            func, args, filename = build_func(inp, tmp_dir, **kwargs)

        ctx = context(str(inp.target), 0)
        time_f = func.time_evaluator(
            func.entry_name, ctx, number=number, repeat=repeat)
        return time_f, ctx, args

    ret = _measure_pack(_fbuild, input_pack,
                        kwargs.get("ref_input", None), kwargs.get("ref_output", None))
    return ret


def gpu_verify_pass(**kwargs):
    """Verify the validity of a gpu kernel
    This pass will check shared memory size and number of threads per block.
    """
    def verify_pass(stmt):
        valid = ir_pass.VerifyGPUCode(stmt, kwargs)
        if not valid:
            raise InstantiationError("Skipped because of invalid gpu kernel")
        return stmt
    return verify_pass


@register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    ptx = nvcc.compile_cuda(code, target="ptx", arch=AutotvmGlobalScope.current.cuda_target_arch)
    return ptx

def set_cuda_target_arch(arch):
    """set target architecture of nvcc compiler"""
    AutotvmGlobalScope.current.cuda_target_arch = arch
