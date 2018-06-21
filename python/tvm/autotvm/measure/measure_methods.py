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

from ...contrib import rpc, ndk, nvcc
from ... import ir_pass, build, build_config, nd, context, TVMError, register_func

from .measure import MeasureResult, MeasureErrorNo
from ..template.space import InstantiationError


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


def _measure_generic(fbuild, input_pack, check_correctness):
    """Generic measurement function

    Parameters
    ----------
    fbuild : function takes MeasureInput returns tuple of (time_func, ctx)
        The build function used to build each input.
    input_pack : list of MeasureInput
        The inputs we need to evaluate
    check_correctness: bool
        Whether check correctness after measurement.

    Returns
    -------
    res_pack : array of MeasureResult
        The list of execution result of measurement.
    """
    res_pack = []
    for inp in input_pack:
        tic = time.time()
        try:
            time_f, ctx, args = fbuild(inp)
        except TVMError as e:
            tstamp = time.time()
            res_pack.append(MeasureResult((e,),
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
            if inp.task.ref_input is None:
                inp.task.init_ref_data(args, check_correctness)
            arg_bufs = [nd.array(x, ctx) for x in inp.task.ref_input]
            costs = time_f(*arg_bufs).results
            if len(costs) > 2:  # remove largest and smallest value to reduce variance
                costs = list(costs)
                costs.sort()
                costs = tuple(costs[1:-1])
            if check_correctness:
                for expected, real in zip(inp.task.ref_output, arg_bufs):
                    if not np.allclose(expected, real.asnumpy(), rtol=1e-4):
                        logging.warning("Wrong Answer!")
                        errno = MeasureErrorNo.WRONG_ANSWER
        except TVMError as exc:
            msg = str(exc)
            if "\n" in msg:
                costs = (RuntimeError(msg[:msg.index("\n")],))
            else:
                costs = (RuntimeError(msg[:msg.index("\\n")]),)
            errno = MeasureErrorNo.RUNTIME_DEVICE
        tstamp = time.time()
        res_pack.append(MeasureResult(costs, errno, tstamp - tic, tstamp))
    return res_pack

def measure_rpc(input_pack,
                device_key,
                number,
                repeat=1,
                check_correctness=False,
                build_option=None,
                rpc_tracker_addr=None,
                rpc_priority=1,
                rpc_timeout=60,
                tmp_dir=None,
                use_ndk=False):
    """Measure the time cost on a device by rpc

    Parameters
    ----------
    input_pack : list of MeasureInput
        The inputs we need to evaluate
    device_key: str
        The device key of registered devices in tracker
    number : int
        Number of times to get the running measurement
    repeat : int, optional
        How many times we want to repeat the measurement.
    check_correctness: bool, optional
        Whether check correctness after measurement.
    build_option: Dict
        build options for tvm.build_config

    rpc_tracker_addr: Tuple(string, int), optional
        The address of rpc tracker in (host, port) format
        If is none, will use environment variable
    rpc_priority: int, optional
        priority of this task, used by scheduler in tracker
    rpc_timeout: int, optional
        timeout of the rpc session

    tmp_dir: tvm.contrib.util.TempDirectory, optional
        directory to store temp file

    use_ndk : bool, optional
        Whether export requires ndk

    Returns
    -------
    res_pack : Array of MeasureResult
        The list of execution results of measurement.
    """
    def _fbuild(inp):
        """ Local build function."""
        with inp.target:
            s, args = inp.task.instantiate(inp.config)
            if not inp.config.valid():
                raise InstantiationError(inp.config.errors)
            code_hash = getattr(s, 'code_hash', None)
            if inp.config.code_hash != code_hash:
                raise HashMismatchError('got {0:s}, expected {1:s}'
                                        .format(str(inp.config.code_hash), str(code_hash)))

            opts = build_option or {}
            if "cuda" in inp.target.keys:
                # Add cuda verify pass to filter out invalid configs in advance.
                # This can accelerate the tuning process
                max_shared_memory_per_block, max_threads_per_block, arch = \
                    get_cuda_device_info(device_key, rpc_tracker_addr, tmp_dir)
                set_cuda_target_arch(arch)
                opts["add_lower_pass"] = \
                    [(2, cuda_verify_pass(max_shared_memory_per_block=max_shared_memory_per_block,
                                          max_thread_per_block=max_threads_per_block))]
            with build_config(**opts):
                func = build(s, args, target_host=inp.task.target_host)

        file_name = "tmp_func_%0x.tar" % getrandbits(64)
        path = tmp_dir.relpath(file_name)
        if not use_ndk:
            func.export_library(path)
        else:
            func.export_library(path, ndk.create_shared)
        remote = request_remote(device_key, rpc_tracker_addr, rpc_priority, rpc_timeout)
        remote.upload(path)
        func = remote.load_module(file_name)
        ctx = remote.context(str(inp.target), 0)
        time_f = func.time_evaluator(
            func.entry_name, ctx, number=number, repeat=repeat)
        return time_f, ctx, args

    ret = _measure_generic(_fbuild, input_pack, check_correctness)
    return ret


def measure_local(input_pack,
                  number,
                  repeat=1,
                  check_correctness=False,
                  build_option=None):
    """Measure the time cost on a local machine.

    Parameters
    ----------
    input_pack : list of MeasureInput
        The inputs we need to evaluate
    number : int
        Number of times to get the running measurement
    repeat : int, optional
        How many times we want to repeat the measurement.
    check_correctness: bool, optional
        Whether check correctness after measurement.
    build_option: dict, optional
        build options for tvm.build_config

    Returns
    -------
    res_pack : Array of MeasureResult
        The list of execution results of measurement.
    """

    def _fbuild(inp):
        """ Local build function """
        with inp.target:
            s, args = inp.task.instantiate(inp.config)
            if not inp.config.valid():
                raise InstantiationError(inp.config.errors)
            code_hash = getattr(s, 'code_hash', None)
            if inp.config.code_hash != code_hash:
                raise HashMismatchError('got {0:s}, expected {1:s}'
                                        .format(str(inp.config.code_hash), str(code_hash)))

            opts = build_option or {}
            ctx = context(str(inp.target), 0)
            if "cuda" in inp.target.keys:
                # Add cuda verify pass to filter out invalid configs in advance.
                # This can accelerate the tuning process
                opts["add_lower_pass"] = \
                    [(2, cuda_verify_pass(ctx.max_shared_memory_per_block,
                                          ctx.max_threads_per_block))]
            with build_config(**opts):
                func = build(s, args, target_host=inp.task.target_host)

        time_f = func.time_evaluator(
            func.entry_name, ctx, number=number, repeat=repeat)
        return time_f, ctx, args

    ret = _measure_generic(_fbuild, input_pack, check_correctness)
    return ret


def cuda_verify_pass(**kwargs):
    """Verify the invalidity of a cuda kernel
    This pass will check shared memory size and number of thread per block.
    """
    def verify_pass(stmt):
        valid = ir_pass.VerifyGPUCode(stmt, kwargs)
        if not valid:
            raise InstantiationError("invalid cuda kernel")
        return stmt
    return verify_pass


_cuda_target_arch = None
@register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    global _cuda_target_arch
    ptx = nvcc.compile_cuda(code, target="ptx", arch=_cuda_target_arch)
    return ptx

def set_cuda_target_arch(arch):
    """set target architecture of nvcc compiler"""
    global _cuda_target_arch
    _cuda_target_arch = arch


def get_cuda_device_info(device_key, rpc_tracker_addr=None, tmp_dir=None):
    """get device query result from remote cuda device

    Parameters
    ----------
    device_key: str
        The device key of registered device in RPC tracker.
    rpc_tracker_addr: Tuple(string, int), optional
        The address of rpc tracker.
        If is none, will use environment variable
    tmp_dir: tvm.contrib.TempDirectory, optional
        Temporary directory to store cache

    Returns
    -------
    max_shared_memory_per_block: int
    max_threads_per_block: int
    """
    if tmp_dir:
        file_name = tmp_dir.relpath(".autotvm_cuda_device_query_%s.cache" % device_key)
        find_cache = False
        if os.path.isfile(file_name):
            # find cache file, try to parse it
            try:
                lines = list(open(file_name).readlines())
                max_shared_memory_per_block = int(lines[1])
                max_threads_per_block = int(lines[2])
                arch = lines[3].strip()
                find_cache = True
            except Exception:  # pylint: disable=broad-except
                pass
    else:
        find_cache = False

    if not find_cache:
        remote = request_remote(device_key, rpc_tracker_addr)
        ctx = remote.context('cuda', 0)
        max_shared_memory_per_block = ctx.max_shared_memory_per_block
        max_threads_per_block = ctx.max_threads_per_block
        arch = "sm_" + "".join(ctx.compute_version.split('.'))
        if tmp_dir:
            with open(file_name, 'w') as fout:
                fout.writelines(["%s\n" % x for x in [
                    "# This is a cache file to store the compilation options for cuda nvcc. "
                    "You can delete this safely",
                    max_shared_memory_per_block,
                    max_threads_per_block,
                    arch]])
        # release remote session
        del ctx
        del remote

    return max_shared_memory_per_block, max_threads_per_block, arch
