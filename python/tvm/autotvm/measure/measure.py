# pylint: disable=pointless-string-statement,consider-using-enumerate,invalid-name
"""User facing API for specifying how to measure the generated code"""
import time
from collections import namedtuple

import numpy as np

from ... import build, nd, target as _target
from ...rpc.tracker import Tracker
from ...rpc.server import Server

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
        If no error occurs for this measurement, it is an array of measured running times.
        If some error occurs during the measurement, it is an array of the exception objections.
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
    INSTANTIATION_ERROR = 1   # error when calling template function
    COMPILE_HOST = 2          # error when compiling code on host (e.g. tvm.build)
    COMPILE_DEVICE = 3        # error when compiling code on device (e.g. opencl JIT on device)
    RUNTIME_DEVICE = 4        # error when run program on device
    WRONG_ANSWER = 5          # answer is wrong when compared to a golden output
    FLEET_ERROR = 6           # error of measure infrastructure


def measure_option(measure_func='local',
                   number=1,
                   repeat=1,
                   timeout=60,
                   parallel_num=1,
                   do_fork=True,
                   pack_size=1,
                   check_correctness=False,

                   rpc_device_key=None,
                   rpc_priority=1,
                   rpc_timeout=60,
                   rpc_tracker_addr=None,

                   build_func='default',

                   replay_db=None,
                   save_to_replay_db=True):
    """Configure how to do measurement

    Parameters
    ----------
    measure_func: str or callable
        'local': use the local device for measurement. The tuner will start a tracker
        and a RPC server silently for the user.

        'rpc': request devices for measurement from the rpc tracker. In this mode,
        you should start a rpc tracker in a separate processing and register your
        device to the tracker.

        callable: It is a customized function for measurement.

    number : int, optional
        Number of times to do the measurement for average
    repeat : int, optional
        Number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up. The returned result contains `repeat` costs,
        each of which is the average of `number` test run.
    timeout: int, optional
        Timeout for a whole batch. TimeoutError will be returned as the result if a
        task timeouts.
    parallel_num: int, optional
        The number of measurement task that can run in parallel.
        Set this according to the number of cpu cores (for compilation) and
        the number of devices you have (for measuring generate code).
    do_fork: bool, optional
        Whether use multiprocessing (based on fork) for running measure jobs in parallel.
        Set this to False if you want to debug or fork is not suitable for you case.
        NOTE: If this is False, parallel and timeout do not work.
    pack_size : int, optional
        Number of configs to measure in one RPC call.
        Usually this can be set to 1. If your device has high cost to establish a rpc connection,
        set this higher.
    check_correctness: bool
        Whether check correctness after measurement. This will use llvm cpu as reference.

    replay_db : Database, optional
        The database that we retrieve saved MeasureResults from
    save_to_replay_db: bool, optional
        Whether save measure result to database. This is useless when replay_db is None

    build_func: str or callable, optional
        'default': call default builder. This works for normal target (llvm, cuda)

        'ndk': use Android NDK to create shared library. Use this for android target.

        callable: customized build function for other backends (e.g. VTA)

    rpc_priority: int, optional
        Priority of this task, used by scheduler in tracker
    rpc_device_key: str, optional
        The device key of registered devices in tracker
    rpc_timeout: int, optional
        Timeout of rpc session
    rpc_tracker_addr: Tuple(str, int), optional
        The address of rpc tracker in Tuple(host, port) format.
        If is set, will use this address.
        If is not set, will use environment variable "TVM_TRACKER_HOST" and "TVM_TRACKER_PORT"

    Returns
    -------
    options: dict
        A dict to store all options
    """
    return {
        'measure_func': measure_func,
        'number': number,
        'repeat': repeat,
        'timeout': timeout,
        'parallel_num': parallel_num,
        'do_fork': do_fork,
        'pack_size': pack_size,
        'check_correctness': check_correctness,

        'rpc_device_key': rpc_device_key,
        'rpc_priority': rpc_priority,
        'rpc_timeout': rpc_timeout,
        'rpc_tracker_addr': rpc_tracker_addr,

        'build_func': build_func,

        'replay_db': replay_db,
        'save_to_replay_db': save_to_replay_db,
    }

def create_measure_batch(task, options):
    """Get a standard measure_batch function.

    Parameters
    ----------
    task: tvm.autotvm.task.Task
        The tuning task
    options: dict
        The option for measuring generated code.
        You should use the return value of :any:`autotvm.measure_option` for this argument

    Returns
    -------
    measure_batch: callable
        a callback function to measure a batch of configs
    """
    from . import measure_methods
    from ..database import filter_inputs

    measure_func = options['measure_func']
    number, repeat = options['number'], options['repeat']
    timeout, parallel_num, do_fork = options['timeout'], options['parallel_num'], options['do_fork']
    pack_size = options['pack_size']
    check_correctness = options['check_correctness']
    rpc_device_key = options['rpc_device_key']
    rpc_priority, rpc_timeout = options['rpc_priority'], options['rpc_timeout']
    build_func = options['build_func']
    replay_db = options['replay_db']
    save_to_replay_db = options['save_to_replay_db']

    kwargs = {}
    executor = LocalExecutor(timeout=timeout)

    if measure_func == 'local':
        if do_fork:
            # start temporary rpc tracker and rpc server for the user
            tracker = Tracker('localhost', port=9000, port_end=10000,
                              silent=True)
            rpc_device_key = '$local$device$%d' % tracker.port
            server = Server('localhost', port=9000, port_end=10000,
                            key=rpc_device_key,
                            use_popen=True, silent=True,
                            tracker_addr=(tracker.host, tracker.port))

            fmeasure = measure_methods.measure_rpc
            kwargs['rpc_device_key'] = rpc_device_key
            kwargs['rpc_tracker_addr'] = (tracker.host, tracker.port)
            kwargs['rpc_timeout'] = timeout
        else:
            fmeasure = measure_methods.measure_local
    elif measure_func == 'rpc':
        fmeasure = measure_methods.measure_rpc
        kwargs['rpc_device_key'] = rpc_device_key
        kwargs['rpc_priority'] = rpc_priority
        kwargs['rpc_timeout'] = rpc_timeout
        assert rpc_device_key, "In rpc mode, a rpc_device_key must be provided"
    else:
        assert callable(measure_func), "In custom mode, custom_measure_func " \
           "must be a callable object"
        fmeasure = measure_func

    if 'cuda' in task.target.keys and 'rpc_device_key' in kwargs:  # query cuda device info
        add_cuda_device_info(kwargs['rpc_device_key'], kwargs.get('rpc_tracker_addr'), kwargs)
    if 'opencl' in task.target.keys and 'rpc_device_key' in kwargs:
        add_opencl_device_info(kwargs['rpc_device_key'], kwargs.get('rpc_tracker_addr'), kwargs)

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
        kwargs['ref_input'], kwargs['ref_output'] = ref_input, ref_output

    def measure_batch(measure_inputs):
        """measure the time cost for a batch of configs in real machines"""
        if replay_db is not None:
            partial_results, measure_inputs =\
                filter_inputs(replay_db, measure_inputs, retry=False)

        # pack configs
        input_packs = []
        for i in range(0, len(measure_inputs), pack_size):
            input_packs.append(measure_inputs[i:i + pack_size])

        # send to measure
        futures = []
        for input_pack in input_packs:
            future = executor.submit(
                fmeasure,
                input_pack,
                number=number,
                repeat=repeat,
                do_fork=do_fork,
                build_func=build_func,
                **kwargs
            )
            futures.append(future)

        # transform results
        results = []
        for future in futures:
            result = future.get()
            if isinstance(result, Exception):
                if measure_func == 'local' and not do_fork:
                    # debug usage, raise exception
                    raise result
                tstamp = time.time()
                results.extend([MeasureResult((result,), MeasureErrorNo.FLEET_ERROR,
                                              timeout, tstamp)] * pack_size)
            else:
                results.extend(result)

        if replay_db is not None:
            if save_to_replay_db:  # save result to database
                for measure_input, result in zip(measure_inputs, results):
                    replay_db.save(measure_input, result)

            result_idx = 0
            for i in range(len(partial_results)):
                if partial_results[i] is None:
                    partial_results[i] = results[result_idx]
                    result_idx += 1
            return partial_results
        return results

    measure_batch.parallel_num = parallel_num
    if measure_func == 'local' and do_fork:
        # attach server and tracker object to avoid them of being garbage-collected
        measure_batch.aux_objects = {"server": server, "tracker": tracker}
    return measure_batch


def add_cuda_device_info(device_key, rpc_tracker_addr, kwargs):
    """Query cuda device info. This is used to set the flags for nvcc compiler
    and check the validity of a generated code."""
    from .measure_methods import request_remote

    remote = request_remote(device_key, rpc_tracker_addr)
    ctx = remote.context('cuda', 0)
    max_dims = ctx.max_thread_dimensions
    kwargs['check_gpu'] = {
        'max_shared_memory_per_block': ctx.max_shared_memory_per_block,
        'max_threads_per_block': ctx.max_threads_per_block,
        'max_thread_x': max_dims[0],
        'max_thread_y': max_dims[1],
        'max_thread_z': max_dims[2],
    }

    kwargs["cuda_arch"] = "sm_" + "".join(ctx.compute_version.split('.'))

def add_opencl_device_info(device_key, rpc_tracker_addr, kwargs):
    """Query opencl device info. This is used to check the validity of a generated code."""
    from .measure_methods import request_remote

    remote = request_remote(device_key, rpc_tracker_addr)
    ctx = remote.context('opencl', 0)
    max_dims = ctx.max_thread_dimensions
    kwargs['check_gpu'] = {
        'max_shared_memory_per_block': ctx.max_shared_memory_per_block,
        'max_threads_per_block': ctx.max_threads_per_block,
        'max_thread_x': max_dims[0],
        'max_thread_y': max_dims[1],
        'max_thread_z': max_dims[2],
    }
