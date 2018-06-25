# pylint: disable=pointless-string-statement,consider-using-enumerate,invalid-name
"""User facing API for specifying how to measure the generated code"""
import time
from collections import namedtuple

from ...contrib.util import tempdir
from ...contrib.rpc.tracker import Tracker
from ...contrib.rpc.server import Server

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
        if no error occurs for this measure, it is an array of measured running times
        if some error occurs during the measure, it is an array of the exception objections.
    error_no: int
        denote error type, defined by MeasureErrorNo
    all_cost: float
        all cost of this measure, including rpc, compilation, test runs
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


def measure_option(mode,
                   number=1,
                   repeat=1,
                   timeout=60,
                   parallel_num=1,
                   pack_size=1,
                   check_correctness=False,
                   build_option=None,

                   replay_db=None,
                   save_to_replay_db=False,

                   rpc_device_key=None,
                   rpc_priority=1,
                   rpc_timeout=60,
                   use_ndk=False,
                   custom_measure_batch=None):
    """Configure how to do measurement

    Parameters
    ----------
    mode: str
        'local': use the local device for measurement

        'rpc': request devices for measurement from rpc tracker

        'custom': use custom measure function

        'local-nofork': use local device for measure but does not use multiprocessing.
        This mode is suitable for debug, but does not support timeout and parallel.

    number : int, optional
        Number of times to do the measurement for average
    repeat : int, optional
        Number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up. The returned result contains `repeat` costs,
        each of which is the average of `number` test run.
    timeout: int, optional
        Timeout for a whole batch. TimeoutError will be returned as the result.
    parallel_num: int, optional
        The number of measurement task that can run in parallel
        Set this according to the number of cpu cores (for compilation) and
        the number of devices you have (for measuring generate code).
    pack_size : int, optional
        Number of configs to measure in one RPC call.
        Usually this can be set to 1. If your device has high cost to establish a rpc connection,
        set this higher.
    check_correctness: bool
        Whether check correctness after measurement.
    build_option: Dict, optional
        Build options for tvm.build_config

    replay_db : Database, optional
        The database that we retrieve saved MeasureResults from
    save_to_replay_db: bool, optional
        Whether save measure result to database. This is useless when replay_db is None

    rpc_priority: int, optional
        Priority of this task, used by scheduler in tracker
    rpc_device_key: str, optional
        The device key of registered devices in tracker
    rpc_timeout: int, optional
        Timeout of rpc session
    use_ndk: bool, option
        Whether export requires ndk

    custom_measure_batch: callable, optional
        custom measure function

    Returns
    -------
    options: dict
        A dict to store all options
    """
    return {
        'mode': mode,
        'number': number,
        'repeat': repeat,
        'timeout': timeout,
        'parallel_num': parallel_num,
        'pack_size': pack_size,
        'check_correctness': check_correctness,
        'build_option': build_option,

        'replay_db': replay_db,
        'save_to_replay_db': save_to_replay_db,

        'rpc_device_key': rpc_device_key,
        'rpc_priority': rpc_priority,
        'rpc_timeout': rpc_timeout,
        'use_ndk': use_ndk,

        'custom_measure_batch': custom_measure_batch
    }


def create_measure_batch(options):
    """Get a standard measure_batch function.

    Parameters
    ----------
    options: dict
        The option for measuring generated code.
        You should use the return value of :any:`autotvm.measure_option` for this argument

    Returns
    -------
    measure_batch: callable
        a callback function to measure a batch of configs
    """
    from . import measure_methods

    mode = options['mode']
    number, repeat = options['number'], options['repeat']
    timeout, parallel_num = options['timeout'], options['parallel_num']
    pack_size = options['pack_size']
    check_correctness = options['check_correctness']
    build_option = options['build_option']
    replay_db = options['replay_db']
    save_to_replay_db = options['save_to_replay_db']
    rpc_device_key = options['rpc_device_key']
    rpc_priority, rpc_timeout = options['rpc_priority'], options['rpc_timeout']
    use_ndk = options['use_ndk']
    custom_measure_batch = options['custom_measure_batch']

    kwargs = {}
    executor = LocalExecutor(timeout=timeout)

    if mode == 'local':
        # start temporary rpc tracker and rpc server for the user
        tracker = Tracker('localhost', port=9000, port_end=10000,
                          silent=True)
        rpc_device_key = '$local$device$%d' % tracker.port
        server = Server('localhost', port=9000, port_end=10000,
                        key=rpc_device_key,
                        use_popen=True, silent=True,
                        tracker_addr=(tracker.host, tracker.port))

        fmeasure = measure_methods.measure_rpc
        kwargs['device_key'] = rpc_device_key
        kwargs['rpc_tracker_addr'] = (tracker.host, tracker.port)
        kwargs['rpc_timeout'] = timeout
        kwargs['tmp_dir'] = tempdir()
    elif mode == 'rpc':
        fmeasure = measure_methods.measure_rpc
        kwargs['device_key'] = rpc_device_key
        kwargs['rpc_priority'] = rpc_priority
        kwargs['rpc_timeout'] = rpc_timeout
        kwargs['use_ndk'] = use_ndk
        kwargs['tmp_dir'] = tempdir()
        assert rpc_device_key, "In rpc mode, a rpc_device_key must be provided"
    elif mode == "custom":
        assert callable(custom_measure_batch), "In custom mode, custom_measure_func " \
                                               "must be a callable object"
    elif mode == 'local-nofork':
        fmeasure = measure_methods.measure_local
        kwargs['fork_new_process'] = False
    else:
        raise RuntimeError("Invalid mode: " + mode)

    def measure_batch(measure_inputs):
        """measure the time cost for a batch of configs in real machines"""
        if replay_db is not None:
            partial_results, measure_inputs =\
                replay_db.filter_inputs(measure_inputs, retry=False)

        # pack configs
        input_packs = []
        for i in range(0, len(measure_inputs), pack_size):
            input_packs.append(measure_inputs[i:i + pack_size])

        # send to measure
        futures = []
        for input_pack in input_packs:
            future = executor.submit(
                fmeasure, input_pack,
                number=number,
                repeat=repeat,
                check_correctness=check_correctness,
                build_option=build_option,
                **kwargs
            )
            futures.append(future)

        # transform results
        results = []
        for future in futures:
            result = future.get()
            if isinstance(result, Exception):
                if mode == 'local-nofork':
                    # debug usage, raise exception
                    raise result
                tstamp = time.time()
                results.extend([MeasureResult((result,), MeasureErrorNo.FLEET_ERROR,
                                              timeout, tstamp)] * pack_size)
            else:
                results.extend(result)

        if replay_db is not None:
            if save_to_replay_db:  # save result to database
                for i in range(len(results)):
                    replay_db.save(measure_inputs[i], results[i])

            result_idx = 0
            for i in range(len(partial_results)):
                if partial_results[i] is None:
                    partial_results[i] = results[result_idx]
                    result_idx += 1
            return partial_results
        return results

    if mode == 'custom':
        measure_batch = custom_measure_batch

    measure_batch.parallel_num = parallel_num
    if mode == 'local':
        measure_batch.aux_objects = (server, tracker)
    return measure_batch
