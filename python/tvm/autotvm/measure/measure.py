# pylint: disable=pointless-string-statement,consider-using-enumerate,invalid-name
"""User facing API for specifying how to measure the generated code"""
from collections import namedtuple

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


def measure_option(measure_func,
                   number=1,
                   repeat=1,
                   timeout=60,
                   n_parallel=1,
                   do_fork=True,
                   build_func='default',
                   check_correctness=False,
                   replay_db=None):
    """Configure how to do measurement

    Parameters
    ----------
    measure_func: str or callable
        'local': use the local device for measurement. The tuner will start a tracker
        and a RPC server silently for the user.

        callable: It is a callable function for measurement.
                  See the return value of measure/measure_methods.py::rpc for example.
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
    n_parallel: int, optional
        The number of measurement task that can run in parallel.
        Set this according to the number of cpu cores (for compilation) and
        the number of devices you have (for measuring generate code).
    do_fork: bool, optional
        Whether use multiprocessing (based on fork) for running measure jobs in parallel.
        Set this to False if you want to debug (see trackback) or using fork is not suitable.
        NOTE: If this is False, parallel and timeout do not work.
    build_func: str or callable, optional
        'default': call default builder. This works for normal target (llvm, cuda)

        'ndk': use Android NDK to create shared library. Use this for android target.

        callable: customized build function for other backends (e.g. VTA).
                  See measure/measure_methods.py::default_build_func for example.
    check_correctness: bool
        Whether check correctness after measurement. This will use llvm cpu as reference.
    replay_db : Database, optional
        The database that we retrieve saved MeasureResult from.

    Returns
    -------
    options: dict
        A dict to store all options

    Note
    ----
    To support customized measure, you can pass callable `measure_func` or
    `build_func` in. The `measure_func` will call `build_func` to build binary library
    and handle the logic of measurement.

    Signature:
    * measure_func (see the return value of measure/measure_methods.py::rpc for example)
    def measure_func(input_pack, build_func, build_kwargs, number, repeat, ref_input, ref_output):
        return measure_results

    * build_func (see measure/measure_methods.py::default_build_func for example)
    def build_func(inp, tmp_dir, **kwargs):
        return func, args, filename
    """
    return {
        'measure_func': measure_func,
        'number': number,
        'repeat': repeat,
        'timeout': timeout,
        'n_parallel': n_parallel,
        'do_fork': do_fork,
        'build_func': build_func,
        'check_correctness': check_correctness,
        'replay_db': replay_db,
    }
