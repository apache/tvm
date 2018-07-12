"""
Decorator functions for hashing schedule code

code hashing is used to check the consistence of schedule code and the parameters loaded from log
"""
import inspect
import zlib

from tvm import schedule

def attach_code_hash(s):
    """Decorator for attaching a code hash to a schedule

    Parameters
    ----------
    s: Schedule
        tvm.schedule.Schedule to attach the hash to
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            raw_hash = zlib.crc32(''.join(inspect.getsourcelines(func)[0]).encode())
            s.code_hash = hex(raw_hash)[2:]
        return wrapper
    return decorator

def attach_code_hash_to_arg(arg_idx=1):
    """Decorator for attaching a code hash to a schedule

    Parameters
    ----------
    arg_idx: int
        index of the argument (expected to be a Schedule) to attach the code
        hash to
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            assert isinstance(args[arg_idx], schedule.Schedule)
            raw_hash = zlib.crc32(''.join(inspect.getsourcelines(func)[0]).encode())
            args[arg_idx].code_hash = hex(raw_hash)[2:]
        return wrapper
    return decorator
