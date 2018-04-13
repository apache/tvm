"""Utilities to start simulator."""
import os
import ctypes
import json
import tvm

def _load_lib():
    """Load local library, assuming they are simulator."""
    # pylint: disable=unused-variable
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [
        os.path.abspath(os.path.join(curr_path, "../../../lib/libvta.so")),
        os.path.abspath(os.path.join(curr_path, "../../../lib/libvta_runtime.so"))
    ]
    runtime_dll = []
    if not all(os.path.exists(f) for f in dll_path):
        return []
    try:
        for fname in dll_path:
            runtime_dll.append(ctypes.CDLL(fname, ctypes.RTLD_GLOBAL))
        return runtime_dll
    except OSError:
        return []


def enabled():
    """Check if simulator is enabled."""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    return f is not None


def clear_stats():
    """Clear profiler statistics"""
    f = tvm.get_global_func("vta.simulator.profiler_clear", True)
    if f:
        f()


def stats():
    """Clear profiler statistics

    Returns
    -------
    stats : dict
        Current profiler statistics
    """
    x = tvm.get_global_func("vta.simulator.profiler_status")()
    return json.loads(x)


LIBS = _load_lib()
