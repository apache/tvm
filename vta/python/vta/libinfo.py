"""Library information."""
from __future__ import absolute_import
import sys
import os

def _get_lib_name():
    if sys.platform.startswith('win32'):
        return "vta.dll"
    if sys.platform.startswith('darwin'):
        return "libvta.dylib"
    return "libvta.so"


def find_libvta(optional=False):
    """Find VTA library"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_search = [curr_path]
    lib_search += [os.path.join(curr_path, "..", "..", "..", "build",)]
    lib_search += [os.path.join(curr_path, "..", "..", "..", "build", "Release")]
    lib_name = _get_lib_name()
    lib_path = [os.path.join(x, lib_name) for x in lib_search]
    lib_found = [x for x in lib_path if os.path.exists(x)]
    if not lib_found and not optional:
        raise RuntimeError("Cannot find libvta: candidates are: " % str(lib_path))
    return lib_found
