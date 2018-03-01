"""Util to invoke emscripten compilers in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import subprocess
from .._ffi.base import py_str
from .._ffi.libinfo import find_lib_path

def create_js(output,
              objects,
              options=None,
              side_module=False,
              cc="emcc"):
    """Create emscripten javascript library.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : list
        List of object files.

    options : str
        The additional options.

    cc : str, optional
        The compile string.
    """
    cmd = [cc]
    cmd += ["-Oz"]
    if not side_module:
        cmd += ["-s", "RESERVED_FUNCTION_POINTERS=2"]
        cmd += ["-s", "NO_EXIT_RUNTIME=1"]
        extra_methods = ['cwrap', 'getValue', 'setValue', 'addFunction']
        cfg = "[" + (','.join("\'%s\'" % x for x in extra_methods)) + "]"
        cmd += ["-s", "EXTRA_EXPORTED_RUNTIME_METHODS=" + cfg]
    else:
        cmd += ["-s", "SIDE_MODULE=1"]
    cmd += ["-o", output]
    objects = [objects] if isinstance(objects, str) else objects
    with_runtime = False
    for obj in objects:
        if obj.find("libtvm_web_runtime.bc") != -1:
            with_runtime = True

    if not with_runtime and not side_module:
        objects += [find_lib_path("libtvm_web_runtime.bc")[0]]

    cmd += objects

    if options:
        cmd += options

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

create_js.object_format = "bc"
