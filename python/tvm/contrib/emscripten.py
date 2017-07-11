"""Util to invoke emscripten compilers in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import subprocess
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
    cmd += ["-s", "RESERVED_FUNCTION_POINTERS=2"]
    cmd += ["-s", "NO_EXIT_RUNTIME=1"]
    cmd += ["-Oz"]
    cmd += ["-o", output]
    if side_module:
        cmd += ["-s", "SIDE_MODULE=1"]

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

    args = ' '.join(cmd)
    proc = subprocess.Popen(
        args, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += out
        raise RuntimeError(msg)
