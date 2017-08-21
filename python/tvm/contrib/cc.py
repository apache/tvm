"""Util to invoke c++ compilers in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import sys
import subprocess

def create_shared(output,
                  objects,
                  options=None,
                  cc="g++",
                  cross_compile=False):
    """Create shared library.

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

    cross_compile : bool, optional
        Do not add additional compile options
        (except for those specfied in argument 'options')
        if it is cross compiling.
    """
    cmd = [cc]
    if not cross_compile:
        cmd += ["-shared", "-fPIC"]

        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]

    cmd += ["-o", output]

    if isinstance(objects, str):
        cmd += [objects]
    else:
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
