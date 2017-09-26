"""Util to invoke c++ compilers in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import sys
import subprocess

def create_shared(output,
                  objects,
                  options=None,
                  cc="g++"):
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
    """
    cmd = [cc]
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

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += out
        raise RuntimeError(msg)
