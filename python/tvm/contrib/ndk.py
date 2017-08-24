"""Util to invoke NDK compiler toolchain."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import subprocess
import os

def create_shared(output,
                  objects,
                  options="-shared -fPIC"):
    """Create shared library.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : list
        List of object files.

    options : str
        The additional options.
    """
    if "TVM_NDK_CC" not in os.environ:
        raise RuntimeError("Require environment variable TVM_NDK_CC"
                           " to be the NDK standalone compiler")
    compiler = os.environ["TVM_NDK_CC"]
    cmd = [compiler]
    cmd += ["-o", output]

    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects

    if options:
        cmd += options.split()

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
