# pylint: disable=invalid-name
"""Util to compile with C++ code"""
from __future__ import absolute_import as _abs
import sys
import subprocess

def create_shared(path_target, objects,
                  options=None, cc="g++"):
    """Create shared library.

    Parameters
    ----------
    path_target : str
        The target shared library.

    objects : list
        List of object files.

    options : str
        The additional options.

    cc : str
        The compile string.
    """
    cmd = [cc]
    cmd += ["-shared"]
    cmd += ["-o", path_target]
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
        sys.stderr.write("Compilation error:\n")
        sys.stderr.write(out)
        sys.stderr.flush()
