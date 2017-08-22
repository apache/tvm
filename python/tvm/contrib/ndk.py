"""Util to invoke NDK compiler toolchain."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import sys
import subprocess
import os
from . import cc

def create_shared(output,
                  objects,
                  options=["-shared", "-fPIC"]):
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
    cc.create_shared(output, objects, options, cc=compiler, cross_compile=True)
