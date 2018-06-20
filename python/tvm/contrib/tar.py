
"""Util to invoke tarball in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import os
import shutil
import subprocess
from . import util
from .._ffi.base import py_str

def tar(output, files):
    """Create tarball containing all files in root.

    Parameters
    ----------
    output : str
        The target shared library.

    files : list
        List of files to be bundled.
    """
    cmd = ["tar"]
    cmd += ["-czf"]
    temp = util.tempdir()
    fset = set()
    for fname in files:
        base = os.path.basename(fname)
        if base in fset:
            raise ValueError("duplicate file name %s" % base)
        fset.add(base)
        shutil.copy(fname, temp.relpath(base))
    cmd += [output]
    cmd += ["-C", temp.temp_dir]
    cmd += temp.listdir()
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Tar error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)


def untar(tar_file, directory):
    """Unpack all tar files into the directory

    Parameters
    ----------
    tar_file : str
        The source tar file.

    directory : str
        The target directory
    """
    cmd = ["tar"]
    cmd += ["-xf"]
    cmd += [tar_file]
    cmd += ["-C", directory]
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Tar error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
