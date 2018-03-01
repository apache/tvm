"""Util to invoke c++ compilers in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import sys
import subprocess
import os

from .._ffi.base import py_str
from .util import tempdir


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

    options : list
        The list of additional options string.

    cc : str, optional
        The compile string.
    """
    if sys.platform == "darwin" or sys.platform.startswith('linux'):
        _linux_shared(output, objects, options, cc)
    elif sys.platform == "win32":
        _windows_shared(output, objects, options)
    else:
        raise ValueError("Unsupported platform")


def _linux_shared(output, objects, options, cc="g++"):
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
        msg += py_str(out)
        raise RuntimeError(msg)


def _windows_shared(output, objects, options):
    cl_cmd = ["cl"]
    cl_cmd += ["-c"]
    if isinstance(objects, str):
        objects = [objects]
    cl_cmd += objects
    if options:
        cl_cmd += options

    temp = tempdir()
    dllmain_path = temp.relpath("dllmain.cc")
    with open(dllmain_path, "w") as dllmain_obj:
        dllmain_obj.write('#include <windows.h>\
BOOL APIENTRY DllMain( HMODULE hModule,\
                       DWORD  ul_reason_for_call,\
                       LPVOID lpReserved)\
{return TRUE;}')

    cl_cmd += [dllmain_path]

    temp_path = dllmain_path.replace("dllmain.cc", "")
    cl_cmd += ["-Fo:" + temp_path]
    try:
        proc = subprocess.Popen(
            cl_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError("can not found cl.exe,"
                           "please run this in Vistual Studio Command Prompt.")
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    link_cmd = ["link"]
    link_cmd += ["-dll", "-FORCE:MULTIPLE"]

    for obj in objects:
        if obj.endswith(".cc"):
            (_, temp_file_name) = os.path.split(obj)
            (shot_name, _) = os.path.splitext(temp_file_name)
            link_cmd += [os.path.join(temp_path, shot_name + ".obj")]
        if obj.endswith(".o"):
            link_cmd += [obj]

    link_cmd += ["-EXPORT:__tvm_main__"]
    link_cmd += [temp_path + "dllmain.obj"]
    link_cmd += ["-out:" + output]

    try:
        proc = subprocess.Popen(
            link_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError("can not found link.exe,"
                           "please run this in Vistual Studio Command Prompt.")
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)

        raise RuntimeError(msg)
