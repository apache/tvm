# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Util to invoke C/C++ compilers in the system."""
import os
import shutil
import subprocess

# pylint: disable=invalid-name
import sys
from typing import Dict

from .._ffi.base import py_str
from . import tar as _tar
from . import utils as _utils


def _is_linux_like():
    return (
        sys.platform == "darwin"
        or sys.platform.startswith("linux")
        or sys.platform.startswith("freebsd")
    )


def _is_windows_like():
    return sys.platform == "win32"


def get_cc():
    """Return the path to the default C/C++ compiler.

    Returns
    -------
    out: Optional[str]
        The path to the default C/C++ compiler, or None if none was found.
    """

    if not _is_linux_like():
        return None

    env_cxx = os.environ.get("CXX") or os.environ.get("CC")
    if env_cxx:
        return env_cxx
    cc_names = ["g++", "gcc", "clang++", "clang", "c++", "cc"]
    dirs_in_path = os.get_exec_path()
    for cc in cc_names:
        for d in dirs_in_path:
            cc_path = os.path.join(d, cc)
            if os.path.isfile(cc_path) and os.access(cc_path, os.X_OK):
                return cc_path
    return None


def create_shared(output, objects, options=None, cc=None, cwd=None, ccache_env=None):
    """Create shared library.

    Parameters
    ----------
    output : str
        The target shared library.

    objects : List[str]
        List of object files.

    options : List[str]
        The list of additional options string.

    cc : Optional[str]
        The compiler command.

    cwd : Optional[str]
        The current working directory.

    ccache_env : Optional[Dict[str, str]]
        The environment variable for ccache. Set `None` to disable ccache by default.
    """
    cc = cc or get_cc()

    if _is_linux_like():
        _linux_compile(output, objects, options, cc, cwd, ccache_env, compile_shared=True)
    elif _is_windows_like():
        _windows_compile(output, objects, options, cwd, ccache_env)
    else:
        raise ValueError("Unsupported platform")


def _linux_ar(output, inputs, ar):
    ar = ar or "ar"

    libname = os.path.basename(output)
    if not libname.startswith("lib"):
        libname = "lib" + libname
    temp = _utils.tempdir()
    temp_output = temp.relpath(libname)
    cmd = [ar, "-crs", temp_output]

    # handles the case where some input files are tar of objects
    # unpack them and return the list of files inside
    objects = _tar.normalize_file_list_by_unpacking_tars(temp, inputs)

    cmd += objects
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "AR error:\n"
        msg += py_str(out)
        msg += "\nCommand line: " + " ".join(cmd)
        raise RuntimeError(msg)

    shutil.move(temp_output, output)


def create_staticlib(output, inputs, ar=None):
    """Create static library.

    Parameters
    ----------
    output : str
        The target shared library.

    inputs : List[str]
        List of inputs files. Each input file can be a tarball
        of objects or an object file.

    ar : Optional[str]
        Path to the ar command to be used
    """

    if _is_linux_like():
        return _linux_ar(output, inputs, ar)
    else:
        raise ValueError("Unsupported platform")


def create_executable(output, objects, options=None, cc=None, cwd=None, ccache_env=None):
    """Create executable binary.

    Parameters
    ----------
    output : str
        The target executable.

    objects : List[str]
        List of object files.

    options : List[str]
        The list of additional options string.

    cc : Optional[str]
        The compiler command.

    cwd : Optional[str]
        The urrent working directory.

    ccache_env : Optional[Dict[str, str]]
        The environment variable for ccache. Set `None` to disable ccache by default.
    """
    cc = cc or get_cc()

    if _is_linux_like():
        _linux_compile(output, objects, options, cc, cwd, ccache_env)
    elif _is_windows_like():
        _windows_compile(output, objects, options, cwd, ccache_env)
    else:
        raise ValueError("Unsupported platform")


def get_global_symbol_section_map(path, *, nm=None) -> Dict[str, str]:
    """Get global symbols from a library via nm -g

    Parameters
    ----------
    path : str
        The library path

    nm: str
        The path to nm command

    Returns
    -------
    symbol_section_map: Dict[str, str]
        A map from defined global symbol to their sections
    """
    if nm is None:
        if not _is_linux_like():
            raise ValueError("Unsupported platform")
        nm = "nm"

    symbol_section_map = {}

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist")

    cmd = [nm, "-gU", path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Runtime error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    for line in py_str(out).split("\n"):
        data = line.strip().split()
        if len(data) != 3:
            continue
        symbol = data[-1]
        section = data[-2]
        symbol_section_map[symbol] = section
    return symbol_section_map


def get_target_by_dump_machine(compiler):
    """Functor of get_target_triple that can get the target triple using compiler.

    Parameters
    ----------
    compiler : Optional[str]
        The compiler.

    Returns
    -------
    out: Callable
        A function that can get target triple according to dumpmachine option of compiler.
    """

    def get_target_triple():
        """Get target triple according to dumpmachine option of compiler."""
        if compiler:
            cmd = [compiler, "-dumpmachine"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (out, _) = proc.communicate()
            if proc.returncode != 0:
                msg = "dumpmachine error:\n"
                msg += py_str(out)
                return None
            return py_str(out)
        return None

    return get_target_triple


# assign so as default output format
create_shared.output_format = "so" if sys.platform != "win32" else "dll"
create_shared.get_target_triple = get_target_by_dump_machine(os.environ.get("CXX", get_cc()))


def cross_compiler(
    compile_func, options=None, output_format=None, get_target_triple=None, add_files=None
):
    """Create a cross compiler function by specializing compile_func with options.

    This function can be used to construct compile functions that
    can be passed to AutoTVM measure or export_library.


    Parameters
    ----------
    compile_func : Union[str, Callable[[str, str, Optional[str]], None]]
        Function that performs the actual compilation

    options : Optional[List[str]]
        List of additional optional string.

    output_format : Optional[str]
        Library output format.

    get_target_triple: Optional[Callable]
        Function that can target triple according to dumpmachine option of compiler.

    add_files: Optional[List[str]]
        List of paths to additional object, source, library files
        to pass as part of the compilation.

    Returns
    -------
    fcompile : Callable[[str, str, Optional[str]], None]
        A compilation function that can be passed to export_library.

    Examples
    --------
    .. code-block:: python

       from tvm.contrib import cc, ndk
       # export using arm gcc
       mod = build_runtime_module()
       mod.export_library(path_dso,
                          fcompile=cc.cross_compiler("arm-linux-gnueabihf-gcc"))
       # specialize ndk compilation options.
       specialized_ndk = cc.cross_compiler(
           ndk.create_shared,
           ["--sysroot=/path/to/sysroot", "-shared", "-fPIC", "-lm"])
       mod.export_library(path_dso, fcompile=specialized_ndk)
    """
    base_options = [] if options is None else options
    kwargs = {}
    add_files = [] if add_files is None else add_files

    # handle case where compile_func is the name of the cc
    if isinstance(compile_func, str):
        kwargs = {"cc": compile_func}
        compile_func = create_shared

    def _fcompile(outputs, objects, options=None):
        all_options = base_options
        if options is not None:
            all_options += options
        compile_func(outputs, objects + add_files, options=all_options, **kwargs)

    if not output_format and hasattr(compile_func, "output_format"):
        output_format = compile_func.output_format
    output_format = output_format if output_format else "so"

    if not get_target_triple and hasattr(compile_func, "get_target_triple"):
        get_target_triple = compile_func.get_target_triple

    _fcompile.output_format = output_format
    _fcompile.get_target_triple = get_target_triple
    return _fcompile


def _linux_compile(
    output, objects, options, compile_cmd, cwd=None, ccache_env=None, compile_shared=False
):
    cmd = [compile_cmd]
    if compile_cmd != "nvcc":
        if compile_shared or output.endswith(".so") or output.endswith(".dylib"):
            cmd += ["-shared", "-fPIC"]
            if sys.platform == "darwin":
                cmd += ["-undefined", "dynamic_lookup"]
        elif output.endswith(".obj"):
            cmd += ["-c"]
    else:
        if compile_shared or output.endswith(".so") or output.endswith(".dylib"):
            cmd += ["-shared"]
    cmd += ["-o", output]
    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects
    if options:
        cmd += options
    env = None
    if ccache_env is not None:
        if shutil.which("ccache"):
            cmd.insert(0, "ccache")
            env = os.environ.copy()
            env.update(ccache_env)
        else:
            raise ValueError("ccache not found")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        msg += "\nCommand line: " + " ".join(cmd)
        raise RuntimeError(msg)


def _windows_compile(output, objects, options, cwd=None, ccache_env=None):
    compiler = os.getenv("TVM_WIN_CC", default="clang")
    win_target = os.getenv("TVM_WIN_TARGET", default="x86_64")
    cmd = [compiler]
    cmd += ["-O2"]
    cmd += ["--target=" + win_target]

    if output.endswith(".so") or output.endswith(".dll"):
        cmd += ["-shared"]
    elif output.endswith(".obj"):
        cmd += ["-c"]

    if isinstance(objects, str):
        objects = [objects]
    cmd += ["-o", output]
    cmd += objects
    if options:
        cmd += options
    env = None
    if ccache_env is not None:
        if shutil.which("ccache"):
            cmd.insert(0, "ccache")
            env = os.environ.copy()
            env.update(ccache_env)
        else:
            raise ValueError("ccache not found")

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env
        )
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError(
            "Can not find the LLVM clang for Windows clang.exe)."
            "Make sure it's installed"
            " and the installation directory is in the %PATH% environment "
            "variable. Prebuilt binaries can be found at: https://llvm.org/"
        )
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += " ".join(cmd) + "\n"
        msg += py_str(out)

        raise RuntimeError(msg)
