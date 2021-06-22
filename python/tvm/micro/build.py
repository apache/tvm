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

"""Defines top-level glue functions for building microTVM artifacts."""

import copy
import logging
import os
import re
import typing
from tvm.contrib import utils

from .micro_library import MicroLibrary
from .._ffi import libinfo


_LOG = logging.getLogger(__name__)


class Workspace:
    """Defines helper functions for manipulating temporary compilation workspaces."""

    def __init__(self, root=None, debug=False):
        if debug or root is not None:
            with utils.TempDirectory.set_keep_for_debug():
                self.tempdir = utils.tempdir(custom_path=root)
                _LOG.info("Created debug mode workspace at: %s", self.tempdir.temp_dir)
        else:
            self.tempdir = utils.tempdir()

    def relpath(self, path):
        return self.tempdir.relpath(path)

    def listdir(self):
        return self.tempdir.listdir()

    @property
    def path(self):
        return self.tempdir.temp_dir


STANDALONE_CRT_DIR = None


class CrtNotFoundError(Exception):
    """Raised when the standalone CRT dirtree cannot be found."""


def get_standalone_crt_dir() -> str:
    """Find the standalone_crt directory.

    Though the C runtime source lives in the tvm tree, it is intended to be distributed with any
    binary build of TVM. This source tree is intended to be integrated into user projects to run
    models targeted with --runtime=c.

    Returns
    -------
    str :
        The path to the standalone_crt
    """
    global STANDALONE_CRT_DIR
    if STANDALONE_CRT_DIR is None:
        for path in libinfo.find_lib_path():
            crt_path = os.path.join(os.path.dirname(path), "standalone_crt")
            if os.path.isdir(crt_path):
                STANDALONE_CRT_DIR = crt_path
                break

        else:
            raise CrtNotFoundError()

    return STANDALONE_CRT_DIR


def get_standalone_crt_lib(name: str) -> str:
    """Find a source library directory in the standalone_crt.

    The standalone C runtime is split into various libraries (one per directory underneath
    src/runtime/crt). This convenience function returns the full path to one of those libraries
    located in get_standalone_crt_dir().

    Parameters
    ----------
    name : str
        Name of the library subdirectory underneath src/runtime/crt.

    Returns
    -------
    str :
         The full path to the the library.
    """
    return os.path.join(get_standalone_crt_dir(), "src", "runtime", "crt", name)


def get_runtime_libs(executor: str) -> str:
    """Return abspath to all CRT directories in link order which contain
    source (i.e. not header) files.
    """
    if executor == "host-driven":
        crt_runtime_lib_names = ["microtvm_rpc_server", "microtvm_rpc_common", "common"]
    elif executor == "aot":
        crt_runtime_lib_names = ["aot_executor", "common"]
    else:
        raise ValueError(f"Incorrect executor: {executor}")
    return [get_standalone_crt_lib(n) for n in crt_runtime_lib_names]


RUNTIME_SRC_REGEX = re.compile(r"^.*\.cc?$", re.IGNORECASE)


_COMMON_CFLAGS = ["-Wall", "-Werror", "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"]


def _build_default_compiler_options(standalone_crt_dir: typing.Optional[str] = None) -> str:
    """Return a dict containing base compile flags for the CRT under gcc common to .

    Parameters
    ----------
    standalone_crt_dir : Optional[str]
        If given, the path to the standalone_crt
    """
    if standalone_crt_dir is None:
        standalone_crt_dir = get_standalone_crt_dir()
    return {
        "cflags": ["-std=c11"] + _COMMON_CFLAGS,
        "ccflags": ["-std=c++11"] + _COMMON_CFLAGS,
        "ldflags": ["-std=c++11"],
        "include_dirs": [os.path.join(standalone_crt_dir, "include")],
    }


def default_options(crt_config_include_dir, standalone_crt_dir=None):
    """Return default opts passed to Compile commands.

    Parameters
    ----------
    crt_config_include_dir : str
        Path to a directory containing crt_config.h for the target. This will be appended
        to the include path for cflags and ccflags.
    standalone_crt_dir : Optional[str]

    Returns
    -------
    Dict :
        A dictionary containing 3 subkeys, each whose value is _build_default_compiler_options()
        plus additional customization.
         - "bin_opts" - passed as "options" to Compiler.binary() when building MicroBinary.
         - "lib_opts" - passed as "options" to Compiler.library() when building bundled CRT
           libraries (or otherwise, non-generated libraries).
         - "generated_lib_opts" - passed as "options" to Compiler.library() when building the
           generated library.
    """
    bin_opts = _build_default_compiler_options(standalone_crt_dir)
    bin_opts["include_dirs"].append(crt_config_include_dir)

    lib_opts = _build_default_compiler_options(standalone_crt_dir)
    lib_opts["cflags"] = ["-Wno-error=incompatible-pointer-types"]
    lib_opts["include_dirs"].append(crt_config_include_dir)

    generated_lib_opts = copy.copy(lib_opts)

    # Disable due to limitation in the TVM C codegen, which generates lots of local variable
    # declarations at the top of generated code without caring whether they're used.
    # Example:
    #   void* arg0 = (((TVMValue*)args)[0].v_handle);
    #   int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
    generated_lib_opts["cflags"].append("-Wno-unused-variable")
    generated_lib_opts["ccflags"].append("-Wno-unused-variable")

    # Many TVM-intrinsic operators (i.e. expf, in particular)
    generated_lib_opts["cflags"].append("-fno-builtin")

    return {"bin_opts": bin_opts, "lib_opts": lib_opts, "generated_lib_opts": generated_lib_opts}


def build_static_runtime(
    workspace,
    compiler,
    module,
    compiler_options,
    executor=None,
    extra_libs=None,
):
    """Build the on-device runtime, statically linking the given modules.

    Parameters
    ----------
    compiler : tvm.micro.Compiler
        Compiler instance used to build the runtime.

    module : IRModule
        Module to statically link.

    compiler_options : dict
        The return value of tvm.micro.default_options(), with any keys overridden to inject
        compiler options specific to this build. If not given, tvm.micro.default_options() is
        used. This dict contains the `options` parameter passed to Compiler.library() and
        Compiler.binary() at various stages in the compilation process.

    executor : Optional[str]
        Executor used for runtime. Based on this we determine the libraries that need to be
        linked with runtime.

    extra_libs : Optional[List[MicroLibrary|str]]
        If specified, extra libraries to be compiled into the binary. If a MicroLibrary, it is
        included into the binary directly. If a string, the path to a directory; all direct children
        of this directory matching RUNTIME_SRC_REGEX are built into a library. These libraries are
        placed before any common CRT libraries in the link order.

    Returns
    -------
    MicroBinary :
        The compiled runtime.
    """
    mod_build_dir = workspace.relpath(os.path.join("build", "module"))
    os.makedirs(mod_build_dir)
    mod_src_dir = workspace.relpath(os.path.join("src", "module"))

    if not executor:
        executor = "host-driven"

    libs = []
    for mod_or_src_dir in (extra_libs or []) + get_runtime_libs(executor):
        if isinstance(mod_or_src_dir, MicroLibrary):
            libs.append(mod_or_src_dir)
            continue

        lib_src_dir = mod_or_src_dir
        lib_name = os.path.basename(lib_src_dir)
        lib_build_dir = workspace.relpath(f"build/{lib_name}")
        os.makedirs(lib_build_dir)

        lib_srcs = []
        for p in os.listdir(lib_src_dir):
            if RUNTIME_SRC_REGEX.match(p):
                lib_srcs.append(os.path.join(lib_src_dir, p))

        libs.append(compiler.library(lib_build_dir, lib_srcs, compiler_options["lib_opts"]))

    mod_src_dir = workspace.relpath(os.path.join("src", "module"))
    os.makedirs(mod_src_dir)
    libs.append(
        module.export_library(
            mod_build_dir,
            workspace_dir=mod_src_dir,
            fcompile=lambda bdir, srcs, **kwargs: compiler.library(
                bdir, srcs, compiler_options["generated_lib_opts"]
            ),
        )
    )

    runtime_build_dir = workspace.relpath(f"build/runtime")
    os.makedirs(runtime_build_dir)
    return compiler.binary(runtime_build_dir, libs, compiler_options["bin_opts"])
