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
from tvm.contrib import util


_LOG = logging.getLogger(__name__)


class Workspace:
    """Defines helper functions for manipulating temporary compilation workspaces."""

    def __init__(self, root=None, debug=False):
        if debug or root is not None:
            with util.TempDirectory.set_keep_for_debug():
                self.tempdir = util.tempdir(custom_path=root)
                _LOG.info("Created debug mode workspace at: %s", self.tempdir.temp_dir)
        else:
            self.tempdir = util.tempdir()

    def relpath(self, path):
        return self.tempdir.relpath(path)

    def listdir(self):
        return self.tempdir.listdir()

    @property
    def path(self):
        return self.tempdir.temp_dir


# Required C runtime libraries, in link order.
CRT_RUNTIME_LIB_NAMES = ["utvm_rpc_server", "utvm_rpc_common", "common"]


TVM_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


CRT_ROOT_DIR = os.path.join(TVM_ROOT_DIR, "src", "runtime", "crt")


RUNTIME_LIB_SRC_DIRS = [os.path.join(CRT_ROOT_DIR, n) for n in CRT_RUNTIME_LIB_NAMES] + [
    os.path.join(TVM_ROOT_DIR, "3rdparty/libcrc/src")
]


RUNTIME_SRC_REGEX = re.compile(r"^.*\.cc?$", re.IGNORECASE)


_CRT_DEFAULT_OPTIONS = {
    "ccflags": ["-std=c++11"],
    "ldflags": ["-std=gnu++14"],
    "include_dirs": [
        f"{TVM_ROOT_DIR}/include",
        f"{TVM_ROOT_DIR}/3rdparty/dlpack/include",
        f"{TVM_ROOT_DIR}/3rdparty/libcrc/include",
        f"{TVM_ROOT_DIR}/3rdparty/dmlc-core/include",
        f"{CRT_ROOT_DIR}/include",
    ],
    "profile": {"common": ["-Wno-unused-variable"]},
}


def default_options(target_include_dir):
    """Return default opts passed to Compile commands."""
    bin_opts = copy.deepcopy(_CRT_DEFAULT_OPTIONS)
    bin_opts["include_dirs"].append(target_include_dir)
    lib_opts = copy.deepcopy(bin_opts)
    lib_opts["profile"]["common"].append("-Werror")
    lib_opts["cflags"] = ["-Wno-error=incompatible-pointer-types"]
    return {"bin_opts": bin_opts, "lib_opts": lib_opts}


def build_static_runtime(workspace, compiler, module, lib_opts=None, bin_opts=None):
    """Build the on-device runtime, statically linking the given modules.

    Parameters
    ----------
    compiler : tvm.micro.Compiler
        Compiler instance used to build the runtime.

    module : IRModule
        Module to statically link.

    lib_opts : dict
        Extra kwargs passed to library(),

    bin_opts : dict
        Extra kwargs passed to binary(),

    Returns
    -------
    MicroBinary :
        The compiled runtime.
    """
    lib_opts = _CRT_DEFAULT_OPTIONS if lib_opts is None else lib_opts
    bin_opts = _CRT_DEFAULT_OPTIONS if bin_opts is None else bin_opts

    mod_build_dir = workspace.relpath(os.path.join("build", "module"))
    os.makedirs(mod_build_dir)
    mod_src_dir = workspace.relpath(os.path.join("src", "module"))
    os.makedirs(mod_src_dir)
    mod_src_path = os.path.join(mod_src_dir, "module.c")
    module.save(mod_src_path, "cc")

    libs = []
    for lib_src_dir in RUNTIME_LIB_SRC_DIRS:
        lib_name = os.path.basename(lib_src_dir)
        lib_build_dir = workspace.relpath(f"build/{lib_name}")
        os.makedirs(lib_build_dir)

        lib_srcs = []
        for p in os.listdir(lib_src_dir):
            if RUNTIME_SRC_REGEX.match(p):
                lib_srcs.append(os.path.join(lib_src_dir, p))

        libs.append(compiler.library(lib_build_dir, lib_srcs, lib_opts))

    libs.append(compiler.library(mod_build_dir, [mod_src_path], lib_opts))

    runtime_build_dir = workspace.relpath(f"build/runtime")
    os.makedirs(runtime_build_dir)
    return compiler.binary(runtime_build_dir, libs, bin_opts)
