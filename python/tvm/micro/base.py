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
"""Base definitions for MicroTVM"""

from __future__ import absolute_import

import os
import re
import sys
import tarfile
from enum import Enum

import tvm
import tvm._ffi

from tvm.contrib import util as _util
from tvm.contrib import cc as _cc

# all sections that comprise a device's memory layout, in order from lowest
# starting address to highest
DEVICE_SECTIONS = [
    "text",
    "rodata",
    "data",
    "bss",
    "args",
    "heap",
    "workspace",
    "stack",
]

class LibType(Enum):
    """Enumeration of library types that can be compiled and loaded onto a device"""
    # library to be used as a MicroTVM runtime
    RUNTIME = 0
    # library to be used as an operator
    OPERATOR = 1


def _calc_max_workspace_usage(src):
    # TODO factor in alignment to the calculation (alloc sizes will be aligned up to the word size)
    alloc_re = re.compile(
        r'.*\* ?(.+) = (\(.+\))? TVMBackendAllocWorkspace\(.+, .+, \(uint64_t\)(.+), .+, .+\).*')
    free_re = re.compile(r'.*if \(TVMBackendFreeWorkspace\(.+, .+, (\(void\*\))? (.+)\) != 0\) {.*')
    max_usage = 0
    alloc_map = {}
    for line in src.split("\n"):
        if line.strip().startswith("//"):
            continue
        match = alloc_re.match(line)
        if match is not None:
            alloc_map[match.group(1)] = int(match.group(3))
            max_usage = max(max_usage, sum(alloc_map.values()))
        else:
            match = free_re.match(line)
            if match is not None:
                print(alloc_map)
                del alloc_map[match.group(2)]
    return max_usage


def create_micro_mod(c_mod, dev_config, lib_src_paths=None, lib_headers=None,
                     lib_include_paths=None):
    """Produces a micro module from a given module.

    Parameters
    ----------
    c_mod : tvm.module.Module
        module with "c" as its target backend

    lib_src_paths: TODO
        TODO

    lib_headers: TODO
        TODO

    lib_include_paths: TODO
        TODO

    Return
    ------
    micro_mod : tvm.module.Module
        micro module for the target device
    """
    temp_dir = _util.tempdir()
    lib_obj_path = temp_dir.relpath("dev_lib.obj")
    # TODO use dev config to dispatch on the type of C codegen to run through
    # (e.g., CodeGenCArm, CodeGenCHost, CodeGenCRiscV)
    c_mod.export_library(
        lib_obj_path,
        fcompile=cross_compiler(
            dev_config,
            LibType.OPERATOR,
            lib_src_paths=lib_src_paths,
            lib_headers=lib_headers,
            lib_include_paths=lib_include_paths))
    micro_mod = tvm.runtime.load_module(lib_obj_path)
    return micro_mod


def cross_compiler(dev_config, lib_type, lib_src_paths=None, lib_headers=None,
                   lib_include_paths=None):
    """Create a cross compile function that wraps `create_lib` for a `Binutil` instance.

    For use in `tvm.runtime.Module.export_library`.

    Parameters
    ----------
    create_micro_lib : func
        function for creating MicroTVM libraries for a specific device (e.g.,
        `tvm.micro.device.get_device_funcs('arm.stm32f746xx')['create_micro_lib']`)

    lib_type : micro.LibType
        whether to compile a MicroTVM runtime or operator library

    lib_src_paths: TODO
        TODO

    lib_headers: TODO
        e.g., `['cmsis_gcc.h', 'arm_math.h']`

    lib_include_paths: TODO
        TODO

    Return
    ------
    func : Callable[[str, str, Optional[str]], None]
        cross compile function taking a destination path for the object file
        and a path for the input source file.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      fcompile = tvm.micro.cross_compiler(dev_config, LibType.OPERATOR)
      c_mod.export_library('dev_lib.obj', fcompile=fcompile)
    """
    assert (lib_headers is None) == (lib_include_paths is None), \
        "must specify both `lib_headers` and `lib_include_paths` or neither"

    if lib_src_paths is None:
        lib_src_paths = []
    if lib_include_paths is None:
        lib_include_paths = []
    include_options = []
    for include_path in lib_include_paths:
        include_options.append("-I")
        include_options.append(include_path)
    create_micro_lib = tvm.micro.device.get_device_funcs(
        dev_config["device_id"])["create_micro_lib"]
    mem_layout = dev_config["mem_layout"]

    def compile_func(obj_path, src_path, **kwargs):
        if isinstance(obj_path, list):
            obj_path = obj_path[0]
        if isinstance(src_path, list):
            src_path = src_path[0]
        options = kwargs.get("options", [])
        options += include_options

        # check that workspace allocations don't exceed available workspace memory
        with open(src_path) as f:
            src_contents = f.read()
            max_ws_usage = _calc_max_workspace_usage(src_contents)
            available_mem = mem_layout["workspace"]["size"]
            if max_ws_usage > available_mem:
                raise RuntimeError(f"workspace allocations in library ({max_ws_usage}) "
                                   f"exceed available memory ({available_mem})")
        # inject headers into new source path, if requested
        if lib_headers:
            headers_to_inject = "\n".join(map(lambda s: f"#include <{s}>", lib_headers)) + "\n"
            new_src_contents = headers_to_inject + src_contents
            tmp_dir = _util.tempdir()
            src_path = tmp_dir.relpath(os.path.basename(src_path))
            with open(src_path, "w") as f:
                f.write(new_src_contents)

        create_micro_lib(obj_path, src_path, lib_type, options, lib_src_paths=lib_src_paths)
    return _cc.cross_compiler(compile_func, output_format="obj")


def get_micro_host_driven_dir():
    """Get directory path for uTVM host-driven runtime source files.

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_host_driven_dir = os.path.join(micro_dir, "..", "..", "..",
                                         "src", "runtime", "micro", "host_driven")
    return micro_host_driven_dir


def get_micro_device_dir():
    """Get directory path for parent directory of device-specific source files

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                    "src", "runtime", "micro", "device")
    return micro_device_dir


class MicroObjectFileBase:

  ENCODING_VERSION = 1

  # Names of the different parts of the tar file. Subclasses should override this and
  # define all `None` fields.
  TAR_FILE_NAMES = {
    'version': '{tar_file_root}/version',
    'elf_data': None,
    'metadata': '{tar_file_root}/metadata.json',
  }

  @classmethod
  def load(cls, file_path):
    with tarfile.open(file_path) as tar_f:
      version_f = tarfile.extractfile(cls.TAR_FILE_NAMES['version'])
      version = str(version_f.read(), 'utf-8').strip('\n')
      assert version == str(cls.ENCODING_VERSION), (
        f'Version does not match: want {cls.ENCODING_VERSION}, got {version}')

      elf_data = tarfile.extractfile(cls.TAR_FILE_NAMES['elf_data']).read()
      metadata = json.load(tarfile.extractfile(cls.TAR_FILE_NAMES['metadata']))

      return cls(os.path.basename(file_elf), pathse_data, metadata)

  def __init__(self, name, elf_data, metadata):
    self.name = name
    self.elf_data = elf_data
    self.metadata = metadata

  def save(self, file_path):
    tar_file_root = os.path.splitext(file_path)[0]
    with tarfile.open(file_path, 'w') as tar_f:
      def _add_file(name, data):
        ti = tarfile.TarInfo(
          name=self.TAR_FILE_NAMES[name].format(tar_file_root=tar_file_root))
        ti.type = tarfile.REGTYPE
        data_bytes = bytes(data)
        ti.size = len(data)
        tar_f.addfile(ti, io.BytesIO(data_bytes))

      _add_file('version', f'{cls.ENCODING_VERSION}\n')
      _add_file('elf_data', self.elf_data)
      _add_file('metadata', json.dumps(self.metadata))


tvm._ffi._init_api("tvm.micro", "tvm.micro.base")
