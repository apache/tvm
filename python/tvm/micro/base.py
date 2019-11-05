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

"""Base definitions for micro."""

from __future__ import absolute_import

import logging
import os
import sys
from collections import namedtuple
from enum import Enum
from pathlib import Path

import tvm
from tvm.contrib import util as _util
from tvm.contrib import cc as _cc
from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path
from tvm.contrib.binutil import run_cmd

class LibType(Enum):
    RUNTIME = 0
    OPERATOR = 1


class Session:
    """MicroTVM Device Session

    Parameters
    ----------
    device_type : str
        type of low-level device

    toolchain_prefix : str
        toolchain prefix to be used. For example, a prefix of
        "riscv64-unknown-elf-" means "riscv64-unknown-elf-gcc" is used as
        the compiler and "riscv64-unknown-elf-ld" is used as the linker,
        etc.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      dev_config = micro.device.stm32f746xx.default_config('127.0.0.1', 6666)
      with tvm.micro.Session(dev_config):
          c_mod.export_library(lib_obj_path, fcompile=tvm.micro.cross_compiler(dev_config))
          micro_mod = tvm.module.load(lib_obj_path, 'micro_dev')
    """

    # TODO(weberlo): remove required trailing dash in toolchain_prefix
    def __init__(self, config):
        self._check_system()
        #self._check_args(device_type, kwargs)

        #self.binutil = config['binutil']
        self.binutil = tvm.micro.device.get_binutil(config['binutil'])
        self.mem_layout = config['mem_layout']
        self.word_size = config['word_size']
        self.thumb_mode = config['thumb_mode']
        self.comms_method = config['comms_method']

        # First, find and compile runtime library.
        runtime_src_path = os.path.join(_get_micro_host_driven_dir(), 'utvm_runtime.c')
        tmp_dir = _util.tempdir()
        runtime_obj_path = tmp_dir.relpath('utvm_runtime.obj')
        self.binutil.create_lib(runtime_obj_path, runtime_src_path, LibType.RUNTIME)

        comms_method = config['comms_method']
        if comms_method == 'openocd':
            server_addr = config['server_addr']
            server_port = config['server_port']
        elif comms_method == 'host':
            server_addr = ''
            server_port = 0
        else:
            raise RuntimeError(f'unknown communication method: f{self.comms_method}')

        # todo remove use of base addrs everywhere
        base_addr = 0

        self.module = _CreateSession(
            comms_method,
            runtime_obj_path,
            self.binutil.toolchain_prefix(),
            self.mem_layout['text'].get('start', 0),
            self.mem_layout['text']['size'],
            self.mem_layout['rodata'].get('start', 0),
            self.mem_layout['rodata']['size'],
            self.mem_layout['data'].get('start', 0),
            self.mem_layout['data']['size'],
            self.mem_layout['bss'].get('start', 0),
            self.mem_layout['bss']['size'],
            self.mem_layout['args'].get('start', 0),
            self.mem_layout['args']['size'],
            self.mem_layout['heap'].get('start', 0),
            self.mem_layout['heap']['size'],
            self.mem_layout['workspace'].get('start', 0),
            self.mem_layout['workspace']['size'],
            self.mem_layout['stack'].get('start', 0),
            self.mem_layout['stack']['size'],
            self.word_size,
            self.thumb_mode,
            base_addr,
            server_addr,
            server_port)
        self._enter = self.module["enter"]
        self._exit = self.module["exit"]

    def create_micro_mod(self, c_mod):
        """Produces a micro module from a given module.

        Parameters
        ----------
        c_mod : tvm.module.Module
            module with "c" as its target backend

        Return
        ------
        micro_mod : tvm.module.Module
            micro module for the target device
        """
        print('[create_micro_mod]')
        temp_dir = _util.tempdir()
        lib_obj_path = temp_dir.relpath('dev_lib.obj')
        c_mod.export_library(
                lib_obj_path,
                fcompile=cross_compiler(self.binutil, LibType.OPERATOR))
        micro_mod = tvm.module.load(lib_obj_path)
        return micro_mod

    def _check_system(self):
        """Check if the user's system is supported by MicroTVM.

        Raises error if not supported.
        """
        if not sys.platform.startswith("linux"):
            raise RuntimeError("MicroTVM is currently only supported on Linux")
        # TODO(weberlo): Add 32-bit support.
        # It's primarily the compilation pipeline that isn't compatible.
        if sys.maxsize <= 2**32:
            raise RuntimeError("MicroTVM is currently only supported on 64-bit platforms")

    def _check_args(self, device_type, args):
        """Check if the given configuration is valid."""
        if device_type == "host":
            pass
        elif device_type == "openocd":
            assert "base_addr" in args
            assert "server_addr" in args
            assert "port" in args

    def __enter__(self):
        self._enter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._exit()


def cross_compiler(dev_binutil, lib_type):
    """Creates a cross compile function that wraps `create_micro_lib`.

    For use in `tvm.module.Module.export_library`.

    Parameters
    ----------
    lib_type: DFSDF

    Return
    ------
    func : Callable[[str, str, Optional[str]], None]
        cross compile function taking a destination path for the object file
        and a path for the input source file.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      fcompile = tvm.micro.cross_compiler(lib_type=LibType.OPERATOR)
      c_mod.export_library('dev_lib.obj', fcompile=fcompile)
    """
    if isinstance(dev_binutil, str):
        dev_binutil = tvm.micro.device.get_binutil(dev_binutil)

    def compile_func(obj_path, src_path, **kwargs):
        if isinstance(obj_path, list):
            obj_path = obj_path[0]
        if isinstance(src_path, list):
            src_path = src_path[0]
        dev_binutil.create_lib(obj_path, src_path, lib_type, kwargs.get('options', None))
    return _cc.cross_compiler(compile_func, output_format='obj')


def _get_micro_host_driven_dir():
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


def _get_micro_device_dir():
    """Get directory path for TODO

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_device_dir = os.path.join(micro_dir, "..", "..", "..",
                                    "src", "runtime", "micro", "device")
    return micro_device_dir


#def cross_compiler(dev_config, lib_type):
#    """Creates a cross compile function that wraps `create_micro_lib`.
#
#    For use in `tvm.module.Module.export_library`.
#
#    Parameters
#    ----------
#    toolchain_prefix : str
#        toolchain prefix to be used
#
#    include_dev_lib_header : Optional[bool]
#        whether to include the device library header containing definitions of
#        library functions.
#
#    Return
#    ------
#    func : Callable[[str, str, Optional[str]], None]
#        cross compile function taking a destination path for the object file
#        and a path for the input source file.
#
#    Example
#    --------
#    .. code-block:: python
#
#      c_mod = ...  # some module generated with "c" as the target
#      fcompile = tvm.micro.cross_compiler(toolchain_prefix="")
#      c_mod.export_library("dev_lib.obj", fcompile=fcompile)
#    """
#    def compile_func(obj_path, src_path, **kwargs):
#        if isinstance(obj_path, list):
#            obj_path = obj_path[0]
#        if isinstance(src_path, list):
#            src_path = src_path[0]
#        dev_config['binutil'].create_lib(obj_path, src_path, lib_type, kwargs.get('options', None))
#        #create_micro_lib(obj_path, src_path, toolchain_prefix,
#        #                 lib_type, kwargs.get("options", None))
#    return _cc.cross_compiler(compile_func, output_format='obj')


#def create_micro_lib(
#        obj_path, src_path, toolchain_prefix, lib_type, options=None):
#    """Compiles code into a binary for the target micro device.
#
#    Parameters
#    ----------
#    obj_path : Optional[str]
#        path to generated object file (defaults to same directory as `src_path`)
#
#    src_path : str
#        path to source file
#
#    toolchain_prefix : str
#        toolchain prefix to be used
#
#    include_dev_lib_header : bool
#        whether to include the device library header containing definitions of
#        library functions.
#    """
#    #import subprocess
#    import os
#    assert False
#
#    #def run_cmd(cmd):
#    #    proc = subprocess.Popen(
#    #            cmd,
#    #            stdout=subprocess.PIPE,
#    #            stderr=subprocess.STDOUT)
#    #    (out, _) = proc.communicate()
#    #    if proc.returncode != 0:
#    #        cmd_str = ' '.join(cmd)
#    #        msg = f"error while running command \"{' '.join(cmd)}\":\n"
#    #        msg += out.decode("utf-8")
#    #        raise RuntimeError(msg)
#
#    base_compile_cmd = [
#            f'{toolchain_prefix}gcc',
#            '-std=c11',
#            '-Wall',
#            '-Wextra',
#            '--pedantic',
#            '-c',
#            '-O0',
#            '-g',
#            '-nostartfiles',
#            '-nodefaultlibs',
#            '-nostdlib',
#            '-fdata-sections',
#            '-ffunction-sections',
#            '-DSTM32F746xx'
#            ]
#
#    if toolchain_prefix == 'arm-none-eabi-':
#        device_id = 'stm32f746'
#        base_compile_cmd += [
#            '-mcpu=cortex-m7',
#            '-mlittle-endian',
#            '-mfloat-abi=hard',
#            '-mfpu=fpv5-sp-d16',
#            '-mthumb',
#            '-gdwarf-5'
#            ]
#    elif toolchain_prefix == '':
#        device_id = 'host'
#        if sys.maxsize > 2**32 and sys.platform.startswith('linux'):
#            base_compile_cmd += ['-mcmodel=large']
#    else:
#        assert False
#
#    src_paths = []
#    include_paths = find_include_path() + [_get_micro_host_driven_dir()]
#    ld_script_path = None
#    tmp_dir = _util.tempdir()
#    if lib_type == LibType.RUNTIME:
#        import glob
#        dev_dir = _get_micro_device_dir() + '/' + device_id
#
#        dev_src_paths = glob.glob(f'{dev_dir}/*.[csS]')
#        # there needs to at least be a utvm_timer.c file
#        assert dev_src_paths
#
#        src_paths += dev_src_paths
#        # TODO: configure this
#        #include_paths += [dev_dir]
#        CMSIS_PATH = '/home/pratyush/Code/nucleo-interaction-from-scratch/stm32-cube'
#        include_paths += [f'{CMSIS_PATH}/Drivers/CMSIS/Include']
#        include_paths += [f'{CMSIS_PATH}/Drivers/CMSIS/Device/ST/STM32F7xx/Include']
#        include_paths += [f'{CMSIS_PATH}/Drivers/STM32F7xx_HAL_Driver/Inc']
#        include_paths += [f'{CMSIS_PATH}/Drivers/BSP/STM32F7xx_Nucleo_144']
#        include_paths += [f'{CMSIS_PATH}/Drivers/BSP/STM32746G-Discovery']
#    elif lib_type == LibType.OPERATOR:
#        # Create a temporary copy of the source, so we can inject the dev lib
#        # header without modifying the original.
#        temp_src_path = tmp_dir.relpath('temp.c')
#        with open(src_path, 'r') as f:
#            src_lines = f.read().splitlines()
#        src_lines.insert(0, '#include "utvm_device_dylib_redirect.c"')
#        with open(temp_src_path, 'w') as f:
#            f.write('\n'.join(src_lines))
#        src_path = temp_src_path
#
#        base_compile_cmd += ['-c']
#    else:
#        raise RuntimeError('unknown lib type')
#
#    src_paths += [src_path]
#
#    print(f'include paths: {include_paths}')
#    for path in include_paths:
#        base_compile_cmd += ['-I', path]
#
#    prereq_obj_paths = []
#    for src_path in src_paths:
#        curr_obj_path = tmp_dir.relpath(Path(src_path).with_suffix('.o').name)
#        i = 2
#        while curr_obj_path in prereq_obj_paths:
#            curr_obj_path = tmp_dir.relpath(Path(os.path.basename(src_path).split('.')[0] + str(i)).with_suffix('.o').name)
#            i += 1
#
#        prereq_obj_paths.append(curr_obj_path)
#        curr_compile_cmd = base_compile_cmd + [src_path, '-o', curr_obj_path]
#        run_cmd(curr_compile_cmd)
#
#    ld_cmd = [f'{toolchain_prefix}ld', '-relocatable']
#    ld_cmd += prereq_obj_paths
#    ld_cmd += ['-o', obj_path]
#    run_cmd(ld_cmd)
#    print(f'compiled obj {obj_path}')
#    #input('check it')
#
#    #if toolchain_prefix == '':
#    #    create_host_micro_lib(obj_path, src_path, toolchain_prefix, lib_type, options)
#    #elif toolchain_prefix == 'arm-none-eabi-':
#    #    create_arm_micro_lib(obj_path, src_path, toolchain_prefix, lib_type, options)
#
#
#def create_host_micro_lib(
#        obj_path, src_path, toolchain_prefix, lib_type, options):
#    # uTVM object files cannot have an ".o" suffix, because it triggers the
#    # code path for creating shared objects in `tvm.module.load`.  So we replace
#    # ".o" suffixes with ".obj".
#    if obj_path.endswith(".o"):
#        logging.warning(
#            "\".o\" suffix in \"%s\" has been replaced with \".obj\"", obj_path)
#        obj_path = str(Path(obj_path).with_suffix("obj"))
#
#    options = ["-I" + path for path in find_include_path()]
#    options += ["-I{}".format(_get_micro_host_driven_dir())]
#    options += ["-fno-stack-protector"]
#    # TODO(weberlo): Don't rely on the toolchain prefix to identify if this is the host
#    # device.
#    if toolchain_prefix == "" and sys.maxsize > 2**32 and sys.platform.startswith("linux"):
#        # Only add this option if the host is a 64-bit Linux.
#        options += ["-mcmodel=large"]
#    compile_cmd = "{}gcc".format(toolchain_prefix)
#
#    if lib_type == LibType.OPERATOR:
#        # Create a temporary copy of the source, so we can inject the dev lib
#        # header without modifying the original.
#        tmp_dir = _util.tempdir()
#        temp_src_path = tmp_dir.relpath("temp.c")
#        with open(src_path, "r") as f:
#            src_lines = f.read().splitlines()
#        src_lines.insert(0, "#include \"utvm_device_dylib_redirect.c\"")
#        with open(temp_src_path, "w") as f:
#            f.write("\n".join(src_lines))
#        src_path = temp_src_path
#
#    _cc.create_shared(obj_path, src_path, options, compile_cmd)


#def create_arm_micro_lib(
#        obj_path, src_path, toolchain_prefix, lib_type, options):
#    import subprocess
#    import os
#    import copy
#    from shutil import copyfile
#
#    from tvm._ffi.libinfo import find_include_path
#    from tvm.contrib import binutil
#
#    def run_cmd(cmd):
#        proc = subprocess.Popen(
#                cmd,
#                stdout=subprocess.PIPE,
#                stderr=subprocess.STDOUT)
#        (out, _) = proc.communicate()
#        if proc.returncode != 0:
#            msg = "Compilation error:\n"
#            msg += out.decode("utf-8")
#            raise RuntimeError(msg)
#
#    base_compile_cmd = [
#            'arm-none-eabi-gcc',
#            '-std=c11',
#            '-Wall',
#            '-Wextra',
#            '--pedantic',
#            '-mcpu=cortex-m7',
#            '-mlittle-endian',
#            '-mfloat-abi=hard',
#            '-mfpu=fpv5-sp-d16',
#            '-mthumb',
#            '-c',
#            '-O0',
#            '-g',
#            '-gdwarf-5',
#            '-nostartfiles',
#            '-nodefaultlibs',
#            '-nostdlib',
#            '-fdata-sections',
#            '-ffunction-sections']
#
#    src_paths = []
#    ld_script_path = None
#    tmp_dir = _util.tempdir()
#    if lib_type == LibType.RUNTIME:
#        import glob
#        DEVICE_ID = 'stm32f746'
#        dev_dir = _get_micro_device_dir() + '/' + DEVICE_ID
#
#        dev_src_paths = glob.glob(f'{dev_dir}/*.[csS]')
#        assert dev_src_paths
#        src_paths += dev_src_paths
#    elif lib_type == LibType.OPERATOR:
#        # Create a temporary copy of the source, so we can inject the dev lib
#        # header without modifying the original.
#        temp_src_path = tmp_dir.relpath("temp.c")
#        with open(src_path, "r") as f:
#            src_lines = f.read().splitlines()
#        src_lines.insert(0, "#include \"utvm_device_dylib_redirect.c\"")
#        with open(temp_src_path, "w") as f:
#            f.write("\n".join(src_lines))
#        src_path = temp_src_path
#
#        base_compile_cmd += ['-c']
#    else:
#        raise RuntimeError('unknown lib type')
#
#    src_paths += [src_path]
#
#    include_paths = find_include_path() + [_get_micro_host_driven_dir()]
#    for path in include_paths:
#        base_compile_cmd += ['-I', path]
#
#    prereq_obj_paths = []
#    for src_path in src_paths:
#        curr_obj_path = tmp_dir.relpath(Path(src_path).with_suffix('o').name)
#        prereq_obj_paths.append(curr_obj_path)
#        curr_compile_cmd = base_compile_cmd + [src_path, '-o', curr_obj_path]
#        run_cmd(curr_compile_cmd)
#
#    # TODO(weberlo): adding '-fPIC' here causes the pc-relative data pools to
#    # not be updated when the obj is reloced. why?
#    ld_cmd = ['arm-none-eabi-ld', '-relocatable']
#    ld_cmd += prereq_obj_paths
#    ld_cmd += ['-o', obj_path]
#    run_cmd(ld_cmd)
#    print(f'compiled obj {obj_path}')


_init_api("tvm.micro", "tvm.micro.base")
