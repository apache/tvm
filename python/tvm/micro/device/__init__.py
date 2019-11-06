import os
import sys
from enum import Enum
from pathlib import Path

from tvm.contrib import util as _util
from tvm.contrib.binutil import run_cmd
from tvm._ffi.libinfo import find_include_path
from tvm.micro import LibType

class MicroBinutil:
    def __init__(self, toolchain_prefix):
        self._toolchain_prefix = toolchain_prefix

    def create_lib(self, obj_path, src_path, lib_type, options=None):
        """Compiles code into a binary for the target micro device.

        Parameters
        ----------
        obj_path : Optional[str]
            path to generated object file (defaults to same directory as `src_path`)

        src_path : str
            path to source file

        toolchain_prefix : str
            toolchain prefix to be used

        include_dev_lib_header : bool
            whether to include the device library header containing definitions of
            library functions.
        """
        print('[MicroBinutil.create_lib]')
        print('  EXTENDED OPTIONS')
        print(f'    {obj_path}')
        print(f'    {src_path}')
        print(f'    {lib_type}')
        print(f'    {options}')
        base_compile_cmd = [
                f'{self.toolchain_prefix()}gcc',
                '-std=c11',
                '-Wall',
                '-Wextra',
                '--pedantic',
                '-c',
                '-O0',
                '-g',
                '-nostartfiles',
                '-nodefaultlibs',
                '-nostdlib',
                '-fdata-sections',
                '-ffunction-sections',
                ]
        if options is not None:
            base_compile_cmd += options

        src_paths = []
        include_paths = find_include_path() + [_get_micro_host_driven_dir()]
        ld_script_path = None
        tmp_dir = _util.tempdir()
        if lib_type == LibType.RUNTIME:
            import glob
            dev_dir = _get_micro_device_dir() + '/' + self.device_id()

            print(dev_dir)
            dev_src_paths = glob.glob(f'{dev_dir}/*.[csS]')
            print(dev_src_paths)
            # there needs to at least be a utvm_timer.c file
            assert dev_src_paths

            src_paths += dev_src_paths
        elif lib_type == LibType.OPERATOR:
            # Create a temporary copy of the source, so we can inject the dev lib
            # header without modifying the original.
            temp_src_path = tmp_dir.relpath('temp.c')
            with open(src_path, 'r') as f:
                src_lines = f.read().splitlines()
            src_lines.insert(0, '#include "utvm_device_dylib_redirect.c"')
            with open(temp_src_path, 'w') as f:
                f.write('\n'.join(src_lines))
            src_path = temp_src_path

            base_compile_cmd += ['-c']
        else:
            raise RuntimeError('unknown lib type')

        src_paths += [src_path]

        print(f'include paths: {include_paths}')
        for path in include_paths:
            base_compile_cmd += ['-I', path]

        prereq_obj_paths = []
        for src_path in src_paths:
            curr_obj_path = self._get_unique_obj_name(src_path, prereq_obj_paths, tmp_dir)
            prereq_obj_paths.append(curr_obj_path)
            curr_compile_cmd = base_compile_cmd + [src_path, '-o', curr_obj_path]
            run_cmd(curr_compile_cmd)

        ld_cmd = [f'{self.toolchain_prefix()}ld', '-relocatable']
        ld_cmd += prereq_obj_paths
        ld_cmd += ['-o', obj_path]
        run_cmd(ld_cmd)

    def _get_unique_obj_name(self, src_path, obj_paths, tmp_dir):
        res = tmp_dir.relpath(Path(src_path).with_suffix('.o').name)
        i = 2
        # if the name collides, try increasing numeric suffixes until the name doesn't collide
        while res in obj_paths:
            res = tmp_dir.relpath(Path(os.path.basename(src_path).split('.')[0] + str(i)).with_suffix('.o').name)
            i += 1
        return res

    def device_id(self):
        raise RuntimeError('no device ID for abstract MicroBinutil')

    def toolchain_prefix(self):
        return self._toolchain_prefix


from . import host
from . import arm
from . import riscv_spike

def get_binutil(name):
    if name == 'host':
        return host.HostBinutil()
    elif name == 'stm32f746xx':
        return arm.stm32f746xx.Stm32F746XXBinutil()
    else:
        assert False


def _get_micro_host_driven_dir():
    """Get directory path for uTVM host-driven runtime source files.

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_host_driven_dir = os.path.join(micro_dir, '..', '..', '..', '..',
                                         'src', 'runtime', 'micro', 'host_driven')
    return micro_host_driven_dir


def _get_micro_device_dir():
    """Get directory path for TODO

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_device_dir = os.path.join(micro_dir, "..", "..", "..", "..",
                                    "src", "runtime", "micro", "device")
    return micro_device_dir

