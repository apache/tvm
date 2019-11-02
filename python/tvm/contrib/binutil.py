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

"""Utilities for binary file manipulation"""
import os
import subprocess
from . import util
from .._ffi.base import py_str
from ..api import register_func


def run_cmd(cmd):
    proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        cmd_str = ' '.join(cmd)
        error_msg = out.decode("utf-8")
        msg = f"error while running command \"{cmd_str}\":\n{error_msg}"
        raise RuntimeError(msg)


@register_func("tvm_callback_get_section_size")
def tvm_callback_get_section_size(binary_path, section_name, toolchain_prefix):
    """Finds size of the section in the binary.
    Assumes `size` shell command exists (typically works only on Linux machines)

    Parameters
    ----------
    binary_path : str
        path of the binary file

    section_name : str
        name of section

    toolchain_prefix : str
        prefix for binary names in target compiler toolchain

    Returns
    -------
    size : integer
        size of the section in bytes
    """
    if not os.path.isfile(binary_path):
        raise RuntimeError("no such file \"{}\"".format(binary_path))
    # We use the "-A" flag here to get the ".rodata" section's size, which is
    # not included by default.
    size_proc = subprocess.Popen(
        ["{}size".format(toolchain_prefix), "-A", binary_path], stdout=subprocess.PIPE)
    (size_output, _) = size_proc.communicate()
    size_output = size_output.decode("utf-8")
    if size_proc.returncode != 0:
        msg = "error in finding section size:\n"
        msg += py_str(size_output)
        raise RuntimeError(msg)

    # TODO(weberlo): Refactor this method and `*relocate_binary` so they are
    # both aware of [".bss", ".sbss", ".sdata"] being relocated to ".bss".
    section_mapping = {
        ".text": [".text", '.isr_vector'],
        ".rodata": [".rodata"],
        ".data": [".data", ".sdata"],
        ".bss": [".bss", ".sbss"],
    }
    sections_to_sum = section_mapping["." + section_name]
    section_size = 0
    # Skip the first two header lines in the `size` output.
    for line in size_output.split("\n")[2:]:
        tokens = list(filter(lambda s: len(s) != 0, line.split(" ")))
        if len(tokens) != 3:
            continue
        entry_name = tokens[0]
        entry_size = int(tokens[1])
        for section_name in sections_to_sum:
            if entry_name.startswith(section_name):
                section_size += entry_size
                break

    # NOTE: For some reason, the size of the BSS section on the RISC-V
    # GCC is sometimes reported to be smaller than it is, so we need to adjust
    # for this.
    if "riscv" in toolchain_prefix and section_name == 'bss':
        # TODO(weberlo): Figure out why 32 is the minimum constant that works.
        #
        # The current hypothesis is that the last symbols in the ".bss" and
        # ".sbss" sections may have size zero, since the symbols in these
        # sections are uninitialized and there's no address that follows that
        # would enforce a particular size.
        #
        # If this is the case, then 32 just happens to be a safe amount of
        # padding for most cases, but symbols can be arbitrarily large, so this
        # isn't bulletproof.
        return section_size + 32
    return section_size


@register_func("tvm_callback_relocate_binary")
def tvm_callback_relocate_binary(
        binary_path, text_addr, rodata_addr, data_addr, bss_addr, toolchain_prefix):
    """Relocates sections in the binary to new addresses

    Parameters
    ----------
    binary_path : str
        path of the binary file

    text_addr : str
        text section absolute address

    rodata_addr : str
        rodata section absolute address

    data_addr : str
        data section absolute address

    bss_addr : str
        bss section absolute address

    toolchain_prefix : str
        prefix for binary names in target compiler toolchain

    Returns
    -------
    rel_bin : bytearray
        the relocated binary
    """
    global TEMPDIR_REFS
    tmp_dir = util.tempdir()
    TEMPDIR_REFS.append(tmp_dir)
    rel_obj_path = tmp_dir.relpath("relocated.o")
    #rel_obj_path = '/home/pratyush/Code/nucleo-interaction-from-scratch/src/main_relocated.o'
    with open('/home/pratyush/Code/nucleo-interaction-from-scratch/.gdbinit', 'r') as f:
        gdbinit_contents = f.read().split('\n')
    new_contents = []
    for line in gdbinit_contents:
        new_contents.append(line)
        if line.startswith('target'):
            new_contents.append(f'add-symbol-file {rel_obj_path}')
    with open('/home/pratyush/Code/nucleo-interaction-from-scratch/.gdbinit', 'w') as f:
        f.write('\n'.join(new_contents))

    #ld_script_contents = ""
    ## TODO(weberlo): There should be a better way to configure this for different archs.
    #if "riscv" in toolchain_prefix:
    #    ld_script_contents += "OUTPUT_ARCH( \"riscv\" )\n\n"

    # TODO: this a fukn hack
    #input(f'binary path: {binary_path}')
    if 'utvm_runtime' in binary_path:
        ld_script_contents = """
ENTRY(UTVMInit)

/* Highest address of the user mode stack */
_estack = 0x20050000;   /* end of RAM */

/*
 * Memory layout (for reference)
 *   RAM   (xrw) : START = 0x20000000, LENGTH = 320K
 *   FLASH (rx)  : START = 0x8000000,  LENGTH = 1024K
 */

/* Define output sections */
SECTIONS
{
  . = %s;
  . = ALIGN(4);
  .text :
  {
    _stext = .;
    KEEP(*(.isr_vector))
    KEEP(*(.text))
    KEEP(*(.text*))
    . = ALIGN(4);
  }

  . = %s;
  . = ALIGN(4);
  .rodata :
  {
    . = ALIGN(4);
    KEEP(*(.rodata))
    KEEP(*(.rodata*))
    . = ALIGN(4);
  }

  . = %s;
  . = ALIGN(4);
  .data :
  {
    . = ALIGN(4);
    KEEP(*(.data))
    KEEP(*(.data*))
    . = ALIGN(4);
  }

  . = %s;
  . = ALIGN(4);
  .bss :
  {
    . = ALIGN(4);
    KEEP(*(.bss))
    KEEP(*(.bss*))
    . = ALIGN(4);
  }
}
        """ % (text_addr, rodata_addr, data_addr, bss_addr)
    else:
        ld_script_contents = """
/*
 * Memory layout (for reference)
 *   RAM   (xrw) : START = 0x20000000, LENGTH = 320K
 *   FLASH (rx)  : START = 0x8000000,  LENGTH = 1024K
 */

/* Define output sections */
SECTIONS
{
  . = %s;
  . = ALIGN(4);
  .text :
  {
    . = ALIGN(4);
    KEEP(*(.text))
    KEEP(*(.text*))
    . = ALIGN(4);
  }

  . = %s;
  . = ALIGN(4);
  .rodata :
  {
    . = ALIGN(4);
    KEEP(*(.rodata))
    KEEP(*(.rodata*))
    . = ALIGN(4);
  }

  . = %s;
  . = ALIGN(4);
  .data :
  {
    . = ALIGN(4);
    KEEP(*(.data))
    KEEP(*(.data*))
    . = ALIGN(4);
  }

  . = %s;
  . = ALIGN(4);
  .bss :
  {
    . = ALIGN(4);
    KEEP(*(.bss))
    KEEP(*(.bss*))
    . = ALIGN(4);
  }
}
        """ % (text_addr, rodata_addr, data_addr, bss_addr)
        print(f'relocing lib {binary_path}')
        print(f'  text_addr: {text_addr}')
        print(f'  rodata_addr: {rodata_addr}')
        print(f'  data_addr: {data_addr}')
        print(f'  bss_addr: {bss_addr}')

    # TODO(weberlo): Generate the script in a more procedural manner.

    # TODO: use the stm32-cube linker script as reference. set UTVMInit as
    # entry point. add KEEP to sections to prevent garbage collection.

#"""
#SECTIONS
#{
#  . = %s;
#  . = ALIGN(8);
#  .text :
#  {
#    *(.text)
#    . = ALIGN(8);
#    *(.text*)
#  }
#  . = %s;
#  . = ALIGN(8);
#  .rodata :
#  {
#    *(.rodata)
#    . = ALIGN(8);
#    *(.rodata*)
#  }
#  . = %s;
#  . = ALIGN(8);
#  .data :
#  {
#    *(.data)
#    . = ALIGN(8);
#    *(.data*)
#    . = ALIGN(8);
#    *(.sdata)
#  }
#  . = %s;
#  . = ALIGN(8);
#  .bss :
#  {
#    *(.bss)
#    . = ALIGN(8);
#    *(.bss*)
#    . = ALIGN(8);
#    *(.sbss)
#  }
#}
#    """ % (text_addr, rodata_addr, data_addr, bss_addr)
    rel_ld_script_path = tmp_dir.relpath("relocated.lds")
    with open(rel_ld_script_path, "w") as f:
        f.write(ld_script_contents)
    ld_proc = subprocess.Popen(["{}ld".format(toolchain_prefix), binary_path,
                                "-T", rel_ld_script_path,
                                "-o", rel_obj_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (out, _) = ld_proc.communicate()
    if ld_proc.returncode != 0:
        msg = "linking error using ld:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    print(f'relocated obj path is {rel_obj_path}')
    with open(rel_obj_path, "rb") as f:
        rel_bin = bytearray(f.read())
    return rel_bin


@register_func("tvm_callback_read_binary_section")
def tvm_callback_read_binary_section(binary, section, toolchain_prefix):
    """Returns the contents of the specified section in the binary byte array

    Parameters
    ----------
    binary : bytearray
        contents of the binary

    section : str
        type of section

    toolchain_prefix : str
        prefix for binary names in target compiler toolchain

    Returns
    -------
    section_bin : bytearray
        contents of the read section
    """
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("temp.bin")
    tmp_section = tmp_dir.relpath("tmp_section.bin")
    with open(tmp_bin, "wb") as out_file:
        out_file.write(bytes(binary))
    objcopy_proc = subprocess.Popen(["{}objcopy".format(toolchain_prefix), "--dump-section",
                                     ".{}={}".format(section, tmp_section),
                                     tmp_bin],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
    (out, _) = objcopy_proc.communicate()
    if objcopy_proc.returncode != 0:
        msg = "error in using objcopy:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    if os.path.isfile(tmp_section):
        # Get section content if it exists.
        with open(tmp_section, "rb") as f:
            section_bin = bytearray(f.read())
    else:
        # Return empty bytearray if the section does not exist.
        section_bin = bytearray("", "utf-8")
    return section_bin


@register_func("tvm_callback_get_symbol_map")
def tvm_callback_get_symbol_map(binary, toolchain_prefix):
    """Obtains a map of symbols to addresses in the passed binary

    Parameters
    ----------
    binary : bytearray
        contents of the binary

    toolchain_prefix : str
        prefix for binary names in target compiler toolchain

    Returns
    -------
    map_str : str
        map of defined symbols to addresses, encoded as a series of
        alternating newline-separated keys and values
    """
    tmp_dir = util.tempdir()
    tmp_obj = tmp_dir.relpath("tmp_obj.bin")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(binary))
    nm_proc = subprocess.Popen(["{}nm".format(toolchain_prefix), "-C", "--defined-only", tmp_obj],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (nm_output, _) = nm_proc.communicate()
    if nm_proc.returncode != 0:
        msg = "error in using nm:\n"
        msg += py_str(nm_output)
        raise RuntimeError(msg)
    nm_output = nm_output.decode("utf8").splitlines()
    map_str = ""
    for line in nm_output:
        line = line.split()
        map_str += line[2] + "\n"
        map_str += line[0] + "\n"
    return map_str
