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
import tvm._ffi
from . import util

# TODO does this file still belong in `contrib`. is it too µTVM-specific?

# TODO shouldn't need so many `ALIGN` directives
RELOCATION_LD_SCRIPT_TEMPLATE = """
/* linker symbol for use in UTVMInit */
_utvm_stack_pointer_init = 0x{stack_pointer_init:x};

SECTIONS
{{
  . = 0x{text_start:x};
  . = ALIGN({word_size});
  .text :
  {{
    . = ALIGN({word_size});
    KEEP(*(.text))
    KEEP(*(.text*))
    . = ALIGN({word_size});
  }}

  . = 0x{rodata_start:x};
  . = ALIGN({word_size});
  .rodata :
  {{
    . = ALIGN({word_size});
    KEEP(*(.rodata))
    KEEP(*(.rodata*))
    . = ALIGN({word_size});
  }}

  . = 0x{data_start:x};
  . = ALIGN({word_size});
  .data :
  {{
    . = ALIGN({word_size});
    KEEP(*(.data))
    KEEP(*(.data*))
    . = ALIGN({word_size});
  }}

  . = 0x{bss_start:x};
  . = ALIGN({word_size});
  .bss :
  {{
    . = ALIGN({word_size});
    KEEP(*(.bss))
    KEEP(*(.bss*))
    . = ALIGN({word_size});
  }}
}}
"""

def run_cmd(cmd):
    """Runs `cmd` in a subprocess and awaits its completion.

    Parameters
    ----------
    cmd : List[str]
        list of command-line arguments

    Returns
    -------
    output : str
        resulting stdout capture from the subprocess
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (output, _) = proc.communicate()
    output = output.decode("utf-8")
    if proc.returncode != 0:
        cmd_str = " ".join(cmd)
        msg = f"error while running command \"{cmd_str}\":\n{output}"
        raise RuntimeError(msg)
    return output


@tvm._ffi.register_func("tvm_callback_get_section_size")
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
        raise RuntimeError('no such file "{}"'.format(binary_path))
    # We use the "-A" flag here to get the ".rodata" section's size, which is
    # not included by default.
    size_output = run_cmd(["{}size".format(toolchain_prefix), "-A", binary_path])

    # TODO(weberlo): Refactor this method and `*relocate_binary` so they are
    # both aware of [".bss", ".sbss", ".sdata"] being relocated to ".bss".
    section_mapping = {
        ".text": [".text"],
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
        for section in sections_to_sum:
            if entry_name.startswith(section):
                section_size += entry_size
                break

    # NOTE: For some reason, the size of the BSS section on the RISC-V
    # GCC is sometimes reported to be smaller than it is, so we need to adjust
    # for this.
    if "riscv" in toolchain_prefix and section_name == "bss":
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

    # NOTE: in the past, section_size has been wrong on x86. it may be
    # inconsistent. TODO: maybe stop relying on `*size` to give us the size and
    # instead read the section with `*objcopy` and count the bytes.
    return section_size


@tvm._ffi.register_func("tvm_callback_relocate_binary")
def tvm_callback_relocate_binary(
        binary_path,
        word_size,
        text_start,
        rodata_start,
        data_start,
        bss_start,
        stack_end,
        toolchain_prefix):
    """Relocates sections in the binary to new addresses

    Parameters
    ----------
    binary_path : str
        path of the binary file

    word_size : int
        word size on the target machine

    text_start : int
        text section address

    rodata_start : int
        rodata section address

    data_start : int
        data section address

    bss_start : int
        bss section address

    stack_end : int
        stack section end address

    toolchain_prefix : str
        prefix for binary names in target compiler toolchain

    Returns
    -------
    rel_bin : bytearray
        the relocated binary
    """
    assert text_start < rodata_start < data_start < bss_start < stack_end
    stack_pointer_init = stack_end - word_size
    ld_script_contents = ""
    # TODO(weberlo): There should be a better way to configure this for different archs.
    # TODO is this line even necessary?
    if "riscv" in toolchain_prefix:
        ld_script_contents += 'OUTPUT_ARCH( "riscv" )\n\n'
    ld_script_contents += RELOCATION_LD_SCRIPT_TEMPLATE.format(
        word_size=word_size,
        text_start=text_start,
        rodata_start=rodata_start,
        data_start=data_start,
        bss_start=bss_start,
        stack_pointer_init=stack_pointer_init)

    tmp_dir = util.tempdir()
    rel_obj_path = tmp_dir.relpath("relocated.obj")
    rel_ld_script_path = tmp_dir.relpath("relocate.lds")
    with open(rel_ld_script_path, "w") as f:
        f.write(ld_script_contents)
    run_cmd([
        "{}ld".format(toolchain_prefix),
        binary_path,
        "-T", rel_ld_script_path,
        "-o", rel_obj_path])

    with open(rel_obj_path, "rb") as f:
        rel_bin = bytearray(f.read())

    gdb_init_dir = os.environ.get("MICRO_GDB_INIT_DIR")
    if gdb_init_dir is not None:
        gdb_init_path = f"{gdb_init_dir}/.gdbinit"
        with open(gdb_init_path, "r") as f:
            gdbinit_contents = f.read().split("\n")
        new_contents = []
        for line in gdbinit_contents:
            new_contents.append(line)
            if line.startswith("target"):
                new_contents.append(f"add-symbol-file {rel_obj_path}")
        with open(gdb_init_path, "w") as f:
            f.write("\n".join(new_contents))

    return rel_bin


@tvm._ffi.register_func("tvm_callback_read_binary_section")
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
    run_cmd([
        "{}objcopy".format(toolchain_prefix),
        "--dump-section",
        ".{}={}".format(section, tmp_section),
        tmp_bin])
    if os.path.isfile(tmp_section):
        # Get section content if it exists.
        with open(tmp_section, "rb") as f:
            section_bin = bytearray(f.read())
    else:
        # Return empty bytearray if the section does not exist.
        section_bin = bytearray("", "utf-8")
    return section_bin


@tvm._ffi.register_func("tvm_callback_get_symbol_map")
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
    nm_output = run_cmd([
        "{}nm".format(toolchain_prefix),
        "-C",
        "--defined-only",
        tmp_obj])
    nm_output = nm_output.splitlines()
    map_str = ""
    for line in nm_output:
        line = line.split()
        map_str += line[2] + "\n"
        map_str += line[0] + "\n"
    return map_str
