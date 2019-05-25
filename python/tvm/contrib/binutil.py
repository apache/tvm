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
import os
from . import util
from .._ffi.base import py_str
from .._ffi.libinfo import find_include_path
from ..api import register_func, convert


@register_func("tvm_callback_get_section_size")
def tvm_callback_get_section_size(binary_path, section_name):
    """Finds size of the section in the binary.
    Assumes `size` shell command exists (typically works only on Linux machines)

    Parameters
    ----------
    binary_path : str
        path of the binary file

    section_name : str
        name of section

    Return
    ------
    size : integer
        size of the section in bytes
    """
    if not os.path.isfile(binary_path):
        raise RuntimeError("no such file \"{}\"".format(binary_path))
    # We use the "-A" flag here to get the ".rodata" section's size, which is
    # not included by default.
    size_proc = subprocess.Popen(["size", "-A", binary_path], stdout=subprocess.PIPE)
    (size_output, _) = size_proc.communicate()
    if size_proc.returncode != 0:
        msg = "error in finding section size:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    size_output = size_output.decode("utf-8")
    section_size = 0
    # Skip the first two header lines in the `size` output.
    for line in size_output.split("\n")[2:]:
        tokens = list(filter(lambda s: len(s) != 0, line.split(" ")))
        if len(tokens) != 3:
            continue
        entry_name = tokens[0]
        entry_size = int(tokens[1])
        if entry_name.startswith("." + section_name):
            # The `.rodata` section should be the only section for which we
            # need to collect the size from *multiple* entries in the command
            # output.
            if section_size != 0 and not entry_name.startswith(".rodata"):
                raise RuntimeError("multiple entries in `size` output for section {}".format(section_name))
            section_size += entry_size
    return section_size


@register_func("tvm_callback_relocate_binary")
def tvm_callback_relocate_binary(binary_path, text_addr, rodata_addr, data_addr, bss_addr):
    """Relocates sections in the binary to new addresses

    Parameters
    ----------
    binary_path : str
        path of the binary file

    text_addr : str
        text section address

    rodata_addr : str
        rodata section address

    data_addr : str
        data section address

    bss_addr : str
        bss section address

    Return
    ------
    rel_bin : bytearray
        the relocated binary
    """
    tmp_dir = util.tempdir()
    rel_obj = tmp_dir.relpath("relocated.o")
    ld_script_contents = """
SECTIONS
{
  . = %s;
  . = ALIGN(8);
  .text :
  {
    *(.text)
    . = ALIGN(8);
    *(.text*)
  }
  . = %s;
  . = ALIGN(8);
  .rodata :
  {
    *(.rodata)
    . = ALIGN(8);
    *(.rodata*)
  }
  . = %s;
  . = ALIGN(8);
  .data :
  {
    *(.data)
    . = ALIGN(8);
    *(.data*)
  }
  . = %s;
  . = ALIGN(8);
  .bss :
  {
    *(.bss)
    . = ALIGN(8);
    *(.bss*)
  }
}
    """ % (text_addr, rodata_addr, data_addr, bss_addr)
    rel_ld_script = tmp_dir.relpath("relocated.lds")
    with open(rel_ld_script, "w") as f:
        f.write(ld_script_contents)
    ld_proc = subprocess.Popen(["ld", binary_path,
                                "-T", rel_ld_script,
                                "-o", rel_obj],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (out, _) = ld_proc.communicate()
    if ld_proc.returncode != 0:
        msg = "linking error using ld:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    with open(rel_obj, "rb") as f:
        rel_bin = bytearray(f.read())
    return rel_bin


@register_func("tvm_callback_read_binary_section")
def tvm_callback_read_binary_section(binary, section):
    """Returns the contents of the specified section in the binary file

    Parameters
    ----------
    binary : bytearray
        contents of the binary

    section : str
        type of section

    Return
    ------
    section_bin : bytearray
        contents of the read section
    """
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("temp.bin")
    tmp_section = tmp_dir.relpath("tmp_section.bin")
    with open(tmp_bin, "wb") as out_file:
        out_file.write(bytes(binary))
    objcopy_proc = subprocess.Popen(["objcopy", "--dump-section",
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
def tvm_callback_get_symbol_map(binary):
    """Obtains a map of symbols to addresses in the passed binary

    Parameters
    ----------
    binary : bytearray
        the object file

    Return
    ------
    map_str : str
        map of defined symbols to addresses, encoded as a series of
        alternating newline-separated keys and values
    """
    tmp_dir = util.tempdir()
    tmp_obj = tmp_dir.relpath("tmp_obj.bin")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(binary))
    nm_proc = subprocess.Popen(["nm", "-C", "--defined-only", tmp_obj],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (out, _) = nm_proc.communicate()
    if nm_proc.returncode != 0:
        msg = "error in using nm:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    out = out.decode("utf8").splitlines()
    map_str = ""
    for line in out:
        line = line.split()
        map_str += line[2] + "\n"
        map_str += line[0] + "\n"
    return map_str

