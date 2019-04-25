"""Utilities for binary file manipulation"""
import os
import subprocess
import os
from . import util
from .._ffi.base import py_str
from .._ffi.libinfo import find_include_path
from ..api import register_func, convert


@register_func("tvm_callback_get_section_size")
def tvm_callback_get_section_size(binary_path, section):
    """Finds size of the section in the binary.
    Assumes "size" shell command exists (typically works only on Linux machines)

    Parameters
    ----------
    binary_path : str
        path of the binary file

    section : str
        type of section

    Return
    ------
    size : integer
        size of the section in bytes
    """
    if not os.path.isfile(binary_path):
        raise RuntimeError("No such file {}".format(binary_path))
    section_map = {"text": "1", "data": "2", "bss": "3"}
    size_proc = subprocess.Popen(["size", binary_path], stdout=subprocess.PIPE)
    awk_proc = subprocess.Popen(["awk", "{print $" + section_map[section] + "}"],
                                stdin=size_proc.stdout, stdout=subprocess.PIPE)
    tail_proc = subprocess.Popen(["tail", "-1"], stdin=awk_proc.stdout, stdout=subprocess.PIPE)
    size_proc.stdout.close()
    awk_proc.stdout.close()
    (out, _) = tail_proc.communicate()
    if tail_proc.returncode != 0:
        msg = "Error in finding section size:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    return int(out)


@register_func("tvm_callback_relocate_binary")
def tvm_callback_relocate_binary(binary_path, text, data, bss):
    """Relocates sections in the binary to new addresses

    Parameters
    ----------
    binary_path : str
        path of the binary file

    text : str
        text section address

    data : str
        data section address

    bss : str
        bss section address

    Return
    ------
    rel_bin : bytearray
        the relocated binary
    """
    tmp_dir = util.tempdir()
    rel_obj = tmp_dir.relpath("relocated.o")
    ld_proc = subprocess.Popen(["ld", binary_path,
                                "-Ttext", text,
                                "-Tdata", data,
                                "-Tbss", bss,
                                "-o", rel_obj],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (out, _) = ld_proc.communicate()
    if ld_proc.returncode != 0:
        msg = "Linking error using ld:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    rel_bin = bytearray(open(rel_obj, "rb").read())
    return rel_bin


@register_func("tvm_callback_read_binary_section")
def tvm_callback_read_binary_section(binary, section):
    """Returns the contents of the specified section in the binary file

    Parameters
    ----------
    binary_path : str
        path of the binary file

    section : str
        type of section

    Return
    ------
    section_bin : bytearray
        contents of the read section
    """
    tmp_dir = util.tempdir()
    tmp_section = tmp_dir.relpath("tmp_section.bin")
    with open(tmp_bin, "wb") as out_file:
        out_file.write(bytes(binary))
    objcopy_proc = subprocess.Popen(["objcopy", "--dump-section",
                                     "." + section + "=" + tmp_section,
                                     tmp_bin],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
    (out, _) = objcopy_proc.communicate()
    if objcopy_proc.returncode != 0:
        msg = "Error in using objcopy:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    if os.path.isfile(tmp_section):
        # get section content if it exists
        with open(tmp_section, "rb") as f:
            section_bin = bytearray(f.read())
    else:
        # return empty bytearray if the section does not exist
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
    symbol_map : dictionary
        map of defined symbols to addresses
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
        msg = "Error in using nm:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    out = out.decode("utf8").splitlines()
    map_str = ""
    for line in out:
        line = line.split()
        map_str += line[2] + "\n"
        map_str += line[0] + "\n"
    return map_str

