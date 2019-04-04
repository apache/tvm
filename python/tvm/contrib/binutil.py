"""Utilities for binary file manipulation"""
import subprocess
from os.path import join, exists
from . import util
from .._ffi.base import py_str
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
    section_map = {"text": "1", "data": "2", "bss": "3"}
    p1 = subprocess.Popen(["size", binary_path], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["awk", "{print $" + section_map[section] + "}"],
                           stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["tail", "-1"], stdin=p2.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    (out, _) = p3.communicate()
    if p3.returncode != 0:
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
    p1 = subprocess.Popen(["ld", binary_path,
                           "-Ttext", text,
                           "-Tdata", data,
                           "-Tbss", bss,
                           "-o", rel_obj],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    (out, _) = p1.communicate()
    if p1.returncode != 0:
        msg = "Linking error using ld:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    rel_bin = bytearray(open(rel_obj, "rb").read())
    return rel_bin


@register_func("tvm_callback_read_binary_section")
def tvm_callback_read_binary_section(binary_path, section):
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
    p1 = subprocess.Popen(["objcopy", "--dump-section",
                           "." + section + "=" + tmp_section,
                           binary_path],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    (out, _) = p1.communicate()
    if p1.returncode != 0:
        msg = "Error in using objcopy:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    try:
        # get section content if it exits
        section_bin = bytearray(open(tmp_section, "rb").read())
    except IOError:
        # return empty bytearray if the section does not exist
        section_bin = bytearray("")
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
    p1 = subprocess.Popen(["nm", "-C", "--defined-only", tmp_obj],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    (out, _) = p1.communicate()
    if p1.returncode != 0:
        msg = "Error in using nm:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    out = out.splitlines()
    map_str = ""
    for line in out:
        line = line.split()
        map_str += line[2] + "\n"
        map_str += line[0] + "\n"
    return map_str


@register_func("tvm_callback_compile_micro")
def tvm_callback_compile_binary(code_path="reasonable_default", cc="gcc"):
    """Compiles code into a binary

    Parameters
    ----------
    code_path : str
        path to code file

    cc : str
        compiler to be used

    Return
    ------
    binary_path : bytearray
        compiled binary filename
    """
    tmp_dir = util.tempdir()
    tmp_obj = tmp_dir.relpath("tmp_obj.bin")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(binary))
    p1 = subprocess.Popen([cc, "-c", "", tmp_obj],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    (out, _) = p1.communicate()
    if p1.returncode != 0:
        msg = "Error in using nm:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    out = out.splitlines()
    map_str = ""
    for line in out:
        line = line.split()
        map_str += line[2] + "\n"
        map_str += line[0] + "\n"
    return map_str
