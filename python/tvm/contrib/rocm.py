"""Utility for ROCm backend"""
import sys
import subprocess
from . import util
from ..api import register_func

def rocm_link(in_file, out_file):
    """Link relocatable ELF object to shared ELF object using lld

    Parameters
    ----------
    in_file : str
        Input file name (relocatable ELF object file)

    out_file : str
        Output file name (shared ELF object file)
    """
    args = "ld.lld -shared " + in_file + " -o " + out_file
    print args
    proc = subprocess.Popen(
        args, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()

    if proc.returncode != 0:
        sys.stderr.write("Linking error using ld.lld:\n")
        sys.stderr.write(str(out))
        sys.stderr.flush()

@register_func("tvm_callback_rocm_link")
def callback_rocm_link(obj_bin):
    """Links object file generated from LLVM to HSA Code Object

    Parameters
    ----------
    obj_bin : bytearray
        The object file

    Return
    ------
    cobj_bin : bytearray
        The HSA Code Object
    """
    tmp_dir = util.tempdir()
    tmp_obj = tmp_dir.relpath("rocm_kernel.o")
    tmp_cobj = tmp_dir.relpath("rocm_kernel.co")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(obj_bin))
    rocm_link(tmp_obj, tmp_cobj)
    cobj_bin = bytearray(open(tmp_cobj, "rb").read())
    return cobj_bin
