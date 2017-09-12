"""Utility for ROCm backend"""
from . import util

@tvm.register_func("tvm_callback_rocm_link")
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
    tmp_obj = tmp_dir.reloath("rocm_kernel.o")
    tmp_cobj = tmp_dir.reloath("rocm_kernel.co")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(obj_bin))
    cobj_bin = bytearray(open(tmp_cobj, "rb").read())
    return cobj_bin
