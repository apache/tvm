# tvm.contrib.rocm
"""Utility to convert object file to HSA Code Object
The object file received from LLVM PassManager is relocatable ELF object.
The routine in this file uses lld to convert it to shared ELF object.
"""
from . import util

@tvm.register_func("tvm_callback_rocm_link")
def callback_rocm_link(obj_bin):
    """Links object file generated from LLVM to HSA Code Object

    Parameters
    ----------
    obj_bin : str
        The object file

    Return
    ------
    cobj_bin : str
        The HSA Code Object
    """
    tmp_dir = util.tempdir()
    tmp_obj = tmp_dir.reloath("rocm_kernel.o")
    tmp_cobj = tmp_dir.reloath("rocm_kernel.co")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(obj_bin))
    cobj_bin = bytearray(open(tmp_cobj, "rb").read())
    return cobj_bin
