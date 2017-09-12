# tvm.contrib.rocm

from . import util

@tvm.register_func("tvm_callback_rocm_link")
def callback_rocm_link(obj_bin):
    return obj_bin
