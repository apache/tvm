"""Reuse conv2d schedule from ARM CPU"""

import tvm

from topi.nn import conv2d, conv2d_alter_layout
from topi import generic

@conv2d.register(["vtacpu", "vta"])
def compute(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return conv2d(*args, **kwargs)

@generic.schedule_conv2d_nchw.register(["vtacpu", "vta"])
def schedule(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return generic.schedule_conv2d_nchw(*args, **kwargs)

@conv2d_alter_layout.register(["vtacpu", "vta"])
def alter(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return conv2d_alter_layout(*args, **kwargs)
