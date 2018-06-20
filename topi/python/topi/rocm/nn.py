"""scheduler for normalization functions on rocm backend"""
from __future__ import absolute_import as _abs

import tvm
from .. import generic
from .. import cpp

@generic.schedule_lrn.register(["rocm", "gpu"])
def schedule_lrn(outs):
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.rocm.schedule_lrn(cpp_target, outs)

@generic.schedule_l2_normalize.register(["rocm", "gpu"])
def schedule_l2_normalize(outs):
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.rocm.schedule_l2_normalize(cpp_target, outs)
