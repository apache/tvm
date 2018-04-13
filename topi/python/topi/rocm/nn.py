"""scheduler for normalization functions on rocm backend"""
from __future__ import absolute_import as _abs

import topi
from .. import generic

@generic.schedule_lrn.register(["rocm", "gpu"])
def schedule_lrn(outs):
    return topi.cuda.schedule_lrn(outs)

@generic.schedule_l2norm.register(["rocm", "gpu"])
def schedule_l2norm(outs):
    return topi.cuda.schedule_l2norm(outs)
