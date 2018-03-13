"""scheduler for normalization functions on rocm backend"""
from __future__ import absolute_import as _abs

import topi
from .. import generic

@generic.schedule_lrn.register(["rocm", "gpu"])
def schedule_lrn(outs):
    return topi.cuda.schedule_lrn(outs)
