# pylint: disable=invalid-name, unused-variable,
"""Schedule for upsampling operator"""
import tvm
from .. import generic
from .injective import schedule_injective

@generic.schedule_upsampling.register(["cuda", "gpu"])
def schedule_upsampling(outs):
    return schedule_injective(outs)
