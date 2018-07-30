# pylint: disable=invalid-name, unused-argument
"""Definition of image ops"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern

# resize
@reg.register_schedule("resize")
def schedule_resize(_, outs, target):
    """Schedule definition of resize"""
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outs)

reg.register_pattern("resize", OpPattern.INJECTIVE)
