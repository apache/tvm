"""Tensor operator property registry

Provide information to lower and schedule tensor operators.
"""
from .attr_dict import AttrDict
from . import tensor
from . import nn
from . import transform
from . import reduction
from . import vision

from .registry import OpPattern
from .registry import register_compute, register_schedule, register_pattern
