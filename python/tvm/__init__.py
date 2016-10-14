"""Init proptype of the TVM"""
from __future__ import absolute_import as _abs

from .op import *
from .expr import Var, const
from .expr_util import *
from .tensor import Tensor
from .domain import RDom, Range, infer_range
