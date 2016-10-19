"""Init proptype of the TVM"""
from __future__ import absolute_import as _abs

from .op import *
from .expr import Var, const
from .expr_util import *
from .tensor import Tensor
from .domain import Range, RDom, infer_range
from .split import Split
from .buffer import Scope, Buffer
from .schedule import Schedule
