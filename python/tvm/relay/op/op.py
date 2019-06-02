# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#pylint: disable=unused-argument
"""The base node types for the Relay language."""
import topi

from ..._ffi.function import _init_api

from ..base import register_relay_node
from ..expr import Expr
from ...api import register_func
from ...build_module import lower, build
from . import _make

@register_relay_node
class Op(Expr):
    """A Relay operator definition."""

    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    def get_attr(self, attr_name):
        """Get additional attribute about the operator.

        Parameters
        ----------
        attr_name : str
            The attribute name.

        Returns
        -------
        value : object
            The attribute value
        """
        return _OpGetAttr(self, attr_name)


def get(op_name):
    """Get the Op for a given name

    Parameters
    ----------
    op_name : str
        The operator name

    Returns
    -------
    op : Op
        The op of the corresponding name
    """
    return _GetOp(op_name)


def register(op_name, attr_key, value=None, level=10):
    """Register an operator property of an operator.


    Parameters
    ----------
    op_name : str
        The name of operator

    attr_key : str
        The attribute name.

    value : object, optional
        The value to set

    level : int, optional
        The priority level

    Returns
    -------
    fregister : function
        Register function if value is not specified.
    """
    def _register(v):
        """internal register function"""
        _Register(op_name, attr_key, v, level)
        return v
    return _register(value) if value is not None else _register


class OpPattern(object):
    """Operator generic patterns

    See Also
    --------
    top.tag : Contains explanation of the tag type.
    """
    # Elementwise operator
    ELEMWISE = 0
    # Broadcast operator
    BROADCAST = 1
    # Injective mapping
    INJECTIVE = 2
    # Communication
    COMM_REDUCE = 3
    # Complex op, can still fuse ewise into it
    OUT_ELEMWISE_FUSABLE = 4
    # Represents tuple node
    TUPLE = 7
    # Not fusable opaque op
    OPAQUE = 8


def register_schedule(op_name, schedule=None, level=10):
    """Register schedule function for an op

    Parameters
    ----------
    op_name : str
        The name of the op.

    schedule : function (attrs: Attrs, outs: List[Tensor], target: Target) -> sch: Schedule
        The schedule function.

    level : int
        The priority level
    """
    return register(op_name, "FTVMSchedule", schedule, level)


def register_compute(op_name, compute=None, level=10):
    """Register compute function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    compute : function (attrs: Attrs, inputs: List[Tensor], out_type: Type, target:Target)
                       -> List[Tensor]
        The compute function.

    level : int
        The priority level
    """
    return register(op_name, "FTVMCompute", compute, level)


def register_alter_op_layout(op_name, alter_layout=None, level=10):
    """Register alter op layout function for an op

    Parameters
    ----------
    op_name : str
        The name of the operator

    alter_layout: function (attrs: Attrs, inputs: List[Expr]) -> new_expr: Expr
        The function for changing the layout or replacing the operator

    level : int
        The priority level
    """
    return register(op_name, "FTVMAlterOpLayout", alter_layout, level)


def register_pattern(op_name, pattern, level=10):
    """Register operator pattern for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    pattern : int
        The pattern being used.

    level : int
        The priority level
    """
    return register(op_name, "TOpPattern", pattern, level)

def register_gradient(op_name, fgradient=None, level=10):
    """Register operator pattern for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    fgradient : function (orig_expr : Expr, output_grad : Expr) -> new_expr : Expr
        The gradient being used.

    level : int
        The priority level
    """
    return register(op_name, "FPrimalGradient", fgradient, level)


_init_api("relay.op", __name__)

@register_func("relay.op.compiler._lower")
def _lower(name, schedule, inputs, outputs):
    return lower(schedule, list(inputs) + list(outputs), name=name)

@register_func("relay.op.compiler._build")
def _build(lowered_funcs):
    return build(lowered_funcs, target="llvm")


def schedule_injective(attrs, outputs, target):
    """Generic schedule for binary broadcast."""
    with target:
        return topi.generic.schedule_injective(outputs)


def schedule_concatenate(attrs, outputs, target):
    """Generic schedule for concatinate."""
    with target:
        return topi.generic.schedule_concatenate(outputs)


__DEBUG_COUNTER__ = 0

def debug(expr, debug_func=None):
    """The main entry point to the debugger."""
    global __DEBUG_COUNTER__

    if debug_func:
        name = "debugger_func{}".format(__DEBUG_COUNTER__)
        register_func(name, debug_func)
        __DEBUG_COUNTER__ += 1
    else:
        name = ''

    return _make.debug(expr, name)
