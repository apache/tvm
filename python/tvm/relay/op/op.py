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
# pylint: disable=unused-argument,invalid-name
"""The base node types for the Relay language."""
import tvm._ffi
import tvm.ir
import tvm.ir._ffi_api
from tvm.driver import build, lower
from tvm.runtime import Object
from tvm.target import GenericFunc, get_native_generic_func

from . import _make


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
    return tvm.ir.Op.get(op_name)


def register(op_name, describe=""):
    """Get the Op for a given name.
    when the op_name is not registered, create a new empty op with the given name.
    when the op_name has been registered, abort with an error message.

    Parameters
    ----------
    op_name : str
        The operator name

    describe : Optional[str]
        The operator description
    """

    tvm.ir._ffi_api.RegisterOp(op_name, describe)


def register_stateful(op_name, stateful, level=10):
    """Register stateful flag for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    stateful : bool
        The stateful flag.

    level : int
        The priority level
    """
    tvm.ir.register_op_attr(op_name, "TOpIsStateful", stateful, level)


class OpPattern(object):
    """Operator generic patterns

    See Also
    --------
    topi.tag : Contains explanation of the tag type.
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


@tvm._ffi.register_object("relay.OpImplementation")
class OpImplementation(Object):
    """Operator implementation"""

    def compute(self, attrs, inputs, out_type):
        """Call compute function.

        Parameters
        ----------
        attrs : Attrs
            Op attributes.

        inputs : list[te.tensor.Tensor]
            The input tensors.

        out_type : relay.Type
            The output type.

        Returns
        -------
        outs : list[te.tensor.Tensor]
            The output tensors.
        """
        return _OpImplementationCompute(self, attrs, inputs, out_type)

    def schedule(self, attrs, outs, target):
        """Call schedule function.

        Parameters
        ----------
        attrs : Attrs
            Op attributes.

        outs : list[te.tensor.Tensor]
            The output tensors.

        target : tvm.target.Target
            The target to schedule the op.

        Returns
        -------
        schedule : tvm.te.Schedule
            The schedule.
        """
        return _OpImplementationSchedule(self, attrs, outs, target)


@tvm._ffi.register_object("relay.OpSpecialization")
class OpSpecialization(Object):
    """Operator specialization"""


@tvm._ffi.register_object("relay.OpStrategy")
class OpStrategy(Object):
    """Operator strategy"""

    def __init__(self):
        self.__init_handle_by_constructor__(_make.OpStrategy)

    def add_implementation(self, compute, schedule, name="default", plevel=10):
        """Add an implementation to the strategy

        Parameters
        ----------
        compute : function (attrs: Attrs, inputs: List[Tensor], out_type: Type)
                           -> List[Tensor]
            The compute function.

        schedule : function (attrs: Attrs, outs: List[Tensor], target:Target) -> Schedule
            The schedule function.

        name : str
            The name of implementation.

        plevel : int
            The priority level of implementation.
        """
        _OpStrategyAddImplementation(self, compute, schedule, name, plevel)


def _wrap_default_fstrategy(compute, schedule, name):
    def _fstrategy(attrs, inputs, out_type, target):
        strategy = OpStrategy()
        strategy.add_implementation(compute, schedule, name=name)
        return strategy

    return _fstrategy


def _create_fstrategy_from_schedule(op_name, schedule):
    assert hasattr(schedule, "dispatch_dict")
    compute = get(op_name).get_attr("FTVMCompute")
    assert compute is not None, f"FTVMCompute is not registered for op {op_name}"
    fstrategy = get_native_generic_func(f"{op_name}_strategy")
    name_pfx = schedule.__name__
    name_pfx = name_pfx[name_pfx.index("_") + 1 :]
    fstrategy.set_default(
        _wrap_default_fstrategy(compute, schedule.fdefault, f"{name_pfx}.generic")
    )
    for key, sch in schedule.dispatch_dict.items():
        fstrategy.register(_wrap_default_fstrategy(compute, sch, f"{name_pfx}.{key}"), [key])
    return fstrategy


def register_compute(op_name, compute=None, level=10):
    """Register compute function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    compute : function (attrs: Attrs, inputs: List[Tensor], out_type: Type)
                       -> List[Tensor]
        The compute function.

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMCompute", compute, level)


def register_strategy(op_name, fstrategy=None, level=10):
    """Register strategy function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    fstrategy : function (attrs: Attrs, inputs: List[Tensor], out_type: Type,
                          target:Target) -> OpStrategy
        The strategy function. Need to be native GenericFunc.

    level : int
        The priority level
    """
    if not isinstance(fstrategy, GenericFunc):
        assert hasattr(fstrategy, "generic_func_node")
        fstrategy = fstrategy.generic_func_node
    return tvm.ir.register_op_attr(op_name, "FTVMStrategy", fstrategy, level)


def register_schedule(op_name, schedule, level=10):
    """Register schedule function for an op.

    This is used when compute function is the same for all targets and only
    schedule is different. It requires FTVMCompute is already registered to
    the op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    schedule : function (attrs: Attrs, outs: List[Tensor], target:Target) -> Schedule
        The schedule function. Need to be target.generic_func.

    level : int
        The priority level
    """
    fstrategy = _create_fstrategy_from_schedule(op_name, schedule)
    return register_strategy(op_name, fstrategy, level)


def register_injective_schedule(op_name, level=10):
    """Register injective schedule function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    level : int
        The priority level
    """
    return register_schedule(op_name, _schedule_injective, level)


def register_broadcast_schedule(op_name, level=10):
    """Register broadcast schedule function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    level : int
        The priority level
    """
    return register_schedule(op_name, _schedule_injective, level)


def register_reduce_schedule(op_name, level=10):
    """Register reduce schedule function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    level : int
        The priority level
    """
    return register_schedule(op_name, _schedule_reduce, level)


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
    return tvm.ir.register_op_attr(op_name, "FTVMAlterOpLayout", alter_layout, level)


def register_convert_op_layout(op_name, convert_layout=None, level=10):
    """Register convert op layout function for an op

    Parameters
    ----------
    op_name : str
        The name of the operator

    convert_layout: function (attrs: Attrs, inputs: List[Expr]) -> new_expr: Expr
        The function for changing the layout or replacing the operator

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMConvertOpLayout", convert_layout, level)


def register_infer_correct_layout(op_name, infer_layout=None, level=10):
    """Register infer op layout function for an op

    Parameters
    ----------
    op_name : str
        The name of the operator

    infer_layout: function (attrs: Attrs, inputs: List[Layout]) -> InferCorrectLayoutOutput
        The function to infer correct layout

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FInferCorrectLayout", infer_layout, level)


def register_legalize(op_name, legal_op=None, level=10):
    """Register legal transformation function for an op

    Parameters
    ----------
    op_name : str
        The name of the operator

    legal_op: function (attrs: Attrs, inputs: List[Expr]) -> new_expr: Expr
        The function for transforming an expr to another expr.

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMLegalize", legal_op, level)


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
    return tvm.ir.register_op_attr(op_name, "TOpPattern", pattern, level)


def register_gradient(op_name, fgradient=None, level=10):
    """Register operator gradient function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    fgradient : function (orig_expr : Expr, output_grad : Expr) -> new_expr : Expr
        The gradient being used.

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FPrimalGradient", fgradient, level)


def register_shape_func(op_name, data_dependent, shape_func=None, level=10):
    """Register operator shape function for an op.

    Parameters
    ----------
    op_name : str
        The name of the op.

    data_dependent : bool or list of bool
        Whether the shape function depends on input data. If this is a list of bool,
        the length of the list must be the same as the number of arguments of this op.
        The list specifies per-input data dependence of the op.

    shape_func : function (attrs: Attrs, inputs: List[Tensor], out_ndims: List[IndexExpr])
                 -> shape_tensors: List<Tensor>
        The function for computing the dynamic output shapes

    level : int
        The priority level
    """
    if not isinstance(data_dependent, list):
        data_dependent = [data_dependent]
    get(op_name).set_attr("TShapeDataDependent", data_dependent, level)
    return tvm.ir.register_op_attr(op_name, "FShapeFunc", shape_func, level)


def register_external_compiler(op_name, fexternal=None, level=10):
    """Register the external compiler for an op.

    Parameters
    ----------
    op_name : str
        The name of the operator.

    fexternal : function (attrs: Attrs, args: List[Expr], compiler: str)
              -> new_expr: Expr
        The function for wrapping a call expr with compiler_begin and
        compiler_end.

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMExternalCompiler", fexternal, level)


def register_fake_quantization_to_integer(op_name, func=None, level=10):
    """Register quantize function for an op

    Given an op and Affine Types on it's inputs, this function should return the op
    in affine space/integer operators and the new type of the output, where affine
    denotes the transformation x_real = (x_affine - zero_point) * scale

    Parameters
    ----------
    op_name : str
        The name of the operator

    func: function (expr: Expr, map: Map<Expr, AffineType>) -> new_expr: Expr
        The function for translating the op into affine space and integer operators

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMFakeQuantizationToInteger", func, level)


def register_optional_fake_quantization_to_integer(op_name, func=None, level=10):
    """Register optional quantize function for an op

    Given an op and Affine Types on it's inputs, this function should return the op
    in affine space/integer operators and the new type of the output, where affine
    denotes the transformation x_real = (x_affine - zero_point) * scale

    Parameters
    ----------
    op_name : str
        The name of the operator

    func: function (expr: Expr, map: Map<Expr, AffineType>) -> new_expr: Expr
        The function for translating the op into affine space and integer operators

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMOptionalFakeQuantizationToInteger", func, level)


def register_mixed_precision_conversion(op_name, func=None, level=10):
    """Register mixed precision conversion function for an op

    Given an op the function should return information on how the value should be
    converted. Specifically the function should take a call node and the target
    mixed precision datatype (e.g. FP16) and return the conversion category
    (see python/tvm/relay/transform/mixed_precision.py) as well as the accumulation
    and output datatype of the operation in the mixed precision dtype space.

    Parameters
    ----------
    op_name : str
        The name of the operator

    func: function (call_node: relay.Call, target_dtype: string)
    -> [conversion category, accumulation dtype, output dtype]: [int, string, string]
        A function which given a call_node and target_dtype (e.g. FP16) returns the
        conversion category and associated accumulation/output of the operation
        when transformed into the mixed precision dtype space.

    level : int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FTVMMixedPrecisionConversionType", func, level)


@tvm._ffi.register_func("relay.op.compiler._lower")
def _lower(name, schedule, inputs, outputs):
    return lower(schedule, list(inputs) + list(outputs), name=name)


@tvm._ffi.register_func("relay.op.compiler._build")
def _build(lowered_funcs):
    return build(lowered_funcs, target="llvm")


_schedule_injective = None
_schedule_reduce = None

__DEBUG_COUNTER__ = 0


def debug(expr, debug_func=None):
    """The main entry point to the debugger."""
    global __DEBUG_COUNTER__

    if debug_func:
        name = f"debugger_func{__DEBUG_COUNTER__}"
        tvm._ffi.register_func(name, debug_func)
        __DEBUG_COUNTER__ += 1
    else:
        name = ""

    return _make.debug(expr, name)


tvm._ffi._init_api("relay.op", __name__)
