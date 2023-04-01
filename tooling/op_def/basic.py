"""Basic operators builtin in Relax."""
# pylint: disable=too-few-public-methods

from ..registry import register_op
from ..ty import (
    AnyRelaxExpr,
    ExternFunc,
    GlobalVar,
    Optional,
    Shape,
    StructInfo,
    Tensor,
    TupleExpr,
)

# TODO: implement R.print, R.assert


@register_op("null_value", sinfo="InferStructInfoReturnsObject")
class NullValue:
    """Create a call node that represents a null value object.

    Attributes
    ----------
    ret
        The created call node.
    """

    ret = AnyRelaxExpr


@register_op("shape_of", sinfo="InferStructInfoShapeOf")
class ShapeOf:
    """Get shape of a tensor. It gets TensorStructInfo and returns ShapeStructInfo

    Attributes
    ----------
    expr
        The input expression of TensorStructInfo.
    ret
        The created call node.
    """

    expr = Tensor
    ret = Shape


@register_op("call_tir", sinfo="InferStructInfoReturnFirstSInfoArg", min_num_args=2)
class CallTIR:
    """Call a PrimFunc in TensorIR, and return its output using a special calling convention
    called destination-passing style (DPS) in TVM.

    In DPS, the caller supplies the callee with arguments that include both inputs and outputs,
    and the callee writes the results to the output arguments. This convention, where caller passes
    the destination of results to the callee to write to, is called destination-passing.

    For example, if a function `f` takes `n` inputs and returns `m` outputs, in DPS, the caller
    supplies `n + m` arguments, and the callee writes the results to the last `m` arguments.

    Note that DPS is the default calling convention across TVM stack, not just in TensorIR,
    but also includes global PackedFunc. When calling a PackedFunc in DPS, the counterpart
    of `call_tir` is `call_dps_packed`.

    Attributes
    ----------
    gvar
        The global variable that points to the function being called.
    args
        The arguments to the function. Always a Relax Tuple expression whose length indicates
        the number `n` in the example the number of arguments.
    out_sinfo
        The StructInfo of the output. It is used to infer the number of outputs, and indicates
        the number `m` in the example.
    tir_vars
        The TIR variables to be used with the call. They are usually used for symbolic shapes.
    ret
        The created call node.
    """

    gvar = GlobalVar
    args = TupleExpr
    out_sinfo = StructInfo
    tir_vars = Optional(
        Shape,
        default=None,
        cc_arg2relax=lambda name: f"{name}.value_or(relax::Expr{{nullptr}})",
    )
    ret = AnyRelaxExpr


@register_op("call_dps_packed", sinfo="InferStructInfoReturnFirstSInfoArg")
class CallDpsPacked:
    """Call a PackedFunc in destination-passing style (DPS).

    In DPS, the caller supplies the callee with arguments that include both inputs and outputs,
    and the callee writes the results to the output arguments. This convention, where caller passes
    the destination of results to the callee to write to, is called destination-passing.

    For example, if a function `f` takes `n` inputs and returns `m` outputs, in DPS, the caller
    supplies `n + m` arguments, and the callee writes the results to the last `m` arguments.

    Note that DPS is the default calling convention across TVM stack, not just in PackedFunc,
    but also includes TensorIR. When calling a PrimFunc in TensorIR, the counterpart
    of `call_dps_packed` is `call_tir`.

    Attributes
    ----------
    func
        The function being called.
    args
        The arguments to the packed func. Always a Relax Tuple expression whose length indicates
        the number `n + m` in the example.
    out_sinfo
        The StructInfo of the output.
    ret
        The created call node.
    """

    func = ExternFunc
    args = TupleExpr
    out_sinfo = StructInfo
    ret = AnyRelaxExpr


@register_op("call_builtin_with_ctx", sinfo="InferStructInfoReturnFirstSInfoArg")
class CallBuiltinWithCtx:
    """Call a VM builtin PackedFunc in destination-passing style (DPS). The difference between
    `call_builtin_with_ctx` and `call_dps_packed` is that `call_builtin_with_ctx` takes
    an extra argument `ctx` at the beginning of the arguments, which is the context of the
    current VM.

    Attributes
    ----------
    func
        The function being called.
    args
        The arguments to the packed func. Always a Relax Tuple expression.
    sinfo_args
        The StructInfo of the arguments.
    ret
        The created call node.
    """

    func = ExternFunc
    args = TupleExpr
    sinfo_args = StructInfo
    ret = AnyRelaxExpr


@register_op("make_closure", sinfo="InferStructInfoReturnsObject")
class MakeClosure:
    """Create a closure with free variables and return the closure.

    Attributes
    ----------
    func
        The function being called.
    args
        The arguments to the packed func. Always a Relax Tuple expression.
    ret
        The created call node.
    """

    func = GlobalVar
    args = TupleExpr
    ret = AnyRelaxExpr


@register_op("invoke_closure", sinfo="InferStructInfoReturnFirstSInfoArg")
class InvokeClosure:
    """Invoke a closure.

    Attributes
    ----------
    closure
        The closure being invoked.
    args
        The arguments to the closure. Always a Relax Tuple expression.
    sinfo_args
        The StructInfo of the output
    ret
        The created call node.
    """

    closure = AnyRelaxExpr  # TODO: more specific type?
    args = TupleExpr
    sinfo_args = StructInfo
    ret = AnyRelaxExpr
