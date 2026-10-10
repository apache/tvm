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
# pylint: disable=redefined-builtin
# ruff: noqa: F821
"""The base Relax operators."""

from collections.abc import Callable

import tvm_ffi

import tvm
import tvm.runtime
from tvm.ir import Attrs, Call, Op
from tvm.ir.attrs import make_node as _make_attrs
from tvm.ir.location import UNKNOWN_LOC, Location
from tvm.ir.op import _make_op_api
from tvm.runtime import Object, ObjectConvertible

from ..expr import Expr, ExternFunc, GlobalVar, Var
from ..type import Type
from ..utils import convert_to_expr

py_print = print  # pylint: disable=invalid-name


def register_gradient(
    op_name: str,
    fgradient: Callable[[Var, Call, Var, "BlockBuilder"], list[Expr]] | None = None,
    override: bool = False,
):
    """Register operator gradient function for a relax operator.

    Parameters
    ----------
    op_name: str
        The name of the op.

    fgradient: function (orig_var: Var, orig_call: Call, output_grad: Var, ctx: BlockBuilder)
         -> partials: List[Expr]
        The gradient function being used.

    override: bool, optional
        Replace an existing gradient if True; duplicate registration otherwise
        raises ValueError.

    Returns
    -------
    result : callable
        The registered gradient, or a decorator if fgradient is not supplied.
    """
    return tvm.ir.register_op_attr(op_name, "FPrimalGradient", fgradient, override)


def null_value(*, ty=None, loc: Location = UNKNOWN_LOC) -> Call:
    """Create a call node that represents a null value object.

    Returns
    -------
    ret: Call
        The created call node.
    """
    return Call("relax.null_value", [], ty=ty, loc=loc)  # type: ignore


def _wrap_inline_arg_tuple(args) -> Expr:
    """Helper function to wrap argument tuple

    Normalize the arguments provided the functions that accept a tuple
    of arguments, and require the tuple of arguments to be written
    in-line.  If the arguments provided are a single relax expression,
    and are not a reference to a relax tuple, then wrap them into an
    in-line relax Tuple.

    """
    if isinstance(args, tuple | list):
        return tvm.relax.Tuple([convert_to_expr(a) for a in args])
    elif (
        isinstance(args, Expr)
        and not isinstance(args, tvm.relax.Tuple)
        and (args.ty is None or not isinstance(args.ty, tvm.relax.TupleType))
    ):
        return tvm.relax.Tuple([args])
    else:
        return args


_call_tir = _make_op_api(Op.get("relax.call_tir"), __name__)


def call_tir(
    func, args, *, ty_args, attrs=None, ty=None, loc: Location = UNKNOWN_LOC, **kwargs
) -> Call:
    """Call a destination-passing TIR function and allocate its output tensors.

    ``func`` is the function's GlobalVar and ``args`` contains its ordered inputs.
    ``ty_args`` contains one output type: a TensorType or a TupleType of tensor
    results. Omitted ``ty`` uses the registered result inference; explicit types
    and locs are forwarded unchanged.
    """
    return _call_tir(
        func, _wrap_inline_arg_tuple(args), ty_args=ty_args, attrs=attrs, ty=ty, loc=loc, **kwargs
    )


def call_tir_packed(gvar: GlobalVar, args: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Call:
    """Call a TIRx Function through its native packed-call contract.

    Every native parameter is supplied explicitly, in order.  Unlike
    :py:func:`call_tir`, this operator does not allocate destination tensors or
    interpret destination parameters as results.  It performs no implicit
    copies, casts, layout conversions, device transfers, or redistribution.

    The result follows the native declared return type: a supported scalar
    keeps its exact primitive type, void becomes the empty tuple, and a pointer
    becomes an ``Any`` carrying an opaque pointer.  A pointer result does not
    imply tensor ownership or a lifetime guarantee.  Native tensor, nonempty
    tuple, callable, vector, and other unsupported return types are rejected.
    There is no ``out_ty`` argument.

    Parameters
    ----------
    gvar : GlobalVar
        The GlobalVar referring to a TIRx Function with its native
        ``tvm.ir.FuncType`` signature.

    args : Expr
        The ordered arguments, supplied as an inline Relax tuple, a Python
        tuple or list, or a single expression.  Tensor parameters accept
        compatible Relax tensors, with dtype, rank, and known shapes checked
        against the native signature.  Remaining supported native tensor
        constraints are checked by the packed ABI at runtime.  Specialized
        global storage scopes require matching ``VDevice.memory_scope``;
        non-default layouts, allocated-address contracts, unsupported storage
        scopes, and unlowered distributed tensors are rejected.

        Scalar parameters require the exact primitive dtype: scalar bool,
        signed or unsigned integers up to 64 bits, or float16/32/64.  These
        same scalar types are supported as direct results.  The existing packed
        integer carrier is signed 64-bit: ``uint64`` values must be in
        ``[0, 2**63 - 1]``; larger unsigned values are not representable.

        Pointer parameters accept ``Any`` or a handle-compatible object,
        including a runtime tensor.  A tensor passed to a pointer parameter
        supplies its DLTensor header handle, not its data pointer.  This erased
        carrier does not prove pointee type, address space, ownership, or
        lifetime compatibility.  Known scalar values cannot serve as pointer
        arguments.  The runtime carrier must satisfy the existing packed ABI's
        null, opaque-pointer, DLTensor-pointer, or object-handle check.

    The call is effectful and may mutate its arguments. For a call known to
    have no observable effects, use an explicit ``call_pure_packed`` wrapper
    around the ``relax.call_tir_packed`` operator. Purity is never inferred
    from the native signature or packed ABI.

    Returns
    -------
    ret : Call
        A call whose result type is derived from the native declared return
        during Relax type inference.

    Examples
    --------
    A native ``(int64, int64) -> int64`` function returns its scalar directly::

        result = relax.call_tir_packed(add_scalar, (a, b))

    A caller that knows the scalar function has no effects may assert purity::

        result = relax.call_pure_packed(
            tvm.ir.Op.get("relax.call_tir_packed"), add_scalar, (a, b)
        )

    A native copy function with a void return writes to a caller-owned tensor::

        relax.call_tir_packed(copy, (source, destination))
    """
    args = _wrap_inline_arg_tuple(args)
    return Call("relax.call_tir_packed", [gvar, args], ty=ty, loc=loc)


@tvm_ffi.register_object("relax.attrs.CallTIRWithGradAttrs")
class CallTIRWithGradAttrs(Attrs):
    """Attributes used in call_tir_with_grad operator"""


_call_tir_with_grad = _make_op_api(Op.get("relax.call_tir_with_grad"), __name__)


def call_tir_with_grad(
    func, args, *, ty_args, attrs=None, ty=None, loc: Location = UNKNOWN_LOC, **kwargs
) -> Call:
    """Call a TIR function with a registered TE gradient rule.

    ``ty_args`` contains the single output type, including a TupleType for
    multiple results. The ``te_grad_name`` and optional ``te_grad_kwargs``
    attribute keywords select the rule used by the Gradient pass. Attributes
    may instead be supplied through ``attrs``.
    """
    if attrs is None and kwargs.get("te_grad_kwargs") is None:
        kwargs["te_grad_kwargs"] = {}
    return _call_tir_with_grad(
        func, _wrap_inline_arg_tuple(args), ty_args=ty_args, attrs=attrs, ty=ty, loc=loc, **kwargs
    )


@tvm_ffi.register_object("relax.attrs.CallTIRInplaceAttrs")
class CallTIRInplaceAttrs(Attrs):
    """Attributes used in call_tir_inplace operator"""


_call_tir_inplace = _make_op_api(Op.get("relax.call_tir_inplace"), __name__)


def call_tir_inplace(
    func, args, *, ty_args, attrs=None, ty=None, loc: Location = UNKNOWN_LOC, **kwargs
) -> Call:
    """Call a TIR function whose selected outputs alias its input tensors.

    ``ty_args`` contains one output type. In the ``inplace_indices`` attribute,
    entry ``i >= 0`` makes the corresponding output alias input ``i``; ``-1``
    allocates a fresh output. At least one output must alias an input.

    Although classified as pure, this operation mutates the selected inputs.
    Only use it after proving there are no live uses or aliases that could
    observe those mutations. Direct construction is intended for testing;
    optimization passes normally establish these preconditions.
    """
    # Keep the existing scalar-index convenience at the Python boundary.
    if isinstance(kwargs.get("inplace_indices"), int):
        kwargs["inplace_indices"] = [kwargs["inplace_indices"]]
    return _call_tir_inplace(
        func, _wrap_inline_arg_tuple(args), ty_args=ty_args, attrs=attrs, ty=ty, loc=loc, **kwargs
    )


_call_dps_packed = _make_op_api(Op.get("relax.call_dps_packed"), __name__)


def call_dps_packed(
    func, args, *, ty_args, attrs=None, ty=None, loc: Location = UNKNOWN_LOC, **kwargs
) -> Call:
    """Call a destination-passing packed function and allocate its outputs.

    Python string callees become ExternFunc; explicit Expr callees retain
    their identity. ``ty_args`` contains one output type, including a TupleType
    for multiple results.

    The function must be pure apart from writing its designated outputs.
    Other effects may be removed, reordered or repeated by the compiler.
    """
    if isinstance(func, str):
        func = ExternFunc(func)
    if isinstance(args, tuple | list):
        args = tvm.ir.Tuple(args)
    else:
        args = _wrap_inline_arg_tuple(args)
    return _call_dps_packed(func, args, ty_args=ty_args, attrs=attrs, ty=ty, loc=loc, **kwargs)


def call_py_func(
    func_name: str | Expr, args: Expr, *, ty_args, ty=None, loc: Location = UNKNOWN_LOC
) -> Call:
    """Call a Python function using canonical operands and one output type argument.

    ``func_name`` names a function in the IRModule's ``pyfuncs`` attribute.
    ``args`` accepts an Expr or a Python tuple through shared Expr conversion.
    ``ty_args`` contains exactly one result type, including a TupleType for
    tuple-valued results. Omitted ``ty`` uses the registered result inference.
    """
    return Call("relax.call_py_func", [func_name, args], ty_args=ty_args, ty=ty, loc=loc)


def call_builtin_with_ctx(
    func: str | Expr,
    args: Expr,
    *,
    ty_args: Type | list[Type] | None = None,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Call:
    """Call a builtin function func.

    Parameters
    ----------
    func : Expr
        The builtin function to be called.

    args : Expr
        The input arguments.

    ty_args: Optional[Union[Type, List[Type]]]
        The type arguments to the call node.

    Returns
    -------
    ret: Call
        The created call node.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    args = _wrap_inline_arg_tuple(args)

    if ty_args is not None and not isinstance(ty_args, list | tuple):
        ty_args = [ty_args]

    if ty_args is not None:
        ty_args = [
            value()
            if callable(value)
            else value.asobject()
            if isinstance(value, ObjectConvertible)
            else value
            for value in ty_args
        ]
    return Call(
        "relax.call_builtin_with_ctx",
        [func, args],
        ty_args=ty_args,
        ty=ty,
        loc=loc,
    )


def make_closure(func: Expr, args: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Object:
    """
    Create a closure with free variables and return the closure.

    Parameters
    ----------
    func : Expr
        The closure, can be ExternFunc or Function.

    args : Expr
        The input arguments.


    Returns
    -------
    ret: Object
        The VMClosure.
    """

    args = _wrap_inline_arg_tuple(args)

    return Call("relax.make_closure", [func, args], ty=ty, loc=loc)  # type: ignore


def invoke_closure(
    closure: Expr, args: Expr, ty_args: list[Type] | Type, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Call:
    """
    Invoke a closure.

    Parameters
    ----------
    closure : Expr
        The VMClosure object.

    args : Expr
        The input arguments.

    type_args: Union[List[Type], Type]
        The type information arguments of the CallNode

    Returns
    -------
    ret: Call
        A call to `invoke_closure`.
    """
    args = _wrap_inline_arg_tuple(args)

    if not isinstance(ty_args, list | tuple):
        ty_args = [ty_args]

    return Call(
        "relax.invoke_closure",
        [closure, args],
        ty_args=ty_args,
        ty=ty,
        loc=loc,
    )  # type: ignore


def render_object(val: tvm.Object) -> str:
    """
    Given a TVM Object, renders it in string form. Used for Relax printing and assertions.

    Parameters
    ----------
    val: tvm.Object
        An object to render

    Returns
    -------
    ret: str
        A string representing the value, ideally human-readable
    """
    if isinstance(val, tvm.runtime.Tensor):
        return str(val)
    if isinstance(val, tvm_ffi.Array):
        fields = ", ".join([render_object(val[i]) for i in range(len(val))])
        return f"({fields})"
    return str(val)


@tvm.register_global_func("relax.run.shape_to_tensor")
def relax_shape_to_tensor(shape_tuple: tvm_ffi.Shape) -> tvm.runtime.Tensor:
    """
    Takes a Shape and convert it to Tensor.

    Parameters
    ----------
    shape_tuple: tvm_ffi.Shape
        Shape tuple that we want to convert to Tensor at runtime
    """
    return tvm.runtime.tensor([int(v) for v in shape_tuple])


@tvm.register_global_func("relax.run.print")
def relax_print(format_str: str, *format_args: tvm.Object) -> None:
    """
    Takes a list of values to print, formats with the given format string.
    If the format string is empty, simply prints.

    Call from TVM script like this:
    `relax.print(format_str, value1, value2, ..., valueN)`
    or
    `relax.print("", value1, value2, ..., valueN)`

    Parameters
    ----------
    format_str: str
        The first argument is a Python-style format string for printing the values

    format_args: List[Object]
        The values to print.
    """
    val_strs = map(render_object, format_args)
    if format_str == "":
        py_print(*val_strs)
    else:
        py_print(format_str.format(*val_strs))


def print(format: str | Expr, *values: Expr, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Print values using the canonical leading format-string operand."""
    return Call("relax.print", [format, *values], ty=ty, loc=loc)


@tvm.register_global_func("relax.run.assert_op")
def relax_assert_op(condition: tvm.Object, format_str: str, *format_args: tvm.Object) -> None:
    """
    A variadic function. The first value serves as the assertion condition:
    If the condition is true, then the operator does nothing.
    If the condition is false, then the operator raises an assertion error.

    The second argument is the format string for the error message, followed by
    its format arguments.
    If the format string is the empty string, then the error message will simply include
    a comma-separated list of the format arguments.
    The condition argument is not included in the format string.

    Parameters
    ----------
    condition: tvm.Object
        The assertion condition. Must be a boolean scalar.

    format_str: str
        A Python-style format string for the error message.

    format_args: List[tvm.Object]
        Values used for formatting the string.
    """
    if not isinstance(format_str, str):
        raise ValueError(
            f"The format string argument to assert must be a string, given {type(format_str)})"
        )

    if isinstance(condition, bool | int):
        val = condition
    elif isinstance(condition, tvm.runtime.Tensor):
        # may happen if the original program had unknown shape or dtype for the tensor's type
        dtype = condition.dtype
        if dtype != "bool":
            raise ValueError(f"The condition must be a bool scalar, but given a {dtype} tensor")
        shape = condition.shape
        if len(shape) != 0:
            raise ValueError(f"The condition must be a scalar, but it has a shape of {shape}")

        val = condition.numpy()

    else:
        # should be guaranteed by the type system
        raise ValueError(
            f"The condition for relax assert must be a bool, int, or Tensor, "
            f"but received a {type(condition)}."
        )

    if not val:
        error_message = "Assertion Failed"
        if format_args or format_str != "":
            rendered = map(render_object, format_args)
            if format_str != "":
                error_message = format_str.format(*rendered)
            else:
                error_message = ", ".join(rendered)
        raise AssertionError(error_message)


def assert_op(
    condition: Expr,
    format: str | Expr = "",
    *values: Expr,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """
    Create a call to Relax's assert_op operation (`assert` is reserved in Python,
    so the name must be distinct).

    Parameters
    ----------
    condition: Expr
        The assertion condition.

    format: Union[str, Expr]
        The format string or StringImm for the error message. If empty, the
        values are rendered as a comma-separated list.

    values: Expr
        Values used to format the error message if the condition fails.
        A tuple-valued expression is one value unless explicitly expanded.

    Returns
    -------
    result : Expr
        A Call to the Relax assert operation.
    """
    return Call("relax.assert_op", [condition, format, *values], ty=ty, loc=loc)


def shape_of(expr: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Get shape of a tensor.

    Parameters
    ----------
    expr : Expr
        The input Expr.

    Returns
    -------
    result : Expr
        A relax Call, which gets the shape of the input
    """
    return Call("relax.shape_of", [expr], ty=ty, loc=loc)  # type: ignore # pylint: disable=no-member


def size(expr: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Get the total number of elements in a tensor.

    Parameters
    ----------
    expr : Expr
        The input tensor.

    Returns
    -------
    result : Expr
        A scalar tensor of dtype int64 containing the total number of elements.
    """
    return Call("relax.size", [expr], ty=ty, loc=loc)  # type: ignore # pylint: disable=no-member


def tensor_to_shape(expr: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Convert tensor to shape expr.
    Parameters
    ----------
    expr : Expr
        The input Expr
    Returns
    -------
    result : Expr
        A relax Call, which transforms the tensor values to the shape
    """
    return Call("relax.tensor_to_shape", [expr], ty=ty, loc=loc)  # type: ignore # pylint: disable=no-member


def shape_to_tensor(expr: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Convert shape to tensor expr.
    Parameters
    ----------
    expr : Expr
        The input Expr
    Returns
    -------
    result : Expr
        A relax Call, which transforms the shape values to the tensor
    """
    return Call("relax.shape_to_tensor", [expr], ty=ty, loc=loc)  # type: ignore # pylint: disable=no-member


@tvm_ffi.register_object("relax.attrs.CallInplacePackedAttrs")
class CallInplacePackedAttrs(Attrs):
    """Attributes used in call_inplace_packed operator"""


def call_inplace_packed(
    func: str | ExternFunc | GlobalVar,
    *args: Expr,
    inplace_indices: int | list[int] | None = None,
    ty_args: Type | list[Type] | None = None,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """
    Construct a call to a packed function that consumes some of its arguments "in-place"
    and returns the mutated arguments (aliased), but should be considered to be otherwise pure.
    The `inplace_indices` argument indicates which of the outputs are mutated arguments.

    The resulting call will have the same semantics as calling the packed function directly.

    Note: This should be used for cases when the user knows that calling the packed function
    with these arguments will **in reality** not cause any other side effects.
    If it is used for a call that **does** result in other side effects, then the compiler
    may end up removing, reordering, or repeating that call, with no guarantees
    made about any side effects from the callee.

    Warning: This operator as treated as pure by the type system even though it *is* performing
    side effects (mutating some arguments). It is therefore incumbent upon the user to ensure
    that it is being used safely (viz., that mutated arguments are not live after the mutation,
    that they do not alias values live after the mutation).

    Parameters
    ----------
    func : Union[str, ExternFunc]
      The name (global symbol) for a PackedFunc or an ExternFunc node.

    args: Expr
      The arguments for the PackedFunc.

    inplace_indices : Union[int, List[int]]
      Specify which arguments should be used for in-place computations.
      If `inplace_indices` is a single integer, it will be made into a singleton list.
      Suppose `inplace_indices[i] = j`, where `j >= 0`. Then the `i`th output
      will be an alias of `args[j]`.
      If `inplace_indices[i] = -1`, then the `i`th output will be a freshly allocated tensor.
      At least one member of `inplace_indices` must not be -1.

    ty_args: Union[Type, List[Type]]
        The list of type information arguments (giving the type information for the returned value).

    Returns
    -------
    result : Expr
      A Relax call, corresponding to
      `call_pure_packed(ExternFunc(func), args, DictAttrs(kwargs), ty_args)`
    """
    op = ExternFunc(func) if isinstance(func, str) else func
    args = tuple(convert_to_expr(a) for a in args)
    if ty_args is None:
        ty_args = []
    if isinstance(ty_args, tuple):  # type: ignore
        ty_args = list(ty_args)
    elif not isinstance(ty_args, list):
        ty_args = [ty_args]
    if inplace_indices is not None and not isinstance(inplace_indices, list):
        inplace_indices = [inplace_indices]

    return Call(
        "relax.call_inplace_packed",
        [op, *args],
        attrs=_make_attrs("relax.attrs.CallInplacePackedAttrs", inplace_indices=inplace_indices),
        ty_args=ty_args,
        ty=ty,
        loc=loc,
    )  # type: ignore # pylint: disable=no-member


def call_pure_packed(
    func: str | ExternFunc | GlobalVar | Op,
    *args: Expr,
    ty_args: Type | list[Type] | None = None,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """
    Construct a call to a packed function that should be treated as pure,
    even though packed calls are normally not treated as pure.

    The resulting call will have the same semantics as calling the packed function directly.

    Note: This should be used for cases when the user knows that calling the packed function
    with these arguments will **in reality** not cause any side effects.
    If it is used for a call that **does** result in side effects, then the compiler
    may end up removing, reordering, or repeating that call, with no guarantees
    made about any side effects from the callee.

    Parameters
    ----------
    func : Union[str, ExternFunc, Op]
      The name (global symbol) for a PackedFunc or an ExternFunc node.
      The explicit ``relax.call_tir_packed`` Op is also accepted; its native
      callee and argument tuple follow as arguments to this wrapper.

    args: Expr
      The arguments for the PackedFunc.

    ty_args: Union[Type, List[Type]]
        The list of type information arguments (giving the type information for the returned value).
        Omit this for the native bridge, whose result follows the native signature.

    Returns
    -------
    result : Expr
      A Relax call, corresponding to
      `call_pure_packed(ExternFunc(func), args, DictAttrs(kwargs), ty_args)`
    """
    op = ExternFunc(func) if isinstance(func, str) else func
    args = tuple(convert_to_expr(a) for a in args)

    if ty_args is None:
        ty_args = []

    if isinstance(ty_args, tuple):  # type: ignore
        ty_args = list(ty_args)
    elif not isinstance(ty_args, list):
        ty_args = [ty_args]

    ty_args = [
        (ty() if callable(ty) else ty.asobject() if isinstance(ty, ObjectConvertible) else ty)
        for ty in ty_args
    ]

    # note: if we need attributes, we can also take them here

    return Call(
        "relax.call_pure_packed",
        [op, *args],
        ty_args=ty_args,
        ty=ty,
        loc=loc,
    )  # type: ignore # pylint: disable=no-member


def invoke_pure_closure(
    closure: Expr, args: Expr, ty_args: list[Type] | Type, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Call:
    """
    Invoke a closure and indicate to the compiler that it is pure.

    Note: This should be used for cases when the user knows that calling the closure
    with these arguments will **in reality** not cause any side effects.
    If it is used for a call that _does_ result in side effects, then the compiler
    may end up removing, reordering, or repeating that call, with no guarantees
    made about any side effects from the callee.

    Parameters
    ----------
    closure : Expr
        The VMClosure object.

    args : Expr
        The input arguments.

    type_args: Union[List[Type], Type]
        The type information arguments of the CallNode

    Returns
    -------
    ret: Call
        A call to `invoke_pure_closure`.
    """
    args = _wrap_inline_arg_tuple(args)

    if not isinstance(ty_args, list | tuple):
        ty_args = [ty_args]

    return Call(
        "relax.invoke_pure_closure",
        [closure, args],
        ty_args=ty_args,
        ty=ty,
        loc=loc,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.ToVDeviceAttrs")
class ToVDeviceAttrs(Attrs):
    """Attributes used in to_vdevice operator"""


def to_vdevice(data, dst_vdevice, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Copy data to the destination device. This
    operator helps data transferring between difference devices for
    heterogeneous execution.

    Parameters
    ----------
    data : Expr
        The tensor to be copied.

    dst_device : VDevice
        The destination device where the data is copied to.

    Returns
    -------
    result : Expr
        The copied result.
    """
    return Call(
        "relax.to_vdevice",
        [data],
        attrs=_make_attrs("relax.attrs.ToVDeviceAttrs", dst_vdevice=dst_vdevice),
        ty=ty,
        loc=loc,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.HintOnDeviceAttrs")
class HintOnDeviceAttrs(Attrs):
    """Attributes used in hint_on_device operator"""


def hint_on_device(
    data, device_type, index=0, memory_scope="global", *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Hint the device type, index and memory scope for executing ``data``."""
    attrs = _make_attrs(
        "relax.attrs.HintOnDeviceAttrs",
        device_type=device_type,
        index=index,
        memory_scope=memory_scope,
    )
    return Call("relax.hint_on_device", [data], attrs=attrs, ty=ty, loc=loc)
