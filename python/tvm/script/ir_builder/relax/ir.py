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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name
"""IRBuilder for Relax dialect"""

import builtins
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import tvm
from tvm import DataType, relax
from tvm.ir import PrimExpr, VDevice, IRModule
from tvm.relax import (
    Call,
    Expr,
    ExternFunc,
    ShapeExpr,
    TupleGetItem,
    Var,
    VarBinding,
    const,
)
from tvm.relax.dpl import PatternMatchingRewriter

############################### Operators ###############################
from tvm.relax.op import (
    abs,
    acos,
    acosh,
    add,
    arange,
    argmax,
    argmin,
    argsort,
    asin,
    asinh,
    assert_op,
    astype,
    atan,
    atanh,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    broadcast_to,
    builtin,
    call_builtin_with_ctx,
    call_dps_packed,
    call_inplace_packed,
    call_pure_packed,
    call_tir,
    call_tir_inplace,
    call_tir_with_grad,
    ccl,
    ceil,
    clip,
    collapse_sum_like,
    collapse_sum_to,
    concat,
    cos,
    cosh,
    cumprod,
    cumsum,
    dequantize,
    divide,
    dynamic_strided_slice,
    einsum,
    equal,
    erf,
    ewise_fma,
    exp,
    expand_dims,
    eye,
    eye_like,
    flatten,
    flip,
    floor,
    floor_divide,
    floor_mod,
    full,
    full_like,
    gather_elements,
    gather_nd,
    grad,
    greater,
    greater_equal,
    hint_on_device,
    image,
    invoke_closure,
    invoke_pure_closure,
    isfinite,
    isinf,
    isnan,
    layout_transform,
    left_shift,
    less,
    less_equal,
    linear,
    log,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    make_closure,
    matmul,
    max,
    maximum,
    mean,
    memory,
    min,
    minimum,
    mod,
    multinomial_from_uniform,
    multiply,
    negative,
    nn,
    not_equal,
    null_value,
    ones,
    ones_like,
    one_hot,
    permute_dims,
    power,
    print,
    prod,
    quantize,
    repeat,
    reshape,
    right_shift,
    round,
    rsqrt,
    scatter_elements,
    scatter_nd,
    shape_of,
    shape_to_tensor,
    sigmoid,
    sign,
    sin,
    sinh,
    sort,
    split,
    sqrt,
    square,
    squeeze,
    std,
    strided_slice,
    subtract,
    sum,
    take,
    tan,
    tanh,
    tensor_to_shape,
    tile,
    topk,
    tril,
    triu,
    unique,
    variance,
    vm,
    where,
    wrap_param,
    zeros,
    zeros_like,
)
from tvm.relax.op.builtin import stop_lift_params
from tvm.relax.struct_info import StructInfo
from tvm.relax.utils import args_converter, gen_call_tir_inputs
from tvm.runtime import Object as tvm_Object
from tvm.runtime import ObjectGeneric
from tvm.runtime.ndarray import (
    cpu,
    cuda,
    device,
    ext_dev,
    gpu,
    hexagon,
    metal,
    opencl,
    rocm,
    vpi,
    vulkan,
    webgpu,
)

from ..ir import decl_function, lookup_vdevice
from . import _ffi_api, frame

##################### Python Native Function Alias ######################

py_print = builtins.print
py_tuple = tuple  # pylint: disable=used-before-assignment
py_str = str  # pylint: disable=used-before-assignment


################################ Device ################################


def to_vdevice(data: Expr, dst_vdevice: Union[py_str, VDevice]) -> Expr:
    """Copy data to the destination device.

    Parameters
    ----------
    data : Expr
        The tensor to be copied.

    dst_device : Union[py_str, VDevice]
        The destination device where the data is copied to.

    Returns
    -------
    result : Expr
        The copied result.
    """
    if isinstance(dst_vdevice, py_str):
        if ":" in dst_vdevice:
            split_vdev = dst_vdevice.split(":")
            dst_vdevice = lookup_vdevice(split_vdev[0], int(split_vdev[1]))
        else:
            dst_vdevice = lookup_vdevice(dst_vdevice, 0)

    return tvm.relax.op.to_vdevice(data, dst_vdevice)


############################### Function ################################


def function(is_pure: bool = True, is_private: bool = False) -> frame.FunctionFrame:
    """Start a function frame.
    Parameters
    ----------
    is_pure: bool
        Whether the function is annotated as pure.

    is_private : bool
        Whether the function is annotated as private.

    Returns
    -------
    frame: FunctionFrame
        The constructed function frame.
    """
    return _ffi_api.Function(  # type: ignore[attr-defined]  # pylint: disable=no-member
        is_pure, is_private
    )


def arg(name: py_str, struct_info: StructInfo) -> Var:
    """Add a parameter to the last function frame.
    Parameters
    ----------
    name: str
        The name of the parameter.
    struct_info: StructInfo
        The Struct Info of the parameter

    Returns
    -------
    var: Var
        The created function parameter var.
    """

    return _ffi_api.Arg(name, struct_info)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_name(name: py_str) -> None:
    """Specify the name of the last function frame.
    Parameters
    ----------
    name: str
        The function name.
    """
    return _ffi_api.FuncName(name)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_attr(attrs: Dict[py_str, tvm_Object]) -> None:
    """Specify the attrs of the last function frame.
    Parameters
    ----------
    attrs: Dict[str, Object]
        The function attrs.
    """
    return _ffi_api.FuncAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_ret_struct_info(ret_sinfo: StructInfo) -> None:
    """Specify the return struct info of the last function frame.
    Parameters
    ----------
    ret_type: StructInfo
        The function return struct info.
    """
    return _ffi_api.FuncRetStructInfo(ret_sinfo)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_ret_value(value: Expr) -> None:
    """Specify the return value of the last function frame.
    Parameters
    ----------
    value: Expr
        The function return value.
    """
    return _ffi_api.FuncRetValue(value)  # type: ignore[attr-defined] # pylint: disable=no-member


def rewriter(rewriter_mod: Union[IRModule, Type]) -> PatternMatchingRewriter:
    """Define a pattern-rewrite rule

    The IRModule must have two publicly-exposed functions, `pattern`
    and `replacement`, where `pattern` and `replacement` have the same
    function signature.

    .. code-block:: python

        @R.rewriter
        class RewriteAddIntoMultiply:
            @R.function
            def pattern(A: R.Tensor):
                B = A + A
                return B

            @R.function
            def replacement(A: R.Tensor):
                B = A * 2
                return B

    Parameters
    ----------
    rewriter_mod: Union[IRModule, Type]

        Either an IRModule that defines a rewrite pattern, or a
        TVMScript class that can be parsed into an IRModule.

    Returns
    -------
    rewriter: PatternMatchingRewriter

        A rewriter object, which can be applied either to a Relax
        function or to an entire IRModule.

    """
    if not isinstance(rewriter_mod, IRModule):
        rewriter_mod = tvm.script.ir_module(rewriter_mod)

    return PatternMatchingRewriter.from_module(rewriter_mod)


############################# BindingBlock ##############################


def dataflow() -> frame.BlockFrame:
    """Start a dataflow binding block frame.
    Returns
    -------
    frame: frame.BlockFrame
        The created ir_builder Block frame.
    """
    return _ffi_api.Dataflow()  # type: ignore[attr-defined] # pylint: disable=no-member


def output(*vars: Tuple[Var]) -> None:
    """Expose the dataflow block output variables as global ones.
    Parameters
    ----------
    vars: Tuple[Var]
        The output variables of a dataflow block.
    """
    return _ffi_api.DataflowBlockOutput(vars)  # type: ignore[attr-defined] # pylint: disable=no-member


################################## Ops #################################


@args_converter.auto
def call_packed(
    func: py_str,
    *args: Expr,
    sinfo_args: Optional[Union[StructInfo, List[StructInfo]]] = None,
    **kwargs: Any,
) -> Call:
    """Create a relax Call, which calls a packed function.
    Parameters
    ----------
    func: str
        The name of extern function.
    *args : Expr
        The arguments.
    sinfo_args: Optional[Union[StructInfo, List[StructInfo]]]
        The list of structure info arguments.
    kwargs: Expr
        The keyword arguments.

    Returns
    -------
    call: Call
        The created Relax Call
    """
    op = ExternFunc(func)
    if sinfo_args is None:
        sinfo_args = []
    if isinstance(sinfo_args, py_tuple):  # type: ignore
        sinfo_args = list(sinfo_args)
    elif not isinstance(sinfo_args, list):
        sinfo_args = [sinfo_args]

    sinfo_args = [
        sinfo()
        if callable(sinfo)
        else sinfo.asobject()
        if isinstance(sinfo, ObjectGeneric)
        else sinfo
        for sinfo in sinfo_args
    ]

    is_default = False
    if "attrs_type_key" in kwargs:
        attrs_type_key = kwargs["attrs_type_key"]
        kwargs.pop("attrs_type_key")
    else:
        attrs_type_key = "DictAttrs"
        is_default = True
    attrs = None
    if kwargs or not is_default:
        attrs = tvm.ir.attrs.make_node(attrs_type_key, **kwargs)

    return Call(op, args, attrs=attrs, sinfo_args=sinfo_args)


def _sinfo_arg_wrapper(func):
    """A wrapper to convert StructInfoProxies to StructInfo for builtin operators with sinfo_args"""

    def _convert_tensor_type(args):
        if isinstance(args, (list, py_tuple)):  # type: ignore
            new_args = [_convert_tensor_type(x) for x in args]
            return type(args)(new_args)
        if isinstance(args, dict):
            return {_convert_tensor_type(k): _convert_tensor_type(v) for k, v in args.items()}
        if inspect.isfunction(args):
            args = args()
        if isinstance(args, ObjectGeneric):
            args = args.asobject()
        return args

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*_convert_tensor_type(args), **_convert_tensor_type(kwargs))

    return wrapped  # type: ignore


invoke_closure = _sinfo_arg_wrapper(invoke_closure)  # pylint: disable=invalid-name

call_builtin_with_ctx = _sinfo_arg_wrapper(call_builtin_with_ctx)  # pylint: disable=invalid-name


############################### Emits ###############################


def emit(value: Expr, annotate_struct_info: Optional[StructInfo] = None) -> Var:
    """Emit a binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The right side value of the bindings to be emitted.

    annotate_struct_info: Optional[StructInfo]
        The optional struct info annotation for the emitted value.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.Emit(value, annotate_struct_info)  # type: ignore[attr-defined] # pylint: disable=no-member


def emit_te(func: Callable, *args: Any, **kwargs: Any) -> Call:
    """Emit a call node according to the te function.
    This function converts arguments from relax expression to te tensor,
    The callback func should return a te tensor or a list of te tensors.

    Parameters
    ----------
    func : Callable
        A function that returns a te tensor or a list of te tensors.

    args : Any, optional
        arguments passed to the function.

    kwargs : Any, optional
        The keyword arguments passed to the function.
        Note that the following keyword args are reserved:

            - 'primfunc_name_hint' for passing name hint to the PrimFunc
                that gets generated.
            - 'primfunc_attrs' is reserved for passing func attributes to
                be added to the PrimFunc that gets created.

    Returns
    -------
    call : Call
        A newly created call that calls into a tir function.
    """
    primfunc_name_hint = kwargs.pop("primfunc_name_hint", None)
    tir_func, call_args, out_sinfo, tir_vars = gen_call_tir_inputs(func, *args, **kwargs)
    if not primfunc_name_hint:
        primfunc_name_hint = func.__name__
    gvar = decl_function(primfunc_name_hint, tir_func)  # type: ignore
    return call_tir(gvar, call_args, out_sinfo, tir_vars)


def emit_match_cast(value: Expr, struct_info: StructInfo) -> Var:
    """Emit a match_cast binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The value of the MatchCast to be emitted.
    struct_info: StructInfo
        The struct_info of the MatchCast to be emitted.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.EmitMatchCast(value, struct_info)  # type: ignore


def emit_var_binding(value: VarBinding) -> Var:
    """Emit a binding to the last binding block frame.
    Parameters
    ----------
    value: VarBinding
        The binding to be emitted.
    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.EmitVarBinding(value)  # type: ignore


############################### SeqExpr ###############################


def SeqExpr() -> frame.SeqExprFrame:  # pylint: disable=invalid-name
    """Create a SeqExpr frame.
    Returns
    -------
    res : frame.SeqExprFrame
        The result SeqExprFrame
    """
    return _ffi_api.SeqExpr()  # type: ignore[attr-defined] # pylint: disable=no-member


############################# If Then Else #############################


def If(condition: Union[Expr, PrimExpr]) -> frame.IfFrame:  # pylint: disable=invalid-name
    """Create an if frame.

    Parameters
    ----------
    condition : Union[Expr, PrimExpr]

        The condition of if statement, executes the true branch if the
        condition is true, otherwise jump into the false branch.

    Returns
    -------
    res : frame.IfFrame
        The result IfFrame.

    """
    if not isinstance(condition, Expr):
        condition = relax.PrimValue(condition)

    return _ffi_api.If(condition)  # type: ignore[attr-defined] # pylint: disable=no-member


def Then() -> frame.ThenFrame:  # pylint: disable=invalid-name
    """Create a then frame.
    Returns
    -------
    res : frame.ThenFrame
        The result ThenFrame.
    """
    return _ffi_api.Then()  # type: ignore[attr-defined] # pylint: disable=no-member


def Else() -> frame.ElseFrame:  # pylint: disable=invalid-name
    """Create an else frame.
    Returns
    -------
    res : frame.ElseFrame
        The result ElseFrame.
    """
    return _ffi_api.Else()  # type: ignore[attr-defined] # pylint: disable=no-member


############################### R.tuple ################################


def tuple(*fields: Expr) -> Expr:
    """Create a tuple expression.
    Parameters
    ----------
    *fields : Expr
        The fields of the tuple.
    Returns
    -------
    res : Expr
        The result tuple.
    """
    if len(fields) == 0:
        fields = py_tuple()

    return relax.Tuple(fields)  # type: ignore[attr-defined] # pylint: disable=no-member


############################### R.shape ################################


def shape(value: List[PrimExpr]) -> Expr:
    """Create a ShapeExpr.
    Parameters
    ----------
    value : List[PrimExpr]
        The fields of the tuple.
    Returns
    -------
    res : Expr
        The result tuple.
    """
    return relax.ShapeExpr(value)  # pylint: disable=no-member # type: ignore


############################### PrimValue ##############################


def prim_value(value: PrimExpr) -> Expr:
    """Create a prim value expression.
    Parameters
    ----------
    value : PrimExpr
        The value of the prim value.
    Returns
    -------
    res : Expr
        The result prim value.
    """
    return relax.PrimValue(value)  # type: ignore[attr-defined] # pylint: disable=no-member


def str(value: py_str) -> Expr:
    """Create a string imm expression.
    Parameters
    ----------
    value : str
        The value of the str.
    Returns
    -------
    res : Expr
        The result str.
    """
    return relax.StringImm(value)  # type: ignore[attr-defined] # pylint: disable=no-member


def dtype(value: Union[py_str, DataType]) -> Expr:
    """Create a dtype imm expression.
    Parameters
    ----------
    value : dtype
        The value of the dtype.
    Returns
    -------
    res : Expr
        The result dtype.
    """
    return relax.DataTypeImm(value)  # type: ignore[attr-defined] # pylint: disable=no-member


############################### Importer ###############################

__all__ = [
    "Else",
    "If",
    "SeqExpr",
    "Then",
    "TupleGetItem",
    "ExternFunc",
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "add",
    "arange",
    "arg",
    "argmax",
    "argmin",
    "argsort",
    "assert_op",
    "astype",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_to",
    "builtin",
    "call_inplace_packed",
    "call_packed",
    "call_pure_packed",
    "call_tir",
    "call_tir_inplace",
    "call_tir_with_grad",
    "call_dps_packed",
    "call_builtin_with_ctx",
    "ceil",
    "clip",
    "collapse_sum_like",
    "collapse_sum_to",
    "concat",
    "cos",
    "cosh",
    "const",
    "cpu",
    "cuda",
    "cumprod",
    "cumsum",
    "einsum",
    "scatter_elements",
    "scatter_nd",
    "dataflow",
    "device",
    "divide",
    "dtype",
    "emit",
    "emit_te",
    "emit_var_binding",
    "emit_match_cast",
    "equal",
    "ewise_fma",
    "exp",
    "expand_dims",
    "ext_dev",
    "eye",
    "eye_like",
    "flatten",
    "flip",
    "floor",
    "floor_divide",
    "floor_mod",
    "full",
    "full_like",
    "func_attr",
    "func_name",
    "func_ret_struct_info",
    "func_ret_value",
    "function",
    "gather_elements",
    "gather_nd",
    "gpu",
    "grad",
    "greater",
    "greater_equal",
    "hexagon",
    "hint_on_device",
    "image",
    "invoke_closure",
    "invoke_pure_closure",
    "isfinite",
    "isinf",
    "isnan",
    "layout_transform",
    "left_shift",
    "less",
    "less_equal",
    "linear",
    "log",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "make_closure",
    "matmul",
    "max",
    "maximum",
    "mean",
    "memory",
    "metal",
    "min",
    "minimum",
    "mod",
    "multinomial_from_uniform",
    "multiply",
    "negative",
    "not_equal",
    "null_value",
    "ones",
    "ones_like",
    "one_hot",
    "opencl",
    "output",
    "permute_dims",
    "power",
    "prim_value",
    "print",
    "prod",
    "quantize",
    "dequantize",
    "repeat",
    "reshape",
    "rewriter",
    "right_shift",
    "tensor_to_shape",
    "shape_to_tensor",
    "rocm",
    "round",
    "rsqrt",
    "shape",
    "shape_of",
    "ShapeExpr",
    "std",
    "str",
    "sum",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sort",
    "split",
    "square",
    "squeeze",
    "sqrt",
    "stop_lift_params",
    "str",
    "strided_slice",
    "dynamic_strided_slice",
    "subtract",
    "take",
    "tan",
    "tanh",
    "tile",
    "topk",
    "to_vdevice",
    "tril",
    "triu",
    "tuple",
    "unique",
    "variance",
    "vm",
    "vpi",
    "vulkan",
    "webgpu",
    "where",
    "wrap_param",
    "zeros",
    "zeros_like",
    "nn",
    "ccl",
    "erf",
]
