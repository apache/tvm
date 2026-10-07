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
"""TIRx expression operators and intrinsic namespaces."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import tvm_ffi as _ffi

# isort: off
# isort: on
from tvm import ir
from tvm import ir as _ir
from tvm import tirx as _tir
from tvm import tirx as tir
from tvm.ir import Call
from tvm.ir import register_op_attr as _register_op_attr
from tvm.ir.prim import _ffi_api as _prim_ffi
from tvm.ir.prim import _ffi_api as _prim_ffi_api
from tvm.script.ir_builder import base as _base
from tvm.script.ir_builder.base import IRBuilder
from tvm.script.ir_builder.frame import IRModuleFrame

# pylint: disable=unused-import
from tvm.target.codegen import llvm_lookup_intrinsic_id
from tvm.tirx import Expr, is_tensor_var
from tvm.tirx import op as _tir_op
from tvm.tirx.exec_scope import Var

# import tirx.expr for direct ir construction to pass structural_equal comparison
from tvm.tirx.expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    Broadcast,
    CallEffectKind,
    Cast,
    CommReducer,
    Div,
    FloorDiv,
    FloorMod,
    LShift,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    Ramp,
    Reduce,
    RShift,
    Select,
    Shuffle,
    Sub,
)

from . import _ffi_api
from .external_kernel import call_kernel

# pylint: enable=unused-import


def _call_global(func: ir.GlobalVar, *args: Expr) -> Call:
    """Build a TIRX call using the declared function's exact result type."""
    if IRBuilder.is_in_scope():
        for module_frame in reversed(list(IRBuilder.current().frames)):
            if isinstance(module_frame, IRModuleFrame) and func in module_frame.functions:
                declaration = module_frame.functions[func]
                if isinstance(declaration, tir.PrimFunc):
                    # The Relax-facing signature may erase pointer results to Any.
                    return Call(func, args, ty=declaration.ret_type)
                break
    if isinstance(func.ty, ir.FuncType):
        return Call(func, args, ty=func.ty.ret_type)
    return Call(func, args)


def cast(value, dtype, span=None):
    """Cast an expression to the requested data type."""
    return _prim_ffi_api._cast(dtype, value, span)  # type: ignore[attr-defined]


def Let(  # pylint: disable=invalid-name
    expr: Expr,
    where: dict[Var, Expr],  # pylint: disable=redefined-outer-name
) -> Expr:
    """Create a Let expression binding"""
    assert len(where) == 1, "T.Let only allows `where` to have exactly one element"
    var, value = next(iter(where.items()))  # pylint: disable=redefined-outer-name
    return tir.Let(var, value, expr)


def min(a: Expr, b: Expr) -> Expr:  # pylint: disable=redefined-builtin
    """Compute the minimum value of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    Returns
    -------
    res : Expr
        The result expression.
    """
    return _ffi_api.min(a, b)  # type: ignore[attr-defined] # pylint: disable=no-member


def max(a: Expr, b: Expr) -> Expr:  # pylint: disable=redefined-builtin
    """Compute the maximum value of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    Returns
    -------
    res : Expr
        The result expression.
    """
    return _ffi_api.max(a, b)  # type: ignore[attr-defined] # pylint: disable=no-member


def comm_reducer(combiner: Callable, identity: list[Expr]) -> CommReducer:
    """
    Create a CommReducer from lambda inputs/outputs and the identities

    Parameters
    ----------
    combiner : Callable
        A binary function which takes two Expr as input to return a Expr.

    identity : List[Expr]
        A list of types of output Expr.

    Returns
    -------
    res : CommReducer
        The CommReducer.
    """
    params = inspect.signature(combiner).parameters
    num_args = len(params)
    args = []
    for name, i in zip(params.keys(), identity + identity):
        if isinstance(i, int):
            args.append(Var(name, "int32"))
        else:
            args.append(Var(name, i.ty))
    res = combiner(*args)
    if not isinstance(res, tuple):
        res = (res,)
    return CommReducer(args[: num_args // 2], args[num_args // 2 :], res, identity)


def _op_wrapper(func):
    """Retain the normal call contract while attaching namespace printer metadata."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    wrapped.__tir_op_name__ = getattr(func, "__name__", None)
    return wrapped


def _dtype_forward(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"), *args)
        return func(*args, **kwargs)

    # Expose underlying tir op name for printer registration
    try:
        wrapped.__tir_op_name__ = getattr(func, "__name__", None)
    except Exception:  # pragma: no cover
        pass
    return wrapped


class WebGPUNamespace:
    """The WebGPU intrinsics submodule."""

    @staticmethod
    def subgroup_shuffle(var, lane):
        if is_tensor_var(var):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.webgpu.subgroup_shuffle", var, lane)

    @staticmethod
    def subgroup_shuffle_up(var, delta):
        if is_tensor_var(var):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.webgpu.subgroup_shuffle_up", var, delta)

    @staticmethod
    def subgroup_shuffle_down(var, delta):
        if is_tensor_var(var):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.webgpu.subgroup_shuffle_down", var, delta)


webgpu = WebGPUNamespace()


def _register_script_namespace_printer_names(ns_obj, dotted_prefix, override=False):
    def register_printer_name(op_name, script_name):
        try:
            op = ir.Op.get(op_name)
        except AttributeError:
            return
        if op.has_attr("TScriptPrinterName") and op.get_attr("TScriptPrinterName") == script_name:
            return
        _register_op_attr(op_name, "TScriptPrinterName", script_name, override=override)

    def visit(ns_obj, dotted_prefix):
        # If the namespace object itself maps to an op via __call__
        call_op = getattr(ns_obj, "__tir_call_op_name__", None)
        if call_op:
            flat_name = f"tirx.{call_op}"
            for op_name in {flat_name, _tir_op._canonical_device_intrin_name(flat_name)}:
                register_printer_name(op_name, dotted_prefix)
        # Walk attributes to find wrapped ops and sub-namespaces
        for name in dir(ns_obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(ns_obj, name)
            except Exception:
                continue
            # Sub-namespace: recurse
            if hasattr(val, "__dict__") and val.__class__.__name__.endswith("Namespace"):
                visit(val, f"{dotted_prefix}.{name}")
                continue
            # Wrapped op (callable with attached __tir_op_name__)
            op_name = getattr(val, "__tir_op_name__", None)
            if callable(val) and op_name:
                flat_name = f"tirx.{op_name}"
                script_name = f"{dotted_prefix}.{name}"
                for full_op_name in {flat_name, _tir_op._canonical_device_intrin_name(flat_name)}:
                    register_printer_name(full_op_name, script_name)

    visit(ns_obj, dotted_prefix)


_SCRIPT_NAMESPACES = {}


def _get_script_namespace(name: str) -> object:
    """Return an explicitly registered backend construction namespace."""
    if name in _SCRIPT_NAMESPACES:
        return _SCRIPT_NAMESPACES[name]
    raise AttributeError(f"No script namespace {name!r}")


def register_script_namespace(
    name: str, namespace: object, override: bool = False, *, canonical_op_names: bool = False
) -> object:
    """Register a construction namespace and return it.

    Parameters
    ----------
    name : str
        Namespace name on the TIRx builder facade.
    namespace : object
        Construction namespace object.
    override : bool, optional
        Replace differing operator printer names if True. Existing equal names
        are reused; other duplicates raise ValueError.
    canonical_op_names : bool, optional
        Publish registered Op names whose canonical attributes expose matching
        callables. Preserve explicit printer aliases. Otherwise discover names
        from legacy wrappers. Repeat registration to publish newly exposed Ops.
    """
    _SCRIPT_NAMESPACES[name] = namespace
    globals()[name] = namespace
    if "__all__" in globals() and name not in __all__:
        __all__.append(name)

    import sys  # pylint: disable=import-outside-toplevel

    for module_name in [
        "tvm.tirx.script.ir_builder",
        "tvm.tirx.script",
        "tvm.script.tirx",
    ]:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        setattr(module, name, namespace)
        module_all = getattr(module, "__all__", None)
        if isinstance(module_all, list) and name not in module_all:
            module_all.append(name)

    if canonical_op_names:
        prefix = f"tirx.{name}."
        for op_name in _ir.Op.list_op_names():
            if not op_name.startswith(prefix):
                continue
            current = namespace
            for part in op_name[len(prefix) :].split("."):
                current = getattr(current, part, None)
            op = _ir.Op.get(op_name)
            identity = getattr(current, "__tvm_op__", None)
            if callable(current) and isinstance(identity, _ir.Op) and identity.same_as(op):
                op.set_attr(
                    "TScriptStandardCall",
                    bool(getattr(current, "__tvm_standard_call__", False)),
                    override=True,
                )
                if op.get_attr("TScriptPrinterName") is None:
                    op.set_attr("TScriptPrinterName", op_name)
    else:
        _register_script_namespace_printer_names(namespace, f"tirx.{name}", override)
    return namespace


def _register_tir_namespace_printer_names():
    try:
        _register_script_namespace_printer_names(webgpu, "tirx.webgpu")
    except Exception:
        # Best-effort registration; avoid import-time hard failure
        pass


_register_tir_namespace_printer_names()


abs = _tir_op.abs  # pylint: disable=redefined-builtin


acos = _tir_op.acos


acosh = _tir_op.acosh


address_of = _tir_op.address_of


asin = _tir_op.asin


asinh = _tir_op.asinh


atan = _tir_op.atan


atan2 = _tir_op.atan2


atanh = _tir_op.atanh


bitwise_and = _tir_op.bitwise_and


bitwise_not = _tir_op.bitwise_not


bitwise_or = _tir_op.bitwise_or


bitwise_xor = _tir_op.bitwise_xor


ceil = _tir_op.ceil


clz = _tir_op.clz


copysign = _tir_op.copysign


cos = _tir_op.cos


cosh = _tir_op.cosh


erf = _tir_op.erf


exp = _tir_op.exp


exp2 = _tir_op.exp2


exp10 = _tir_op.exp10


filter = _tir_op.filter  # pylint: disable=redefined-builtin


selector = _tir_op.selector


floor = _tir_op.floor


ceildiv = _tir_op.ceildiv


floordiv = _tir_op.floordiv


floormod = _tir_op.floormod


fmod = _tir_op.fmod


fma = _tir_op.fma


hypot = _tir_op.hypot


if_then_else = _tir_op.if_then_else


infinity = _tir_op.infinity


isfinite = _tir_op.isfinite


isinf = _tir_op.isinf


isnan = _tir_op.isnan


isnullptr = _tir_op.isnullptr


ldexp = _tir_op.ldexp


likely = _tir_op.likely


log = _tir_op.log


log1p = _tir_op.log1p


log2 = _tir_op.log2


log10 = _tir_op.log10


max_value = _tir_op.max_value


min_value = _tir_op.min_value


nearbyint = _tir_op.nearbyint


nextafter = _tir_op.nextafter


popcount = _tir_op.popcount


pow = _tir_op.pow  # pylint: disable=redefined-builtin


q_multiply_shift = _tir_op.q_multiply_shift


q_multiply_shift_per_axis = _tir_op.q_multiply_shift_per_axis


round = _tir_op.round  # pylint: disable=redefined-builtin


rsqrt = _tir_op.rsqrt


shift_left = _tir_op.shift_left


shift_right = _tir_op.shift_right


sigmoid = _tir_op.sigmoid


sin = _tir_op.sin


sinh = _tir_op.sinh


sqrt = _tir_op.sqrt


tan = _tir_op.tan


tanh = _tir_op.tanh


thread_return = _tir_op.thread_return


trunc = _tir_op.trunc


truncdiv = _tir_op.truncdiv


truncmod = _tir_op.truncmod


tvm_access_ptr = _tir_op.tvm_access_ptr


ptr_byte_offset = _tir_op.ptr_byte_offset


tvm_throw_last_error = _tir_op.tvm_throw_last_error


print_buffer = _tir_op.print_buffer


tvm_stack_alloca = _tir_op.tvm_stack_alloca


tvm_stack_make_shape = _tir_op.tvm_stack_make_shape


tvm_stack_make_array = _tir_op.tvm_stack_make_array


call_packed = _tir_op.call_packed


call_ffi_kernel = _tir_op.call_ffi_kernel


tensormap_encode_tiled = _tir_op.tensormap_encode_tiled


call_cpacked = _tir_op.call_cpacked


call_packed_lowered = _tir_op.call_packed_lowered


call_cpacked_lowered = _tir_op.call_cpacked_lowered


handle_add_byte_offset = _tir_op.handle_add_byte_offset


tvm_struct_set = _tir_op.tvm_struct_set


tvm_struct_get = _tir_op.tvm_struct_get


tvm_thread_invariant = _tir_op.tvm_thread_invariant


tvm_thread_allreduce = _tir_op.tvm_thread_allreduce


tvm_load_matrix_sync = _tir_op.tvm_load_matrix_sync


tvm_mma_sync = _tir_op.tvm_mma_sync


tvm_bmma_sync = _tir_op.tvm_bmma_sync


tvm_fill_fragment = _tir_op.tvm_fill_fragment


tvm_store_matrix_sync = _tir_op.tvm_store_matrix_sync


tvm_storage_sync = _tir_op.tvm_storage_sync
cpu_parallel_barrier = _tir_op.cpu_parallel_barrier


tvm_kernel_replace_point = _tir_op.tvm_kernel_replace_point


tvm_warp_shuffle = _tir_op.tvm_warp_shuffle


tvm_warp_shuffle_up = _tir_op.tvm_warp_shuffle_up


tvm_warp_shuffle_down = _tir_op.tvm_warp_shuffle_down


tvm_warp_shuffle_xor = _tir_op.tvm_warp_shuffle_xor


tvm_warp_activemask = _tir_op.tvm_warp_activemask


cooperative_tensor_fill = _tir_op.cooperative_tensor_fill


cooperative_tensor_load = _tir_op.cooperative_tensor_load


cooperative_tensor_store = _tir_op.cooperative_tensor_store


cooperative_tensor_multiply_accumulate = _tir_op.cooperative_tensor_multiply_accumulate


assume = _tir_op.assume
assume_aligned = _tir_op.assume_aligned


undef = _tir_op.undef


TVMBackendAllocWorkspace = _tir_op.TVMBackendAllocWorkspace


TVMBackendFreeWorkspace = _tir_op.TVMBackendFreeWorkspace


vscale = _tir_op.vscale


ignore_loop_partition = _tir_op.ignore_loop_partition


reinterpret = _dtype_forward(_tir_op.reinterpret)


call_extern = _dtype_forward(_tir_op.call_extern)


call_intrin = _dtype_forward(_tir_op.call_intrin)


call_llvm_intrin = _dtype_forward(_tir_op.call_llvm_intrin)


call_llvm_pure_intrin = _dtype_forward(_tir_op.call_llvm_pure_intrin)


call_pure_extern = _dtype_forward(_tir_op.call_pure_extern)


vectorlow = _dtype_forward(_tir_op.vectorlow)


vectorhigh = _dtype_forward(_tir_op.vectorhigh)


vectorcombine = _dtype_forward(_tir_op.vectorcombine)


get_active_lane_mask = _dtype_forward(_tir_op.get_active_lane_mask)


masked_load = _dtype_forward(_tir_op.masked_load)


masked_store = _tir_op.masked_store


dp4a = _tir_op.dp4a


broadcast = Broadcast


ramp = Ramp


fabs = abs


tvm_call_packed = call_packed


tvm_call_cpacked = call_cpacked


tvm_call_packed_lowered = call_packed_lowered


tvm_call_cpacked_lowered = call_cpacked_lowered


def _as_expr(value):
    if isinstance(value, _ffi.ObjectConvertible):
        value = value.asobject()
    if isinstance(value, _ir.Expr):
        return value
    if isinstance(value, str):
        return _ir.StringImm(value)
    if isinstance(value, list | tuple):
        return _ir.Tuple([_as_expr(item) for item in value])
    return _tir.const(value)


def logical_and(*values):
    """Construct scalar or vector conjunction from eager operands.

    Parameters
    ----------
    values : Expr or Python value
        One or more operands. Object-convertible values are normalized first.
        Host pairs follow Python logical operations; IR pairs use scalar logical
        or vector bitwise operations according to their types.

    Returns
    -------
    result : Expr or Python value
        The conjunction reduced from left to right.

    Notes
    -----
    All arguments are evaluated before this call; it does not provide Python
    short-circuit evaluation of the argument expressions.
    """
    if not values:
        raise TypeError("logical_and requires at least one operand")
    values = [
        value.asobject() if isinstance(value, _ffi.ObjectConvertible) else value for value in values
    ]
    result = values[0]
    for value in values[1:]:
        if not isinstance(result, _ir.Expr) and not isinstance(value, _ir.Expr):
            result = result and value
        else:
            lhs, rhs = _as_expr(result), _as_expr(value)
            result = _tir.And(lhs, rhs) if lhs.ty.is_scalar() and rhs.ty.is_scalar() else lhs & rhs
    return result


def logical_or(*values):
    """Construct scalar or vector disjunction from eager operands.

    Parameters
    ----------
    values : Expr or Python value
        One or more operands. Object-convertible values are normalized first.
        Host pairs follow Python logical operations; IR pairs use scalar logical
        or vector bitwise operations according to their types.

    Returns
    -------
    result : Expr or Python value
        The disjunction reduced from left to right.

    Notes
    -----
    All arguments are evaluated before this call; it does not provide Python
    short-circuit evaluation of the argument expressions.
    """
    if not values:
        raise TypeError("logical_or requires at least one operand")
    values = [
        value.asobject() if isinstance(value, _ffi.ObjectConvertible) else value for value in values
    ]
    result = values[0]
    for value in values[1:]:
        if not isinstance(result, _ir.Expr) and not isinstance(value, _ir.Expr):
            result = result or value
        else:
            lhs, rhs = _as_expr(result), _as_expr(value)
            result = _tir.Or(lhs, rhs) if lhs.ty.is_scalar() and rhs.ty.is_scalar() else lhs | rhs
    return result


def logical_not(value):
    """Negate a host or IR value.

    Parameters
    ----------
    value : Expr or Python value
        Operand to negate. Object-convertible values are normalized first;
        IR expressions use primitive Not and host values use Python not.

    Returns
    -------
    result : Expr or bool
        The logical negation without testing an IR expression as a Python bool.
    """
    if isinstance(value, _ffi.ObjectConvertible):
        value = value.asobject()
    return _tir.Not(value) if isinstance(value, _ir.Expr) else not value


def select(condition, true_value, false_value):
    """Construct a scalar conditional or select a host value.

    Parameters
    ----------
    condition : Expr or Python value
        Scalar IR condition or host truth value. Object-convertible conditions
        are normalized before selection.
    true_value : Expr or Python value
        Value selected when the condition is true.
    false_value : Expr or Python value
        Value selected when the condition is false.

    Returns
    -------
    result : Expr or Python value
        A primitive if_then_else expression, or the selected host object.

    Notes
    -----
    Both Python value arguments are constructed before this call. The generated
    IR conditional evaluates only its selected arm at runtime.
    """
    if isinstance(condition, _ffi.ObjectConvertible):
        condition = condition.asobject()
    if not isinstance(condition, _ir.Expr):
        return true_value if condition else false_value
    return _tir.if_then_else(condition, true_value, false_value)


__all__ = [
    "EQ",
    "GE",
    "GT",
    "LE",
    "LT",
    "NE",
    "Add",
    "And",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "Broadcast",
    "Call",
    "CallEffectKind",
    "Cast",
    "CommReducer",
    "Div",
    "FloorDiv",
    "FloorMod",
    "LShift",
    "Let",
    "Max",
    "Min",
    "Mod",
    "Mul",
    "Not",
    "Or",
    "RShift",
    "Ramp",
    "Reduce",
    "Select",
    "Shuffle",
    "Sub",
    "TVMBackendAllocWorkspace",
    "TVMBackendFreeWorkspace",
    "abs",
    "acos",
    "acosh",
    "address_of",
    "asin",
    "asinh",
    "assume",
    "assume_aligned",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast",
    "call_cpacked",
    "call_cpacked_lowered",
    "call_extern",
    "call_ffi_kernel",
    "call_intrin",
    "call_kernel",
    "call_llvm_intrin",
    "call_llvm_pure_intrin",
    "call_packed",
    "call_packed_lowered",
    "call_pure_extern",
    "cast",
    "ceil",
    "ceildiv",
    "clz",
    "comm_reducer",
    "cooperative_tensor_fill",
    "cooperative_tensor_load",
    "cooperative_tensor_multiply_accumulate",
    "cooperative_tensor_store",
    "copysign",
    "cos",
    "cosh",
    "cpu_parallel_barrier",
    "dp4a",
    "erf",
    "exp",
    "exp2",
    "exp10",
    "fabs",
    "filter",
    "floor",
    "floordiv",
    "floormod",
    "fma",
    "fmod",
    "get_active_lane_mask",
    "handle_add_byte_offset",
    "hypot",
    "if_then_else",
    "ignore_loop_partition",
    "infinity",
    "isfinite",
    "isinf",
    "isnan",
    "isnullptr",
    "ldexp",
    "likely",
    "llvm_lookup_intrinsic_id",
    "log",
    "log1p",
    "log2",
    "log10",
    "logical_and",
    "logical_not",
    "logical_or",
    "masked_load",
    "masked_store",
    "max",
    "max_value",
    "min",
    "min_value",
    "nearbyint",
    "nextafter",
    "popcount",
    "pow",
    "print_buffer",
    "ptr_byte_offset",
    "q_multiply_shift",
    "q_multiply_shift_per_axis",
    "ramp",
    "register_script_namespace",
    "reinterpret",
    "round",
    "rsqrt",
    "select",
    "selector",
    "shift_left",
    "shift_right",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "tensormap_encode_tiled",
    "thread_return",
    "trunc",
    "truncdiv",
    "truncmod",
    "tvm_access_ptr",
    "tvm_bmma_sync",
    "tvm_call_cpacked",
    "tvm_call_cpacked_lowered",
    "tvm_call_packed",
    "tvm_call_packed_lowered",
    "tvm_fill_fragment",
    "tvm_kernel_replace_point",
    "tvm_load_matrix_sync",
    "tvm_mma_sync",
    "tvm_stack_alloca",
    "tvm_stack_make_array",
    "tvm_stack_make_shape",
    "tvm_storage_sync",
    "tvm_store_matrix_sync",
    "tvm_struct_get",
    "tvm_struct_set",
    "tvm_thread_allreduce",
    "tvm_thread_invariant",
    "tvm_throw_last_error",
    "tvm_warp_activemask",
    "tvm_warp_shuffle",
    "tvm_warp_shuffle_down",
    "tvm_warp_shuffle_up",
    "tvm_warp_shuffle_xor",
    "undef",
    "vectorcombine",
    "vectorhigh",
    "vectorlow",
    "vscale",
    "webgpu",
]

_Span = _base.SpanEntry | _ir.Span | None


def if_then_else_(condition: Any, true_value: Any, false_value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.if_then_else_`."""
    return select(condition, true_value, false_value)


def and_(*values: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.and_`."""
    return logical_and(*values)


def or_(*values: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.or_`."""
    return logical_or(*values)


def not_(value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.not_`."""
    return logical_not(value)


def lt_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.lt_`."""
    return _prim_ffi._OpLT(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def le_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.le_`."""
    return _prim_ffi._OpLE(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def gt_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.gt_`."""
    return _prim_ffi._OpGT(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def ge_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ge_`."""
    return _prim_ffi._OpGE(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def eq_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.eq_`."""
    return _prim_ffi._OpEQ(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def ne_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ne_`."""
    return _prim_ffi._OpNE(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


__all__ += ["and_", "eq_", "ge_", "gt_", "if_then_else_", "le_", "lt_", "ne_", "not_", "or_"]
