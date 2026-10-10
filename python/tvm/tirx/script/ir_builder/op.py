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
from typing import Any

import tvm_ffi as _ffi

# isort: off
# isort: on
from tvm import ir
from tvm import ir as _ir
from tvm import tirx as _tir
from tvm import tirx as tir
from tvm.ir import Call, Var
from tvm.ir.prim import _ffi_api as _prim_ffi
from tvm.ir.prim import _ffi_api as _prim_ffi_api
from tvm.script.ir_builder import base as _base
from tvm.script.ir_builder.base import IRBuilder
from tvm.script.ir_builder.frame import IRModuleFrame
from tvm.script.ir_builder.stmt import _as_expr as _as_expr

# pylint: disable=unused-import
from tvm.target.codegen import llvm_lookup_intrinsic_id
from tvm.tirx import Expr, is_tensor_var
from tvm.tirx import op as _tir_op

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
    RShift,
    Select,
    Shuffle,
    Sub,
)
from tvm.tirx.op import CallFFIKernelAttr

from . import _ffi_api
from .external_kernel import call_kernel

# pylint: enable=unused-import


def _call_global(func: ir.GlobalVar, *args: Expr) -> Call:
    """Build a TIRX call using the declared function's exact result type."""
    if IRBuilder.is_in_scope():
        for module_frame in reversed(list(IRBuilder.current().frames)):
            if isinstance(module_frame, IRModuleFrame) and func in module_frame.functions:
                declaration = module_frame.functions[func]
                if isinstance(declaration, tir.Function):
                    # The Relax-facing signature may erase pointer results to Any.
                    return Call(func, args, ty=declaration.ret_type)
                break
    if isinstance(func.ty, ir.FuncType):
        return Call(func, args, ty=func.ty.ret_type)
    return Call(func, args)


def cast(value, dtype, loc=None):
    """Cast an expression to the requested data type."""
    return _prim_ffi_api._cast(dtype, value, loc)  # type: ignore[attr-defined]


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


def _llvm_result_type(func):
    """Keep symbolic intrinsic-name conversion when spelling the result as ty."""

    @functools.wraps(func)
    def wrapped(name, *args, ty, loc=None):
        return func(ty, name, *args, loc=loc)

    return wrapped


class WebGPUNamespace:
    """The WebGPU intrinsics submodule."""

    @staticmethod
    def subgroup_shuffle(var, lane, *, ty=None, loc=None):
        if is_tensor_var(var):
            var = var[0]
        return Call(
            "tirx.webgpu.subgroup_shuffle",
            [var, lane],
            ty=ty,
            loc=loc,
        )

    @staticmethod
    def subgroup_shuffle_up(var, delta, *, ty=None, loc=None):
        if is_tensor_var(var):
            var = var[0]
        return Call(
            "tirx.webgpu.subgroup_shuffle_up",
            [var, delta],
            ty=ty,
            loc=loc,
        )

    @staticmethod
    def subgroup_shuffle_down(var, delta, *, ty=None, loc=None):
        if is_tensor_var(var):
            var = var[0]
        return Call(
            "tirx.webgpu.subgroup_shuffle_down",
            [var, delta],
            ty=ty,
            loc=loc,
        )


webgpu = WebGPUNamespace()


_SCRIPT_NAMESPACES = {}


def _get_script_namespace(name: str) -> object:
    """Return an explicitly registered backend construction namespace."""
    if name in _SCRIPT_NAMESPACES:
        return _SCRIPT_NAMESPACES[name]
    raise AttributeError(f"No script namespace {name!r}")


def register_script_namespace(name: str, namespace: object) -> object:
    """Register a construction namespace and return it.

    Parameters
    ----------
    name : str
        Namespace name on the TIRx builder facade.
    namespace : object
        Construction namespace object.
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

    return namespace


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


ceil = _ir.op._make_op_api(_ir.Op.get("prim.ceil"), __name__)


clz = _ir.op._make_op_api(_ir.Op.get("prim.clz"), __name__)


copysign = _tir_op.copysign


cos = _tir_op.cos


cosh = _tir_op.cosh


erf = _tir_op.erf


exp = _tir_op.exp


exp2 = _tir_op.exp2


exp10 = _tir_op.exp10


gpu_thread_filter = _tir_op.gpu_thread_filter


gpu_active_thread_selector = _tir_op.gpu_active_thread_selector


floor = _ir.op._make_op_api(_ir.Op.get("prim.floor"), __name__)


ceildiv = _tir_op.ceildiv


floordiv = _tir_op.floordiv


floormod = _tir_op.floormod


fmod = _tir_op.fmod


fma = _tir_op.fma


hypot = _tir_op.hypot


if_then_else = _ir.op._make_op_api(_ir.Op.get("prim.if_then_else"), __name__)


infinity = _tir_op.infinity


isfinite = _tir_op.isfinite


isinf = _tir_op.isinf


isnan = _tir_op.isnan


isnullptr = _tir_op.isnullptr


ldexp = _tir_op.ldexp


likely = _ir.op._make_op_api(_ir.Op.get("prim.likely"), __name__)


log = _tir_op.log


log1p = _tir_op.log1p


log2 = _tir_op.log2


log10 = _tir_op.log10


max_value = _tir_op.max_value


min_value = _tir_op.min_value


nearbyint = _ir.op._make_op_api(_ir.Op.get("prim.nearbyint"), __name__)


nextafter = _tir_op.nextafter


popcount = _tir_op.popcount


pow = _ir.op._make_op_api(_ir.Op.get("prim.pow"), __name__)


round = _ir.op._make_op_api(_ir.Op.get("prim.round"), __name__)


rsqrt = _tir_op.rsqrt


shift_left = _tir_op.shift_left


shift_right = _tir_op.shift_right


sigmoid = _tir_op.sigmoid


sin = _tir_op.sin


sinh = _tir_op.sinh


sqrt = _tir_op.sqrt


tan = _tir_op.tan


tanh = _tir_op.tanh


gpu_thread_return = _tir_op.gpu_thread_return


trunc = _ir.op._make_op_api(_ir.Op.get("prim.trunc"), __name__)


truncdiv = _tir_op.truncdiv


truncmod = _tir_op.truncmod


ptr_byte_offset = _tir_op.ptr_byte_offset


throw_last_error = _tir_op.throw_last_error


stack_alloca = _tir_op.stack_alloca


stack_make_shape = _tir_op.stack_make_shape


stack_make_dltensor = _tir_op.stack_make_dltensor


call_packed = _tir_op.call_packed


_call_ffi_kernel_raw = _ir.op._make_op_api(_ir.Op.get("tirx.call_ffi_kernel"), __name__)


def call_ffi_kernel(*args, launch=None, **kwargs):
    """Call a kernel with a CUDA LaunchConfig, or read a printed low-level call."""
    if launch is not None:
        return _tir_op.call_ffi_kernel(*args, launch=launch, **kwargs)
    return _call_ffi_kernel_raw(*args, **kwargs)


call_cpacked = _tir_op.call_cpacked


call_packed_lowered = _tir_op.call_packed_lowered


call_cpacked_lowered = _tir_op.call_cpacked_lowered


tensor_data_ptr = _tir_op.tensor_data_ptr


handle_add_byte_offset = _tir_op.handle_add_byte_offset


abi_field_set = _tir_op.abi_field_set


abi_field_get = _ir.op._make_op_api(_ir.Op.get("tirx.abi_field_get"), __name__)


gpu_thread_invariant = _tir_op.gpu_thread_invariant


gpu_thread_allreduce = _tir_op.gpu_thread_allreduce


gpu_load_matrix_sync = _tir_op.gpu_load_matrix_sync


gpu_mma_sync = _tir_op.gpu_mma_sync


gpu_fill_fragment = _tir_op.gpu_fill_fragment


gpu_store_matrix_sync = _tir_op.gpu_store_matrix_sync


gpu_storage_sync = _tir_op.gpu_storage_sync
cpu_parallel_barrier = _tir_op.cpu_parallel_barrier


kernel_replace_point = _tir_op.kernel_replace_point


gpu_warp_shuffle = _tir_op.gpu_warp_shuffle


gpu_warp_shuffle_up = _tir_op.gpu_warp_shuffle_up


gpu_warp_shuffle_down = _tir_op.gpu_warp_shuffle_down


gpu_warp_shuffle_xor = _tir_op.gpu_warp_shuffle_xor


gpu_warp_activemask = _tir_op.gpu_warp_activemask


assume = _tir_op.assume
assume_aligned = _tir_op.assume_aligned


undef = _tir_op.undef


alloc_workspace = _tir_op.alloc_workspace


free_workspace = _tir_op.free_workspace


vscale = _tir_op.vscale


ignore_loop_partition = _tir_op.ignore_loop_partition


reinterpret = _ir.op._make_op_api(_ir.Op.get("tirx.reinterpret"), __name__)


call_extern = _ir.op._make_op_api(_ir.Op.get("tirx.call_extern"), __name__)


def call_intrin(func_name, *args, ty, attrs=None, loc=None):
    """Call an intrinsic with an explicit result type."""
    return _tir_op.call_intrin(ty, func_name, *args, attrs=attrs, loc=loc)


call_llvm_intrin = _llvm_result_type(_tir_op.call_llvm_intrin)


call_llvm_pure_intrin = _llvm_result_type(_tir_op.call_llvm_pure_intrin)


call_pure_extern = _ir.op._make_op_api(_ir.Op.get("tirx.call_pure_extern"), __name__)


vector_low = _ir.op._make_op_api(_ir.Op.get("tirx.vector_low"), __name__)


vector_high = _ir.op._make_op_api(_ir.Op.get("tirx.vector_high"), __name__)


vector_combine = _ir.op._make_op_api(_ir.Op.get("tirx.vector_combine"), __name__)


get_active_lane_mask = _ir.op._make_op_api(_ir.Op.get("tirx.get_active_lane_mask"), __name__)


masked_load = _ir.op._make_op_api(_ir.Op.get("tirx.masked_load"), __name__)


masked_store = _tir_op.masked_store


gpu_dp4a = _tir_op.gpu_dp4a


broadcast = Broadcast


ramp = Ramp


fabs = _ir.op._make_op_api(_ir.Op.get("prim.fabs"), __name__)


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
    "CallFFIKernelAttr",
    "Cast",
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
    "Select",
    "Shuffle",
    "Sub",
    "abi_field_get",
    "abi_field_set",
    "abs",
    "acos",
    "acosh",
    "address_of",
    "alloc_workspace",
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
    "copysign",
    "cos",
    "cosh",
    "cpu_parallel_barrier",
    "erf",
    "exp",
    "exp2",
    "exp10",
    "fabs",
    "floor",
    "floordiv",
    "floormod",
    "fma",
    "fmod",
    "free_workspace",
    "get_active_lane_mask",
    "gpu_active_thread_selector",
    "gpu_dp4a",
    "gpu_fill_fragment",
    "gpu_load_matrix_sync",
    "gpu_mma_sync",
    "gpu_storage_sync",
    "gpu_store_matrix_sync",
    "gpu_thread_allreduce",
    "gpu_thread_filter",
    "gpu_thread_invariant",
    "gpu_thread_return",
    "gpu_warp_activemask",
    "gpu_warp_shuffle",
    "gpu_warp_shuffle_down",
    "gpu_warp_shuffle_up",
    "gpu_warp_shuffle_xor",
    "handle_add_byte_offset",
    "hypot",
    "if_then_else",
    "ignore_loop_partition",
    "infinity",
    "isfinite",
    "isinf",
    "isnan",
    "isnullptr",
    "kernel_replace_point",
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
    "ptr_byte_offset",
    "ramp",
    "register_script_namespace",
    "reinterpret",
    "round",
    "rsqrt",
    "select",
    "shift_left",
    "shift_right",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "stack_alloca",
    "stack_make_dltensor",
    "stack_make_shape",
    "tan",
    "tanh",
    "tensor_data_ptr",
    "throw_last_error",
    "trunc",
    "truncdiv",
    "truncmod",
    "undef",
    "vector_combine",
    "vector_high",
    "vector_low",
    "vscale",
    "webgpu",
]

_Loc = _base.LocationEntry | _ir.Location | None


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


def lt_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.lt_`."""
    return _prim_ffi._OpLT(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def le_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.le_`."""
    return _prim_ffi._OpLE(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def gt_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.gt_`."""
    return _prim_ffi._OpGT(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def ge_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ge_`."""
    return _prim_ffi._OpGE(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def eq_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.eq_`."""
    return _prim_ffi._OpEQ(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def ne_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ne_`."""
    return _prim_ffi._OpNE(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


__all__ += ["and_", "eq_", "ge_", "gt_", "if_then_else_", "le_", "lt_", "ne_", "not_", "or_"]
