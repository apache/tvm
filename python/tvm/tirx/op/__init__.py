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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments
"""Operators used in TIR expression.

For helpers exposing ``ty``, omitted or ``None`` result types use core Call
inference; explicit types, including ``Type.missing()``, are retained.
"""

from typing import Any

import tvm_ffi
from tvm_ffi import Array

import tvm
from tvm import tirx
from tvm.ir import (
    Call,
    Expr,
    ExprWithOp,
    PointerType,
    PrimType,
    TensorLoad,
    TensorRegion,
    Var,
    const,
)
from tvm.ir.base import Span
from tvm.ir.prim import clz as clz
from tvm.ir.prim import max_value as max_value
from tvm.ir.prim import min_value as min_value
from tvm.ir.prim.op import all as all
from tvm.ir.prim.op import any as any
from tvm.ir.prim.op import bitwise_and as bitwise_and
from tvm.ir.prim.op import bitwise_not as bitwise_not
from tvm.ir.prim.op import bitwise_or as bitwise_or
from tvm.ir.prim.op import bitwise_xor as bitwise_xor
from tvm.ir.prim.op import ceil as ceil
from tvm.ir.prim.op import ceildiv as ceildiv
from tvm.ir.prim.op import div as div
from tvm.ir.prim.op import floordiv as floordiv
from tvm.ir.prim.op import floormod as floormod
from tvm.ir.prim.op import if_then_else as if_then_else
from tvm.ir.prim.op import indexdiv as indexdiv
from tvm.ir.prim.op import indexmod as indexmod
from tvm.ir.prim.op import likely as likely
from tvm.ir.prim.op import log2 as log2
from tvm.ir.prim.op import max as max
from tvm.ir.prim.op import min as min
from tvm.ir.prim.op import shift_left as shift_left
from tvm.ir.prim.op import shift_right as shift_right
from tvm.ir.prim.op import truncdiv as truncdiv
from tvm.ir.prim.op import truncmod as truncmod
from tvm.ir.prim.op import vscale as vscale

from .. import _ffi_api
from ..buffer import buffer_data, is_tensor_var
from ..expr import ExprOp, IntImm
from ..expr import TensorLoad as _make_tensor_load
from ..type import TensorMapType

tir = tirx  # alias for backward compat with upstream tir.convert() calls

# Insertion order matters: a longer prefix has to be tried before the shorter
# one it starts with, or `ptx_legacy_mma` would strip as `ptx` + `legacy_mma`.
_DEVICE_INTRIN_PREFIX_TO_NAMESPACE = {
    "cuda_": "cuda",
    "ptx_legacy_": "ptx_legacy",
    "ptx_": "ptx",
    "s_tir_": "s_tir",
    "nvshmem_": "nvshmem",
    "nki_": "nki",
}


def register_intrin_lowering(
    op_name,
    target,
    *,
    f=None,
    override=False,
):
    """Register Op lowering function

    Parameters
    ----------
    op_name : str
        The op name

    target : str
        The target string for given intrinsic lowering function

    f : function, optional
        The function to be registered.

    override : bool, optional
        Replace an existing lowering if True; duplicate registration otherwise
        raises ValueError.

    Returns
    -------
    result : function
        The registered lowering, or a decorator if f is not supplied.
    """

    def _register(f):
        """internal register function"""
        tvm.ir.register_op_attr(op_name, target + ".FLowerIntrinsic", f, override)
        return f

    return _register(f) if f is not None else _register


def _canonical_device_intrin_name(func_name: str) -> str:
    """Return the canonical registry name for statically registered device intrinsics."""

    if not isinstance(func_name, str) or not func_name.startswith("tirx."):
        return func_name
    basename = func_name[len("tirx.") :]
    if "." in basename:
        return func_name
    for prefix, namespace in _DEVICE_INTRIN_PREFIX_TO_NAMESPACE.items():
        if basename.startswith(prefix):
            return f"tirx.{namespace}.{basename[len(prefix) :]}"
    return func_name


def _reject_buffer_region(value, api_name):
    """Reject region metadata where a call argument must denote a runtime value."""
    if isinstance(value, TensorRegion):
        raise TypeError(
            f"tirx.{api_name} does not accept TensorRegion arguments; "
            "construct a TensorLoad with explicit indices"
        )
    return value


def _primexpr_ty(expr):
    """Return the runtime primitive type of an expression."""
    if isinstance(expr, tvm.ir.PrimType):
        return expr
    ty = getattr(expr, "ty", None)
    if isinstance(ty, tvm.ir.PrimType):
        return ty
    if isinstance(expr, ExprOp):
        return expr.expr_ty()
    raise TypeError(f"Cannot determine Expr type for {type(expr).__name__}")


def _primexpr_dtype(expr):
    """Return the runtime dtype of a primitive expression without using Expr.dtype."""
    ty = _primexpr_ty(expr)
    if not isinstance(ty, tvm.ir.PrimType):
        raise TypeError(f"Expected PrimType for {type(expr).__name__}, but got {ty}")
    return ty.dtype


def _pack_buffer(buf, span=None):
    """Build intrinsics that packs the buffer."""
    shape = Call(
        "tirx.tvm_stack_make_shape",
        buf.ty.shape,
        span=span,
        ty=PointerType(tvm.ir.PrimType("int64")),
    )
    strides = (
        Call(
            "tirx.tvm_stack_make_shape",
            buf.ty.strides,
            span=span,
            ty=PointerType(tvm.ir.PrimType("int64")),
        )
        if buf.ty.strides
        else 0
    )
    pack_args = [
        buffer_data(buf),
        shape,
        strides,
        len(buf.ty.shape),
        const(0, dtype=buf.ty.dtype),
        buf.ty.elem_offset,
    ]
    return Call("tirx.tvm_stack_make_array", pack_args, span=span, ty="handle")


def call_packed_lowered(*args, span=None, ty=None):
    """Lowered version of call packed.
    The argument to a packed function can be an Expr or a tensor variable.
    The argument is the corresponding POD type when Expr is presented.
    When the argument is a Var carrying TensorType, the corresponding PackedFunc
    will receive an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is Tensor.

    Parameters
    ----------
    args : list of Expr or Var.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [
        _pack_buffer(x) if is_tensor_var(x) else _reject_buffer_region(x, "call_packed_lowered")
        for x in args
    ]
    return Call(
        "tirx.tvm_call_packed_lowered",
        call_args,
        ty=ty,
        span=span,
    )


def call_cpacked_lowered(*args, span=None, ty=None):
    """Lowered version of call c-packed.
    Same as call_packed, except that the first argument is the function name
    (as in call_extern), and the last argument is the resource handle.

    Parameters
    ----------
    args : list of Expr or Var.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [
        _pack_buffer(x) if is_tensor_var(x) else _reject_buffer_region(x, "call_cpacked_lowered")
        for x in args
    ]
    return Call(
        "tirx.tvm_call_cpacked_lowered",
        call_args,
        ty=ty,
        span=span,
    )


def call_packed(*args, span=None, ty=None):
    """Build expression by call an external packed function.

    The argument to a packed function can be an Expr or a tensor variable.
    The argument is the corresponding POD type when Expr is presented.

    When the argument is a Var carrying TensorType, the corresponding PackedFunc
    will receive an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is Tensor.

    Parameters
    ----------
    args : list of Expr or Var.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [
        _pack_buffer(x) if is_tensor_var(x) else _reject_buffer_region(x, "call_packed")
        for x in args
    ]
    return Call("tirx.tvm_call_packed", call_args, ty=ty, span=span)


@tvm_ffi.register_object("tirx.CallFFIKernelAttr")
class CallFFIKernelAttr(tvm.ir.Attrs):
    """Ordered launch tags for an explicit FFI kernel call."""

    launch_params: list[str]

    def __init__(self, launch_params):
        self.__init_handle_by_constructor__(_ffi_api.CallFFIKernelAttr, launch_params)


def call_ffi_kernel(*args, launch_params, ty="int32", span=None):
    """Call a kernel with its symbol, kernel operands, then launch values.

    ``launch_params`` contains ordered tags for the launch-value suffix.
    Flag-only tags consume no argument, and dynamic shared-memory bytes are
    last when present. Host codegen may launch directly; other hosts use the
    existing packed-function calling convention.
    """
    return Call(
        "tirx.call_ffi_kernel",
        args,
        attrs=CallFFIKernelAttr(launch_params),
        ty=ty,
        span=span,
    )


@tvm_ffi.register_object("tirx.TensorMapEncodeTiledAttr")
class TensorMapEncodeTiledAttr(tvm.ir.Attrs):
    """Descriptor dtype and fixed options for tiled tensor-map encoding."""

    def __init__(
        self,
        descriptor_dtype,
        rank,
        interleave=0,
        swizzle=0,
        l2_promotion=0,
        oob_fill=0,
        force_cu_dtype=-1,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorMapEncodeTiledAttr,
            descriptor_dtype,
            rank,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
            force_cu_dtype,
        )


def tensormap_encode_tiled(
    *args,
    descriptor_dtype,
    rank,
    interleave=0,
    swizzle=0,
    l2_promotion=0,
    oob_fill=0,
    force_cu_dtype=-1,
    span=None,
):
    """Encode a tiled tensor map using runtime pointers and shape operands.

    Arguments are the descriptor and data pointers, global dimensions (rank),
    byte strides (rank - 1), box dimensions (rank), and element strides (rank).
    The dtype describes the final descriptor units, including any promotion.
    CUDA-host codegen encodes directly; other hosts use the runtime packed call.
    """
    if not 1 <= rank <= 5 or len(args) != 4 * rank + 1:
        raise ValueError("tensormap_encode_tiled requires rank 1..5 and 4 * rank + 1 operands")
    return Call(
        "tirx.tensormap_encode_tiled",
        args,
        attrs=TensorMapEncodeTiledAttr(
            descriptor_dtype, rank, interleave, swizzle, l2_promotion, oob_fill, force_cu_dtype
        ),
        ty="int32",
        span=span,
    )


def call_cpacked(*args, span=None, ty=None):
    """Build expression by call an external packed function.

    Same as call_packed, except that the first argument is the function name
    (as in call_extern), and the last argument is the resource handle.

    Parameters
    ----------
    args : list of Expr or Var.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [
        _pack_buffer(x) if is_tensor_var(x) else _reject_buffer_region(x, "call_cpacked")
        for x in args
    ]
    return Call("tirx.tvm_call_cpacked", call_args, ty=ty, span=span)


def call_intrin(dtype: str | tvm.ir.Type, func_name, *args, attrs=None, span=None):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str or tvm.ir.Type
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    attrs : Optional[tvm.ir.Attrs or Dict[str, Object]]
        Additional attributes for the call.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.
    """
    if isinstance(func_name, str):
        func_name = _canonical_device_intrin_name(func_name)
    args = tuple(_reject_buffer_region(arg, "call_intrin") for arg in args)
    return Call(func_name, args, attrs=attrs, span=span, ty=dtype)


def call_pure_extern(dtype, func_name, *args, span=None):
    """Build expression by calling a pure extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return Call(
        "tirx.call_pure_extern",
        [func_name, *(_reject_buffer_region(arg, "call_pure_extern") for arg in args)],
        span=span,
        ty=dtype,
    )


def call_extern(dtype, func_name, *args, span=None):
    """Build expression by calling a extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return Call(
        "tirx.call_extern",
        [func_name, *(_reject_buffer_region(arg, "call_extern") for arg in args)],
        span=span,
        ty=dtype,
    )


def _require_float_arg(op_name, x):
    x = tirx.convert(x)
    dtype = _primexpr_dtype(x)
    if "float" not in dtype and "bfloat" not in dtype:
        raise TypeError(f"tirx.{op_name} only supports floating-point inputs, but got {dtype}")
    return x


def call_llvm_intrin(dtype, name, *args, span=None):
    """Build expression by calling a llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.target import codegen

    if isinstance(name, str):
        llvm_id = codegen.llvm_lookup_intrinsic_id(name)
    elif isinstance(name, IntImm):
        llvm_id = name.value
    else:
        llvm_id = name
    if llvm_id == 0:
        raise ValueError(f"Unknown llvm intrinsic function {name}")
    return call_intrin(
        dtype,
        "tirx.call_llvm_intrin",
        name
        if isinstance(name, IntImm)
        else tvm.tirx.const(llvm_id, "int32" if isinstance(name, str) else "uint32"),
        *args,
        span=span,
    )


def call_llvm_pure_intrin(dtype, name, *args, span=None):
    """Build expression by calling a pure llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.target import codegen

    if isinstance(name, str):
        llvm_id = codegen.llvm_lookup_intrinsic_id(name)
    elif isinstance(name, IntImm):
        llvm_id = name.value
    else:
        llvm_id = name
    if llvm_id == 0:
        raise ValueError(f"Unknown llvm intrinsic function {name}")
    return call_intrin(
        dtype,
        "tirx.call_llvm_pure_intrin",
        name
        if isinstance(name, IntImm)
        else tvm.tirx.const(llvm_id, "int32" if isinstance(name, str) else "uint32"),
        *args,
        span=span,
    )


def tvm_stack_alloca(dtype_str, num, *, ty=None, span=None):
    """Return new on stack dtype[num]

    Parameters
    ----------
    dtype_str : str
        The data type of array.

    num : int
        The size of array.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(ty, "tirx.tvm_stack_alloca", dtype_str, num, span=span)


def tvm_stack_make_shape(*args, ty=None, span=None):
    """Allocate a shape tuple on stack, return the handle

    Parameters
    ----------
    args : int
        The tuple shape.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(ty, "tirx.tvm_stack_make_shape", *args, span=span)


def tvm_stack_make_array(
    data,
    shape,
    strides,
    ndim,
    arr_dtype,
    elem_offset,
    *,
    ty=None,
    span=None,
):
    """Allocate a Tensor(DLTensor) on stack, return the handle

    Parameters
    ----------
    data : Expr
        The data of array.

    shape : Expr
        The shape of array.

    strides : Expr
        The strides of array.

    ndim : Expr
        The dimensions of array.

    arr_dtype : Expr
        The data type of array.

    elem_offse : Expr
        The element offset of array.

    Returns
    -------
    call : Expr
        The call expression.
    """
    if isinstance(arr_dtype, str | tvm.DataType | tvm.ir.PrimType):
        arr_dtype = const(0, dtype=arr_dtype)

    return call_intrin(
        ty,
        "tirx.tvm_stack_make_array",
        data,
        shape,
        strides,
        ndim,
        arr_dtype,
        elem_offset,
        span=span,
    )


def assume(cond=None, *, ty=None, span=None):
    """Provide a true statement that can be used for simplifications

    Parameters
    ----------
    cond : Expr
       The constraint condition.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(ty, "tirx.assume", cond, span=span)


def assume_aligned(tensor, alignment_bytes, *, ty=None, span=None):
    """Assume the tensor's base address is aligned to ``alignment_bytes``.

    This compiler fact does not check or modify the address. The tensor must
    be a tensor variable and alignment a scalar integer constant, a power of
    two between 1 and 2**27 bytes (inclusive).
    """
    return call_intrin(ty, "tirx.assume_aligned", tensor, alignment_bytes, span=span)


def undef(*, ty=None, span=None):
    """Returns an initialized but arbitrary value

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(ty, "tirx.undef", span=span)


def handle_add_byte_offset(handle, offset, *, ty=None, span=None):
    """Add offset to handle

    Parameters
    ----------
    handle : Expr
        The handle.

    offset : int
        The offset.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(ty, "tirx.handle_add_byte_offset", handle, offset, span=span)


def tvm_struct_get(arr, index, field, dtype):
    """Get struct field value in array

    Parameters
    ----------
    dtype : str
        The date type of the result.

    arr : StructType*
        The array of struct.

    index : int
        The index of struct.

    field : int
        The field of struct.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(dtype, "tirx.tvm_struct_get", arr, index, field)


def tvm_struct_set(arr, index, field, value, *, ty=None, span=None):
    """Set value in struct field in array

    Parameters
    ----------
    arr : StructType*
        The array of struct.

    index : int
        The index of struct.

    field : int
        The field of struct.

    value : Expr
        The value to be set in field.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(ty, "tirx.tvm_struct_set", arr, index, field, value, span=span)


def _is_tensormap_var(obj: Var) -> bool:
    return isinstance(obj.ty, PointerType) and isinstance(obj.ty.element_type, TensorMapType)


def address_of(obj: Var | TensorLoad, span: Span | None = None, *, ty=None) -> Expr:
    """Returns the address of a buffer element or addressable variable.

    Parameters
    ----------
    obj: Union[Var, TensorLoad]
        The buffer, buffer load, or addressable variable.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : Expr
        The call expression.
    """
    if is_tensor_var(obj):
        n_dim = len(obj.ty.shape)
        buffer_load = _make_tensor_load(obj, [0] * n_dim)
        return Call("tirx.address_of", [buffer_load], ty=ty, span=span)
    elif isinstance(obj, Var):
        if _is_tensormap_var(obj):
            return call_intrin(ty, "tirx.address_of", obj, span=span)
        if not isinstance(obj.ty, tvm.ir.PrimType):
            raise TypeError(f"address_of expects a scalar or TensorMap Var, but got {obj.ty}")
        return Call("tirx.address_of", [obj], ty=ty, span=span)
    elif isinstance(obj, TensorLoad):
        return Call("tirx.address_of", [obj], ty=ty, span=span)
    else:
        raise ValueError(f"Invalid object type: {type(obj)}")


def tvm_thread_allreduce(
    combine,
    identity,
    values,
    predicate,
    destinations,
    thread_axes,
    *,
    ty=None,
    span=None,
):
    """Perform an all-reduce inside a thread block.

    Parameters
    ----------
    combine : tvm.ir.LambdaExpr
        Typed combining lambda with parameters ordered as all left-hand values
        followed by all right-hand values. Its body returns a scalar for one
        result or a Tuple of results.
    identity : Expr or Sequence[Expr]
        Identity value for each reduction result.
    values : Expr or Sequence[Expr]
        Values contributed by the current thread.
    predicate : PrimExpr
        Boolean participation predicate. Inactive threads contribute identities.
    destinations : Expr or Sequence[Expr]
        Tensor loads at index zero of one-element result temporaries, optionally
        cast for boolean storage. Each temporary must be accessed only at index zero.
    thread_axes : Expr or Sequence[Expr]
        Thread variables participating in the reduction.

    Returns
    -------
    call : Expr
        The void call expression with six explicit operands.
    """

    def as_operand(value):
        return tvm.ir.Tuple(value) if isinstance(value, list | tuple | Array) else value

    return call_intrin(
        ty,
        "tirx.tvm_thread_allreduce",
        combine,
        as_operand(identity),
        as_operand(values),
        predicate,
        as_operand(destinations),
        as_operand(thread_axes),
        span=span,
    )


def tvm_thread_invariant(cond, *, ty=None, span=None):
    """Mark condition as thread invariant.

    Parameters
    ----------
    cond : Expr
        The condition.

    Returns
    -------
    call : Expr
        The call expression.
    """
    assert tvm.ir.is_prim_expr(cond)
    return call_intrin(ty, "tirx.tvm_thread_invariant", cond, span=span)


def tvm_storage_sync(storage_scope, is_load=False, num_blocks=-1, *, dtype="void"):
    """Perform synchronization in specified scope.

    Parameters
    ----------
    storage_scope : str
        The storage scope to perform synchronization.

    is_load : bool or Expr or None
        Whether to perform load synchronization. (for global sync only)
        Set both ``is_load`` and ``num_blocks`` to None to omit these operands.

    num_blocks : int or Expr or None
        The number of blocks to synchronize. (for global sync only)
        Set to None to omit this operand.

    dtype : str or tvm.ir.Type
        The stored result type. Defaults to void.

    Returns
    -------
    call : Expr
        The call expression.
    """
    args = [storage_scope]
    if is_load is None:
        if num_blocks is not None:
            raise ValueError("num_blocks must be None when is_load is omitted")
    else:
        args.append(is_load)
        if num_blocks is not None:
            args.append(num_blocks)
    return call_intrin(dtype, "tirx.tvm_storage_sync", *args)


def cpu_parallel_barrier(*, ty=None, span=None):
    """Synchronize all workers in the current CPU parallel launch.

    Every worker must reach this operation at the same program point. Place it
    inside a ``parallel_launch`` region, outside parallel loops,
    whose iteration counts can differ between workers. Writes before the barrier
    are visible to all workers after it.

    To replace ``pragma_parallel_barrier_when_finish``, place this operation
    after the former attribute body.
    """
    return call_intrin(ty, "tirx.cpu_parallel_barrier", span=span)


def tvm_kernel_replace_point(*, ty=None, span=None):
    """Mark where a transform should replace generated kernel initialization."""
    return call_intrin(ty, "tirx.tvm_kernel_replace_point", span=span)


def tvm_warp_shuffle(mask, value, warp_id, width, warp_size, *, ty=None, span=None):
    """Exchange value between threads inside a warp.

    Parameters
    ----------
    mask : Expr
        The warp mask indicates active threads inside warp.
    value : Expr
        The value to exchange.
    warp_id : Expr
        The source lane index to fetch value.
    width : Expr
        The width of sub-sections to perform warp shuffle.
    warp_size : Expr
        The warp size.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_warp_shuffle",
        mask,
        value,
        warp_id,
        width,
        warp_size,
        span=span,
    )


def tvm_warp_shuffle_up(mask, value, offset, width, warp_size, *, ty=None, span=None):
    """Copy value from a lane with lower (by offset) index relative to caller.

    Parameters
    ----------
    mask : Expr
        The warp mask indicates active threads inside warp.
    value : Expr
        The value to exchange.
    offset : Expr
        The difference between source lane index and destination lane index:
        `offset = dst_lane_idx - src_lane_idx`
    width : Expr
        The width of sub-sections to perform warp shuffle.
    warp_size : Expr
        The warp size.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_warp_shuffle_up",
        mask,
        value,
        offset,
        width,
        warp_size,
        span=span,
    )


def tvm_warp_shuffle_down(mask, value, offset, width, warp_size, *, ty=None, span=None):
    """Copy value from a lane with higher (by offset) index relative to caller.

    Parameters
    ----------
    mask : Expr
        The warp mask indicates active threads inside warp.
    value : Expr
        The value to exchange.
    offset : Expr
        The difference between source lane index and destination lane index:
        `offset = src_lane_idx - dst_lane_idx`
    width : Expr
        The width of sub-sections to perform warp shuffle.
    warp_size : Expr
        The warp size.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_warp_shuffle_down",
        mask,
        value,
        offset,
        width,
        warp_size,
        span=span,
    )


def tvm_warp_shuffle_xor(mask, value, lane_mask, width, warp_size, *, ty=None, span=None):
    """Copy value from a lane with index computed by `src_lane_idx ^ lane_mask`.

    Parameters
    ----------
    mask : Expr
        The warp mask indicates active threads inside warp.
    value : Expr
        The value to exchange.
    lane_mask : Expr
        The mask to compute source lane index:
    width : Expr
        The width of sub-sections to perform warp shuffle.
    warp_size : Expr
        The warp size.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_warp_shuffle_xor",
        mask,
        value,
        lane_mask,
        width,
        warp_size,
        span=span,
    )


def tvm_warp_activemask(*, ty=None, span=None):
    """Return a 32-bit mask indicates currently active threads in a calling warp.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(ty, "tirx.tvm_warp_activemask", span=span)


def type_annotation(dtype):
    """Create a type annotation expression

    Parameters
    ----------
    dtype : Expr
        The data type.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(dtype, "tirx.type_annotation")


def tvm_access_ptr(ptype, data, offset, extent, rw_mask, *, ty=None, span=None):
    """Get head access address with memory access pattern info

    Parameters
    ----------
    ptype : Expr, PrimType, or str
        The data type of pointer. If a ``PrimType`` or ``str``, it is wrapped
        via :func:`type_annotation` so that the lowering rule (which reads
        ``args[0].dtype()`` for the cast type) sees the intended dtype instead
        of StringType from a string literal.

    data : DType*
        The data of pointer.

    offset : int
        The offset of pointer.

    extent : int
        The extent of pointer.

    rw_mask : int
        The read write mask.

    Returns
    -------
    call : Expr
        The call expression.
    """
    if isinstance(ptype, str | PrimType):
        ptype = type_annotation(ptype)
    return call_intrin(
        ty,
        "tirx.tvm_access_ptr",
        ptype,
        data,
        offset,
        extent,
        rw_mask,
        span=span,
    )


def ptr_byte_offset(data, byte_offset, dtype, *, ty=None, span=None):
    """Cast ``data + byte_offset`` to ``dtype*``.

    ``byte_offset`` is always in bytes.  Use this when the source CUDA shape
    needs an explicitly typed local pointer derived from a byte-addressed base.
    """
    if isinstance(dtype, str | PrimType):
        dtype = type_annotation(dtype)
    return call_intrin(
        ty,
        "tirx.ptr_byte_offset",
        data,
        byte_offset,
        dtype,
        span=span,
    )


def tvm_throw_last_error(*, ty=None, span=None):
    """Throw TVMGetLastError()

    Returns
    -------
    ret : Expr
        The return expression
    """
    return call_intrin(ty, "tirx.tvm_throw_last_error", span=span)


def print_buffer(
    buffer_var,
    dtype,
    is_string,
    is_scalar,
    dim_num,
    *shape,
    ty=None,
    span=None,
):
    """Print out buffer memory during runtime."""
    if len(shape) == 1 and isinstance(shape[0], tuple | list | tvm.ir.Array):
        final_shape_args = list(shape[0])
    else:
        final_shape_args = list(shape)
    if isinstance(dtype, tvm.ir.PrimType):
        dtype = dtype.dtype
    if isinstance(dtype, tvm.ir.StringImm):
        dtype = dtype.value
    if isinstance(is_string, IntImm):
        is_string = is_string.value
    if isinstance(is_scalar, IntImm):
        is_scalar = is_scalar.value
    if isinstance(dim_num, IntImm):
        dim_num = dim_num.value
    return Call(
        "tirx.print_buffer",
        [
            buffer_var,
            str(tvm.DataType(dtype)),
            const(bool(is_string), "bool"),
            const(bool(is_scalar), "bool"),
            const(dim_num, "uint32"),
            *final_shape_args,
        ],
        ty=ty,
        span=span,
    )


def cooperative_tensor_fill(
    d: Var,
    index: Expr,
    value: Expr,
    rows: int,
    cols: int,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.cooperative_tensor_fill",
        d,
        index,
        value,
        rows,
        cols,
        span=span,
    )


def cooperative_tensor_load(
    d: Var,
    index: Expr,
    ptr: Expr,
    stride: Expr,
    rows: int,
    cols: int,
    transpose_matrix: bool = False,
    mma_M: int = 0,
    mma_N: int = 0,
    mma_K: int = 0,
    operand_role: int = 0,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.cooperative_tensor_load",
        d,
        index,
        ptr,
        stride,
        rows,
        cols,
        transpose_matrix,
        mma_M,
        mma_N,
        mma_K,
        operand_role,
        span=span,
    )


def cooperative_tensor_store(
    d: Expr,
    index: Expr,
    ptr: Expr,
    stride: Expr,
    rows: int,
    cols: int,
    transpose_matrix: bool = False,
    mma_M: int = 0,
    mma_N: int = 0,
    mma_K: int = 0,
    operand_role: int = 0,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.cooperative_tensor_store",
        d,
        index,
        ptr,
        stride,
        rows,
        cols,
        transpose_matrix,
        mma_M,
        mma_N,
        mma_K,
        operand_role,
        span=span,
    )


def cooperative_tensor_multiply_accumulate(
    d: Var,
    index_d: Expr,
    a: Var,
    index_a: Expr,
    b: Var,
    index_b: Expr,
    c: Var,
    index_c: Expr,
    M: int,
    N: int,
    K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.cooperative_tensor_multiply_accumulate",
        d,
        index_d,
        a,
        index_a,
        b,
        index_b,
        c,
        index_c,
        M,
        N,
        K,
        transpose_a,
        transpose_b,
        span=span,
    )


def vectorlow(dtype, vec):
    """Get the low level half of the vector

    Parameters
    ----------
    dtype : str
       The data type of the result.

    vec : list
       The input vector.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(dtype, "tirx.vectorlow", vec)


def vectorhigh(dtype, vec):
    """Get the high level half of the vector

    Parameters
    ----------
    dtype : str
       The data type of the result.

    vec : list
       The input vector.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(dtype, "tirx.vectorhigh", vec)


def vectorcombine(dtype, vec1, vec2):
    """Concat two vectors

    Parameters
    ----------
    vec1 : list
       The input vector.

    vec2 : list
       The input vector.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(dtype, "tirx.vectorcombine", vec1, vec2)


def dp4a(vec1, vec2, acc=0, **kwargs):
    """Dot product of two int8x4 vectors and add an optional accumulator

    Parameters
    ----------
    vec1 : int8x4
       The input vector.

    vec2 : int8x4
       The input vector.

    acc : int32
       The accumulator.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return Call("tirx.dp4a", [vec1, vec2, acc], **kwargs)


def infinity(dtype: str, span: Span | None = None) -> Any:
    """infinity value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The infinity value of dtype.
    """
    return _ffi_api.infinity(dtype, span)  # type: ignore


def reinterpret(dtype, value, span: Span | None = None) -> Expr:
    """Reinterpret a value as an exact primitive or pointer type.

    Parameters
    ----------
    dtype : str or tvm.ir.Type
        The data type.

    value : Expr
        The input value.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The reinterpret cast value of dtype.
    """
    if isinstance(dtype, str):
        dtype = (
            PointerType(tvm.ir.PrimType("void")) if dtype == "handle" else tvm.ir.PrimType(dtype)
        )
    return _ffi_api.reinterpret(dtype, value, span)  # type: ignore


def exp(x, *, ty=None, span=None):
    """Take exponential of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.exp", x, span=span)


def exp2(x, *, ty=None, span=None):
    """Calculate 2**x

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.exp2", x, span=span)


def exp10(x, *, ty=None, span=None):
    """Calculate 10**x

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.exp10", x, span=span)


def fma(x, y, z, *, ty=None, span=None):
    """Take fused multiply-add of input x, y, z.

    Parameters
    ----------
    x : Expr
        First input argument.

    y : Expr
        Second input argument.

    z : Expr
        Third input argument.

    Returns
    -------
    out : Expr
        The result of x * y + z.
    """
    x = tir.convert(x)
    y = tir.convert(y)
    z = tir.convert(z)
    return call_intrin(ty, "tirx.fma", x, y, z, span=span)


def erf(x, *, ty=None, span=None):
    """Take gauss error function of the input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.erf", x, span=span)


def tanh(x, *, ty=None, span=None):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.tanh", x, span=span)


def sigmoid(x, *, ty=None, span=None):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.sigmoid", x, span=span)


def log(x, *, ty=None, span=None):
    """Take log of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.log", x, span=span)


def log10(x, *, ty=None, span=None):
    """Take log10 of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.log10", x, span=span)


def log1p(x, *, ty=None, span=None):
    """Take log(x + 1) with respect to input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.log1p", x, span=span)


def tan(x, *, ty=None, span=None):
    """Take tan of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = _require_float_arg("tan", x)
    return call_intrin(ty, "tirx.tan", x, span=span)


def cos(x, *, ty=None, span=None):
    """Take cos of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = _require_float_arg("cos", x)
    return call_intrin(ty, "tirx.cos", x, span=span)


def cosh(x, *, ty=None, span=None):
    """Take cosh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.cosh", x, span=span)


def acos(x, *, ty=None, span=None):
    """Take acos of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.acos", x, span=span)


def acosh(x, *, ty=None, span=None):
    """Take acos of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.acosh", x, span=span)


def sin(x, *, ty=None, span=None):
    """Take sin of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = _require_float_arg("sin", x)
    return call_intrin(ty, "tirx.sin", x, span=span)


def sinh(x, *, ty=None, span=None):
    """Take sinh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.sinh", x, span=span)


def asin(x, *, ty=None, span=None):
    """Take asin of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.asin", x, span=span)


def asinh(x, *, ty=None, span=None):
    """Take asinh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.asinh", x, span=span)


def atan(x, *, ty=None, span=None):
    """Take atan of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.atan", x, span=span)


def atanh(x, *, ty=None, span=None):
    """Take atanh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.atanh", x, span=span)


def atan2(x1, x2, *, ty=None, span=None):
    """Take arctan2(x1, x2).

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(ty, "tirx.atan2", x1, x2, span=span)


def sqrt(x, *, ty=None, span=None):
    """Take square root of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.sqrt", x, span=span)


def rsqrt(x, *, ty=None, span=None):
    """Take reciprocal of square root of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.rsqrt", x, span=span)


def floor(x: ExprWithOp, span=None):
    """Take floor of float input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.floor(x, span)  # type: ignore


def trunc(x, span=None):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.trunc(x, span)  # type: ignore


def abs(x, span=None):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.abs(x, span)  # type: ignore


def round(x, span=None):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.round(x, span)  # type: ignore


def nearbyint(x, span=None):
    """Round elements of the array to the nearest integer.
    This intrinsic uses llvm.nearbyint instead of llvm.round
    which is faster but will results different from te.round.
    Notably nearbyint rounds according to the rounding mode,
    whereas te.round (llvm.round) ignores that.
    For differences between the two see:
    https://en.cppreference.com/w/cpp/numeric/math/round
    https://en.cppreference.com/w/cpp/numeric/math/nearbyint

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.nearbyint(x, span)  # type: ignore


def nextafter(x1, x2, *, ty=None, span=None):
    """Return the next floating-point value after x1 towards x2.

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(ty, "tirx.nextafter", x1, x2, span=span)  # type: ignore


def hypot(x1, x2, *, ty=None, span=None):
    """Equivalent to sqrt(x1**2 + x2**2), element-wise.

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(ty, "tirx.hypot", x1, x2, span=span)  # type: ignore


def copysign(x1, x2, *, ty=None, span=None):
    """Change the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(ty, "tirx.copysign", x1, x2, span=span)  # type: ignore


def ldexp(x1, x2, *, ty=None, span=None):
    """Returns x1 * (2 ** x2).

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(ty, "tirx.ldexp", x1, x2, span=span)  # type: ignore


def filter(var, pred, *, span=None, ty=None):  # pylint: disable=redefined-builtin
    """Thread-set filter escape hatch.

    Use this wrapper only when the predicate is *not* in the canonical
    thread-filter grammar (see ``src/tirx/analysis/filter_canonical.h``).
    Canonical predicates -- pure conjunctions of ``scopeid_var <op> const``
    comparisons plus bare ``T.cuda.elect_sync()`` calls -- are recognized by
    the lowering pass directly from ``if cond:``, so the wrapper is redundant
    for them.

    When wrapped: ``var`` (a ``ScopeIdDef``-declared scope identifier) tells
    the compiler which active-set axis to collapse to a singleton when the
    opaque predicate evaluates true; ``pred`` is preserved verbatim and
    evaluated at runtime.

    The legacy three-argument range form ``filter(var, lo, hi)`` has been
    removed -- write ``lo <= var and var < hi`` (or ``var == lo`` when
    ``hi == lo + 1``) at the call site instead.
    """
    return call_intrin(ty, "tirx.filter", var, pred, span=span)


def selector(var, pred, span=None, *, ty=None):
    """Analysis-only active-thread selector.

    ``selector(var, pred)`` denotes the unique value of ``var`` in the current
    active domain for which ``pred`` is true. It is intended for compiler
    metadata and should not survive to executable codegen.
    """
    return call_intrin(ty, "tirx.selector", var, pred, span=span)


def isnan(x, span=None):
    """Check if input value is Nan.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isnan(x, span)  # type: ignore


def isnullptr(x, span=None, *, ty=None):
    """Check if input value is nullptr.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return call_intrin(ty, "tirx.isnullptr", x, span=span)  # type: ignore


def isfinite(x, span=None):
    """Check if input value is finite.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isfinite(x, span)  # type: ignore


def isinf(x, span=None):
    """Check if input value is infinite.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isinf(x, span)  # type: ignore


def power(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api._OpPow(x, y, span)  # type: ignore


def pow(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api._OpPow(x, y, span)  # type: ignore


def popcount(x, *, ty=None, span=None):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(ty, "tirx.popcount", x, span=span)


def q_multiply_shift(x, y, q, s, *, ty=None, span=None):
    """Execute a multiplication between two Q-numbers x and y
    followed by a right shift s. The mathematical expression is:

       out = round(x*y*2^-s)

    More about Q-numbers here: https://en.wikipedia.org/wiki/Q_(number_format)
    The rounding rule is to the nearest value, rounding half up
    (i.e., round(x.1) = x and round (x.5) = x+1)

    Parameters
    ----------
    x : Expr
        First Q-number
    y : Expr
        Second Q-number
    q : Expr
        Number of fractional bits in x and y. Needs to be > 0
    s : Expr
        Integer shift

    Returns
    -------
    y : Expr
        The result.
    """
    return call_intrin(ty, "tirx.q_multiply_shift", x, y, q, s, span=span)


def q_multiply_shift_per_axis(
    x: Expr,
    y: Expr,
    ls: Expr,
    rs: Expr,
    q: IntImm,
    is_lshift_required: IntImm,
    is_rshift_required: IntImm,
    *,
    ty=None,
    span=None,
):
    """Execute a multiplication between two Q-numbers x and y

    Parameters
    ----------
    x : Expr
        First Q-number.
    y : Expr
        Second Q-number.
    ls : Expr
         Integer left shift.
    rs : Expr
         Integer right shift.
    q : IntImm
        Number of fractional bits in x and y. Needs to be > 0.
    is_lshift_required : IntImm
                         Whether we need to do left shift or not.
    is_rshift_required : IntImm
                         Whether we need to do right shift or not.

    Returns
    -------
    z : Expr
        The result.
    """
    return call_intrin(
        ty,
        "tirx.q_multiply_shift_per_axis",
        x,
        y,
        ls,
        rs,
        q,
        is_lshift_required,
        is_rshift_required,
        span=span,
    )


def fmod(x, y, *, ty=None, span=None):
    """Return the remainder of x divided by y with the same sign as x.

    Parameters
    ----------
    x : Expr
        Input argument.
    y : Expr
        Input argument.

    Returns
    -------
    z : Expr
        The result.
    """
    x = tir.convert(x)
    y = tir.convert(y)
    return call_intrin(ty, "tirx.fmod", x, y, span=span)


def logaddexp(a, b, span=None):
    """Compute the logaddexp of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.
    """
    return _ffi_api._OpLogAddExp(a, b, span)  # type: ignore


def TVMBackendAllocWorkspace(
    device_type,
    device_id,
    nbytes,
    dtype_code_hint,
    dtype_bits_hint,
    *,
    ty=None,
    span=None,
):
    """Backend function to allocate temporal workspace

    Parameters
    ----------
    device_type : int
        The device type which the space will be allocated.

    device_id : int
        The device id which the space will be allocated.

    nbytes : int
        The size of the space requested.

    dtype_code_hint : int
        The type code of the array elements. Only used in certain backends such as OpenGL.

    dtype_bits_hint : int
        The type bits of the array elements. Only used in certain backends such as OpenGL.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.TVMBackendAllocWorkspace",
        device_type,
        device_id,
        nbytes,
        dtype_code_hint,
        dtype_bits_hint,
        span=span,
    )


def TVMBackendFreeWorkspace(device_type, device_id, ptr, *, ty=None, span=None):
    """Backend function to free temporal workspace.

    Parameters
    ----------
    device_type : int
        The device type which the space will be allocated.

    device_id : int
        The device id which the space will be allocated.

    ptr : Var
        The result allocated space pointer.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.TVMBackendFreeWorkspace",
        device_type,
        device_id,
        ptr,
        span=span,
    )


def get_active_lane_mask(dtype, base, limit):
    """
    Calculate a predicate mask given an upper bound (limit) and a current value (base).

    It will be lowered to the llvm.get.active.lane.mask intrinsic.
    (https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics)

    Parameters
    ----------
    dtype : str
        The data type of the result.

    base : Expr
        An expression reprsenting the base.

    limit : Expr
        An expression representing the limit.
    """
    return call_intrin(dtype, "tirx.get_active_lane_mask", base, limit)


def masked_load(dtype, buffer, *indices_and_mask):
    """Load vector lanes selected by a predicate mask.

    Parameters
    ----------
    dtype : str
        The vector data type to load.

    buffer : Var
        The buffer to load.

    indices_and_mask : Expr
        The buffer indices followed by a boolean lane mask. The mask must match the
        lane count and scalability of the loaded vector.

    Returns
    -------
    call : Expr
        A ``tirx.masked_load`` call with result type ``dtype``.
    """
    return call_intrin(dtype, "tirx.masked_load", buffer, *indices_and_mask)


def masked_store(buffer, value, *indices_and_mask, ty=None, span=None):
    """Store vector lanes selected by a predicate mask.

    Parameters
    ----------
    buffer : Var
        The buffer to update.

    value : Expr
        The vector value to store.

    indices_and_mask : Expr
        The buffer indices followed by a boolean lane mask. The mask must match the
        lane count and scalability of ``value``.

    Returns
    -------
    call : Expr
        A void-typed ``tirx.masked_store`` call.
    """
    return call_intrin(
        ty,
        "tirx.masked_store",
        buffer,
        value,
        *indices_and_mask,
        span=span,
    )


def get_vscale_expr(dtype: str | tvm_ffi.dtype, min_size: int = 128) -> Expr:
    """
    Create a datatype dependent scalable expression.

    Parameters
    ----------
    dtype : Union[str, tvm_ffi.DataType]
        Element data type.
    min_size : int
        The minimum size of the scalable vector in bits.
    """
    if isinstance(dtype, str):
        dtype = tvm_ffi.dtype(dtype)
    return min_size // dtype.bits * vscale()


def ignore_loop_partition(predicate, *, ty=None, span=None) -> Expr:
    """
    Annotate a predicate not be considered as target condition of loop partition.

    Parameters
    ----------
    predicate : Expr
        The annotated predicate expression.
    """
    return call_intrin(ty, "tirx.ignore_loop_partition", predicate, span=span)


def tvm_load_matrix_sync(
    fragment,
    m,
    n,
    k,
    index,
    buffer_ptr,
    stride,
    layout,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for tensor core load operators

    Parameters
    ----------
    fragment : Var
        The wmma fragment.

    m : UIntImm
        The shape of wmma fragment.

    n : UIntImm
        The shape of wmma fragment.

    k : UIntImm
        The shape of wmma fragment.

    index : Expr
        The fragment index.

    buffer_ptr : Expr
        The fragment buffer pointer.

    stride : Expr
        The fragment stride.

    layout : Literal["row_major", "column_major"]
        The fragment layout.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_load_matrix_sync",
        fragment,
        m,
        n,
        k,
        index,
        buffer_ptr,
        stride,
        layout,
        span=span,
    )


def tvm_mma_sync(
    fragment_d,
    index_d,
    fragment_a,
    index_a,
    fragment_b,
    index_b,
    fragment_c,
    index_c,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for tensor core mma_sync operators

    Parameters
    ----------
    fragment_d : Var
        The wmma fragment_d.

    index_d : Expr
        The fragment_d index.

    fragment_a : Var
        The wmma fragment_a.

    index_a : Expr
        The fragment_a index.

    fragment_b : Var
        The wmma fragment_b.

    index_b : Expr
        The fragment_b index.

    fragment_c : Var
        The wmma fragment_c.

    index_c : Expr
        The fragment_c index.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_mma_sync",
        fragment_d,
        index_d,
        fragment_a,
        index_a,
        fragment_b,
        index_b,
        fragment_c,
        index_c,
        span=span,
    )


def tvm_bmma_sync(
    fragment_d,
    index_d,
    fragment_a,
    index_a,
    fragment_b,
    index_b,
    fragment_c,
    index_c,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for tensor core bmma_sync operators

    Parameters
    ----------
    fragment_d : Var
        The bwmma fragment_d.

    index_d : Expr
        The fragment_d index.

    fragment_a : Var
        The bwmma fragment_a.

    index_a : Expr
        The fragment_a index.

    fragment_b : Var
        The bwmma fragment_b.

    index_b : Expr
        The fragment_b index.

    fragment_c : Var
        The bwmma fragment_c.

    index_c : Expr
        The fragment_c index.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_bmma_sync",
        fragment_d,
        index_d,
        fragment_a,
        index_a,
        fragment_b,
        index_b,
        fragment_c,
        index_c,
        span=span,
    )


def tvm_fill_fragment(fragment, m, n, k, index, value, *, ty=None, span=None):
    """TVM intrinsic for tensor core fill_fragment operators

    Parameters
    ----------
    fragment : Var
        The wmma fragment

    m : UIntImm
        The shape of wmma fragment.

    n : UIntImm
        The shape of wmma fragment.

    k : UIntImm
        The shape of wmma fragment.

    index : Expr
        The fragment index.

    value : Expr
        The value to be filled in fragment.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_fill_fragment",
        fragment,
        m,
        n,
        k,
        index,
        value,
        span=span,
    )


def tvm_store_matrix_sync(
    fragment,
    m,
    n,
    k,
    index,
    buffer_ptr,
    stride,
    layout,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for tensor core store operators

    Parameters
    ----------
    fragment : Var
        The wmma fragment.

    m : UIntImm
        The shape of wmma fragment.

    n : UIntImm
        The shape of wmma fragment.

    k : UIntImm
        The shape of wmma fragment.

    index : Expr
        The fragment index.

    buffer_ptr : Expr
        The fragment buffer pointer.

    stride : Expr
        The fragment stride.

    layout : Literal["row_major", "column_major"]
        The fragment layout.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.tvm_store_matrix_sync",
        fragment,
        m,
        n,
        k,
        index,
        buffer_ptr,
        stride,
        layout,
        span=span,
    )


def thread_return(*, ty=None, span=None):
    """Return from the current GPU thread without a function value."""
    return call_intrin(ty, "tirx.thread_return", span=span)


def __getattr__(name):
    # Tile classes resolve registered Ops at import time. Load the family only
    # when requested, after the core TIRx types are available.
    if name == "tile":
        from importlib import import_module  # pylint: disable=import-outside-toplevel

        return import_module(".tile", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
