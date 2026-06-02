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
"""Operators used in TIR expression."""

from typing import Any

import tvm_ffi
from tvm_ffi import Array

import tvm
from tvm import tirx
from tvm.ir import Op, PointerType, PrimExpr
from tvm.ir.base import Span
from tvm.ir.type import TensorMapType
from tvm.runtime import const

from . import _ffi_api
from .buffer import Buffer
from .expr import BufferLoad, Call, CommReducer, IntImm, PrimExprWithOp, Var

# Choice / IntAttr value tables — single source of truth in
# tvm.tirx.operator.intrinsics._common. Re-exported here under their
# underscored names so the existing _choice(name, value, _FOO) call sites
# below keep working without changes.
from .operator.intrinsics._common import CLUSTER_BARRIER_SEM as _CLUSTER_BARRIER_SEM
from .operator.intrinsics._common import CP_ASYNC_BULK_CACHE_HINT as _CP_ASYNC_BULK_CACHE_HINT
from .operator.intrinsics._common import CP_ASYNC_BULK_RED_OP as _CP_ASYNC_BULK_RED_OP
from .operator.intrinsics._common import CP_ASYNC_CACHE_HINT as _CP_ASYNC_CACHE_HINT
from .operator.intrinsics._common import CP_ASYNC_FILL_MODE as _CP_ASYNC_FILL_MODE
from .operator.intrinsics._common import CP_ASYNC_PREFETCH_SIZE as _CP_ASYNC_PREFETCH_SIZE
from .operator.intrinsics._common import F32X2_ROUND as _F32X2_ROUND
from .operator.intrinsics._common import FENCE_PROXY_ASYNC_SPACE as _FENCE_PROXY_ASYNC_SPACE
from .operator.intrinsics._common import FENCE_SCOPE as _FENCE_SCOPE
from .operator.intrinsics._common import FENCE_SEM as _FENCE_SEM
from .operator.intrinsics._common import LDMATRIX_DTYPE as _LDMATRIX_DTYPE
from .operator.intrinsics._common import LDMATRIX_NUM as _LDMATRIX_NUM
from .operator.intrinsics._common import NVSHMEM_CMP as _NVSHMEM_CMP
from .operator.intrinsics._common import NVSHMEM_SIG_OP as _NVSHMEM_SIG_OP
from .operator.intrinsics._common import TCGEN05_CP_DECOMPRESS as _TCGEN05_CP_DECOMPRESS
from .operator.intrinsics._common import TCGEN05_CP_MULTICAST as _TCGEN05_CP_MULTICAST
from .operator.intrinsics._common import TCGEN05_CP_SHAPES as _TCGEN05_CP_SHAPES
from .operator.intrinsics._common import TCGEN05_CTA_GROUP as _TCGEN05_CTA_GROUP
from .operator.intrinsics._common import TCGEN05_LDST_SHAPES as _TCGEN05_LDST_SHAPES

tir = tirx  # alias for backward compat with upstream tir.convert() calls


def _pack_buffer(buf, span=None):
    """Build intrinsics that packs the buffer."""
    shape = Call("handle", "tirx.tvm_stack_make_shape", buf.shape, span=span)
    strides = (
        Call("handle", "tirx.tvm_stack_make_shape", buf.strides, span=span) if buf.strides else 0
    )
    pack_args = [
        buf.data,
        shape,
        strides,
        len(buf.shape),
        const(0, dtype=buf.dtype),
        buf.elem_offset,
    ]
    return Call("handle", Op.get("tirx.tvm_stack_make_array"), pack_args, span=span)


def call_packed_lowered(*args, span=None):
    """Lowered version of call packed.
    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.
    When the argument is Buffer, the corresponding PackedFunc
    will receive an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is Tensor.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    return Call("int32", Op.get("tirx.tvm_call_packed_lowered"), call_args, span=span)


def call_cpacked_lowered(*args, span=None):
    """Lowered version of call c-packed.
    Same as call_packed, except that the first argument is the function name
    (as in call_extern), and the last argument is the resource handle.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    return Call("int32", Op.get("tirx.tvm_call_cpacked_lowered"), call_args, span=span)


def call_packed(*args, span=None):
    """Build expression by call an external packed function.

    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.

    When the argument is Buffer, the corresponding PackedFunc
    will receive an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is Tensor.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    return Call("int32", Op.get("tirx.tvm_call_packed"), call_args, span=span)


def call_cpacked(*args, span=None):
    """Build expression by call an external packed function.

    Same as call_packed, except that the first argument is the function name
    (as in call_extern), and the last argument is the resource handle.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    te.extern : Create tensor with extern function call.
    """
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    return Call("int32", Op.get("tirx.tvm_call_cpacked"), call_args, span=span)


def call_intrin(dtype, func_name, *args, attrs=None, span=None):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
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
    call : PrimExpr
        The call expression.
    """
    return Call(dtype, func_name, args, attrs=attrs, span=span)


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
    call : PrimExpr
        The call expression.
    """
    return Call(dtype, Op.get("tirx.call_pure_extern"), [func_name, *args], span=span)


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
    call : PrimExpr
        The call expression.
    """
    return Call(dtype, Op.get("tirx.call_extern"), [func_name, *args], span=span)


def _require_float_arg(op_name, x):
    x = tirx.convert(x)
    if "float" not in x.dtype and "bfloat" not in x.dtype:
        raise TypeError(f"tirx.{op_name} only supports floating-point inputs, but got {x.dtype}")
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
    call : PrimExpr
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
        Op.get("tirx.call_llvm_intrin"),
        tvm.tirx.const(llvm_id, "uint32"),
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
    call : PrimExpr
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
        Op.get("tirx.call_llvm_pure_intrin"),
        tvm.tirx.const(llvm_id, "uint32"),
        *args,
        span=span,
    )


def tvm_stack_alloca(dtype_str, num):
    """Return new on stack dtype[num]

    Parameters
    ----------
    dtype_str : str
        The data type of array.

    num : int
        The size of array.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.tvm_stack_alloca", dtype_str, num)


def tvm_stack_make_shape(*args):
    """Allocate a shape tuple on stack, return the handle

    Parameters
    ----------
    args : int
        The tuple shape.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.tvm_stack_make_shape", *args)


def tvm_stack_make_array(data, shape, strides, ndim, arr_dtype, elem_offset):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle",
        "tirx.tvm_stack_make_array",
        data,
        shape,
        strides,
        ndim,
        arr_dtype,
        elem_offset,
    )


def assume(cond=None):
    """Provide a true statement that can be used for simplifications

    Parameters
    ----------
    cond : Expr
       The constraint condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("bool", "tirx.assume", cond)


def undef():
    """Returns an initialized but arbitrary value

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int32", "tirx.undef")


def call_tir(global_var: tvm.ir.GlobalVar, *args):
    """Performs a call into another PrimFunc in the same IRModule

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    assert isinstance(global_var, tvm.ir.GlobalVar)

    dtype = "void"
    if global_var.struct_info is not None:
        ret_sinfo = global_var.struct_info.ret
        if hasattr(ret_sinfo, "dtype"):
            dtype = ret_sinfo.dtype

    return Call(dtype=dtype, op=global_var, args=args)


def start_profile_intrinsic(id):
    """Start profile intrinsic.
    Parameters
    ----------
    id : int
        The intrinsic id.
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.start_profile_intrinsic", id)


def end_profile_intrinsic(id):
    """End profile intrinsic.
    Parameters
    ----------
    id : int
        The intrinsic id.
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.end_profile_intrinsic", id)


def tvm_tuple(*value):
    """Create a tuple structure in value field of AttrStmt

    Parameters
    ----------
    value : Expr
        The value in tuple.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.tvm_tuple", *value)


def handle_add_byte_offset(handle, offset):
    """Add offset to handle

    Parameters
    ----------
    handle : Expr
        The handle.

    offset : int
        The offset.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.handle_add_byte_offset", handle, offset)


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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(dtype, "tirx.tvm_struct_get", arr, index, field)


def tvm_struct_set(arr, index, field, value):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int32", "tirx.tvm_struct_set", arr, index, field, value)


def _is_tensormap_var(obj: Var) -> bool:
    type_annotation = obj.type_annotation
    return isinstance(type_annotation, PointerType) and isinstance(
        type_annotation.element_type, TensorMapType
    )


def address_of(obj: Buffer | BufferLoad | Var, span: Span | None = None) -> PrimExpr:
    """Returns the address of a buffer element or addressable variable.

    Parameters
    ----------
    obj: Union[Buffer, BufferLoad, Var]
        The buffer, buffer load, or addressable variable.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if isinstance(obj, Buffer):
        n_dim = len(obj.shape)
        buffer_load = BufferLoad(obj, [0] * n_dim)
        return call_intrin("handle", "tirx.address_of", buffer_load, span=span)
    elif isinstance(obj, Var):
        dtype = "uint64" if _is_tensormap_var(obj) else "handle"
        return call_intrin(dtype, "tirx.address_of", obj, span=span)
    elif isinstance(obj, BufferLoad):
        return call_intrin("handle", "tirx.address_of", obj, span=span)
    else:
        raise ValueError(f"Invalid object type: {type(obj)}")


def lookup_param(param_name, span=None):
    """Returns the param by name

    Parameters
    ----------
    param_name : str
        The name of param.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.lookup_param", param_name, span=span)


def tvm_thread_allreduce(*freduce_args):
    """Perform allreduce inside threadblock.

    Parameters
    ----------
    freduce_args : Expr
        The args.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.tvm_thread_allreduce", *freduce_args)


def tvm_thread_invariant(cond):
    """Mark condition as thread invariant.

    Parameters
    ----------
    cond : Expr
        The condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    assert isinstance(cond, PrimExpr)
    return call_intrin(cond.dtype, "tirx.tvm_thread_invariant", cond)


def tvm_storage_sync(storage_scope, is_load=False, num_blocks=-1):
    """Perform synchronization in specified scope.

    Parameters
    ----------
    storage_scope : str
        The storage scope to perform synchronization.

    is_load : bool
        Whether to perform load synchronization. (for global sync only)

    num_blocks : int
        The number of blocks to synchronize. (for global sync only)

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("void", "tirx.tvm_storage_sync", storage_scope, is_load, num_blocks)


def tvm_global_barrier_kinit():
    """Initialize the global barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("void", "tirx.tvm_global_barrier_kinit")


def tvm_warp_shuffle(mask, value, warp_id, width, warp_size):
    """Exchange value between threads inside a warp.

    Parameters
    ----------
    mask : PrimExpr
        The warp mask indicates active threads inside warp.
    value : PrimExpr
        The value to exchange.
    warp_id : PrimExpr
        The source lane index to fetch value.
    width : PrimExpr
        The width of sub-sections to perform warp shuffle.
    warp_size : PrimExpr
        The warp size.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(value.dtype, "tirx.tvm_warp_shuffle", mask, value, warp_id, width, warp_size)


def tvm_warp_shuffle_up(mask, value, offset, width, warp_size):
    """Copy value from a lane with lower (by offset) index relative to caller.

    Parameters
    ----------
    mask : PrimExpr
        The warp mask indicates active threads inside warp.
    value : PrimExpr
        The value to exchange.
    offset : PrimExpr
        The difference between source lane index and destination lane index:
        `offset = dst_lane_idx - src_lane_idx`
    width : PrimExpr
        The width of sub-sections to perform warp shuffle.
    warp_size : PrimExpr
        The warp size.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        value.dtype, "tirx.tvm_warp_shuffle_up", mask, value, offset, width, warp_size
    )


def tvm_warp_shuffle_down(mask, value, offset, width, warp_size):
    """Copy value from a lane with higher (by offset) index relative to caller.

    Parameters
    ----------
    mask : PrimExpr
        The warp mask indicates active threads inside warp.
    value : PrimExpr
        The value to exchange.
    offset : PrimExpr
        The difference between source lane index and destination lane index:
        `offset = src_lane_idx - dst_lane_idx`
    width : PrimExpr
        The width of sub-sections to perform warp shuffle.
    warp_size : PrimExpr
        The warp size.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        value.dtype, "tirx.tvm_warp_shuffle_down", mask, value, offset, width, warp_size
    )


def tvm_warp_shuffle_xor(mask, value, lane_mask, width, warp_size):
    """Copy value from a lane with index computed by `src_lane_idx ^ lane_mask`.

    Parameters
    ----------
    mask : PrimExpr
        The warp mask indicates active threads inside warp.
    value : PrimExpr
        The value to exchange.
    lane_mask : PrimExpr
        The mask to compute source lane index:
    width : PrimExpr
        The width of sub-sections to perform warp shuffle.
    warp_size : PrimExpr
        The warp size.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        value.dtype, "tirx.tvm_warp_shuffle_xor", mask, value, lane_mask, width, warp_size
    )


def tvm_warp_activemask():
    """Return a 32-bit mask indicates currently active threads in a calling warp.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("uint32", "tirx.tvm_warp_activemask")


def type_annotation(dtype):
    """Create a type annotation expression

    Parameters
    ----------
    dtype : Expr
        The data type.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(dtype, "tirx.type_annotation")


def tvm_access_ptr(ptype, data, offset, extent, rw_mask):
    """Get head access address with memory access pattern info

    Parameters
    ----------
    ptype : Expr or str
        The data type of pointer. If a ``str``, it is wrapped via
        :func:`type_annotation` so that the lowering rule (which reads
        ``args[0].dtype()`` for the cast type) sees the intended dtype
        instead of ``void`` from a raw StringImm.

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
    call : PrimExpr
        The call expression.
    """
    if isinstance(ptype, str):
        ptype = type_annotation(ptype)
    return call_intrin("handle", "tirx.tvm_access_ptr", ptype, data, offset, extent, rw_mask)


def ptr_byte_offset(data, byte_offset, dtype):
    """Cast ``data + byte_offset`` to ``dtype*``.

    ``byte_offset`` is always in bytes.  Use this when the source CUDA shape
    needs an explicitly typed local pointer derived from a byte-addressed base.
    """
    if isinstance(dtype, str):
        dtype = type_annotation(dtype)
    return call_intrin("handle", "tirx.ptr_byte_offset", data, byte_offset, dtype)


def tvm_throw_last_error():
    """Throw TVMGetLastError()

    Returns
    -------
    ret : PrimExpr
        The return expression
    """
    return call_intrin("handle", "tirx.tvm_throw_last_error")


def make_filled_simdgroup_matrix(
    d: Var,
    index: PrimExpr,
    value: PrimExpr,
    col: int = 8,
    row: int = 8,
):
    """Create a filled SIMDGroup matrix

    Parameters
    ----------
    d : var
        The simdgroup var

    index : PrimExpr
        The index of the matrix.

    value : PrimExpr
        The value to fill.

    col : int
        The number of columns.

    row : int
        The number of rows.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.make_filled_simdgroup_matrix", d, index, value, col, row)


def simdgroup_load(
    d: Var,
    index: PrimExpr,
    ptr: PrimExpr,
    stride: PrimExpr,
    col: int = 8,
    row: int = 8,
    transpose_matrix: bool = False,
):
    """Load data from device memory or threadgroup memory to simdgroup

    Parameters
    ----------
    d : var
        The simdgroup var

    index : PrimExpr
        The index of the matrix.

    ptr : PrimExpr
        The pointer.

    stride : PrimExpr
        The stride.

    col : int
        The number of columns.

    row : int
        The number of rows.

    transpose_matrix : bool
        Whether to transpose the matrix.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle",
        "tirx.simdgroup_load",
        d,
        index,
        ptr,
        stride,
        col,
        row,
        transpose_matrix,
    )


def simdgroup_store(
    d: PrimExpr,
    index: PrimExpr,
    ptr: PrimExpr,
    stride: PrimExpr,
    col: int = 8,
    row: int = 8,
    transpose_matrix: bool = False,
):
    """Store data from simdgroup to device memory or threadgroup memory

    Parameters
    ----------
    d : PrimExpr
        The SIMDGroup.

    index : PrimExpr
        The index of the matrix.

    ptr : PrimExpr
        The pointer.

    stride : PrimExpr
        The stride.

    col : int
        The number of columns.

    row : int
        The number of rows.


    transpose_matrix : bool
        Whether to transpose the matrix.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle",
        "tirx.simdgroup_store",
        d,
        index,
        ptr,
        stride,
        col,
        row,
        transpose_matrix,
    )


def simdgroup_multiply_accumulate(
    d: Var,
    index_d: PrimExpr,
    a: Var,
    index_a: PrimExpr,
    b: Var,
    index_b: PrimExpr,
    c: Var,
    index_c: PrimExpr,
):
    """Multiply and accumulate two matrices in simdgroup
    i.e. d = a * b + c

    Parameters
    ----------
    d : Var
        The destination matrix.

    index_d : PrimExpr
        The index of the destination matrix.

    a : Var
        The first matrix.

    index_a : PrimExpr
        The index of the first matrix.

    b : Var
        The second matrix.

    index_b : PrimExpr
        The index of the second matrix.

    c : Var
        The third matrix.

    index_c : PrimExpr
        The index of the third matrix.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle",
        "tirx.simdgroup_multiply_accumulate",
        d,
        index_d,
        a,
        index_a,
        b,
        index_b,
        c,
        index_c,
    )


def cooperative_tensor_fill(
    d: Var,
    index: PrimExpr,
    value: PrimExpr,
    rows: int,
    cols: int,
):
    return call_intrin("handle", "tirx.cooperative_tensor_fill", d, index, value, rows, cols)


def cooperative_tensor_load(
    d: Var,
    index: PrimExpr,
    ptr: PrimExpr,
    stride: PrimExpr,
    rows: int,
    cols: int,
    transpose_matrix: bool = False,
    mma_M: int = 0,
    mma_N: int = 0,
    mma_K: int = 0,
    operand_role: int = 0,
):
    return call_intrin(
        "handle",
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
    )


def cooperative_tensor_store(
    d: PrimExpr,
    index: PrimExpr,
    ptr: PrimExpr,
    stride: PrimExpr,
    rows: int,
    cols: int,
    transpose_matrix: bool = False,
    mma_M: int = 0,
    mma_N: int = 0,
    mma_K: int = 0,
    operand_role: int = 0,
):
    return call_intrin(
        "handle",
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
    )


def cooperative_tensor_multiply_accumulate(
    d: Var,
    index_d: PrimExpr,
    a: Var,
    index_a: PrimExpr,
    b: Var,
    index_b: PrimExpr,
    c: Var,
    index_c: PrimExpr,
    M: int,
    N: int,
    K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
):
    return call_intrin(
        "handle",
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
    call : PrimExpr
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
    call : PrimExpr
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(dtype, "tirx.vectorcombine", vec1, vec2)


def dp4a(vec1, vec2, acc=0):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int32", "tirx.dp4a", vec1, vec2, acc)


def ret(val, span=None):
    """Create a tir return expression

    Parameters
    ----------
    val : Expr
        The returned tir expression, whose data type is int, float or void pointer.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    ret : PrimExpr
        The return expression
    """

    return _ffi_api.ret(val, span)


def any(*args, span=None):
    """Create a new experssion of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    val = _ffi_api._OpOr(args[0], args[1], span)  # type: ignore
    for i in range(2, len(args)):
        val = _ffi_api._OpOr(val, args[i], span)  # type: ignore
    return val


def all(*args, span=None):
    """Create a new expression of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    val = _ffi_api._OpAnd(args[0], args[1], span)  # type: ignore
    for i in range(2, len(args)):
        val = _ffi_api._OpAnd(val, args[i], span)  # type: ignore
    return val


@tvm_ffi.register_global_func("tvm.default_trace_action")
def _tvm_default_trace_action(*args):
    print(list(args))


def trace(args, trace_action="tvm.default_trace_action"):
    """Trace tensor data at the runtime.

    The trace function allows to trace specific tensor at the
    runtime. The tracing value should come as last argument.
    The trace action should be specified, by default
    tvm.default_trace_action is used.

    Parameters
    ----------
    args : list of Expr or Buffers.
        Positional arguments.

    trace_action : str.
        The name of the trace action.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    tvm.tirx.call_packed : Creates packed function.
    """
    if not isinstance(args, list):
        raise Exception("tvm.tirx.trace consumes the args as list type")
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    call_args.insert(0, trace_action)
    return tvm.tirx.Call(args[-1].dtype, Op.get("tirx.tvm_call_trace_packed"), call_args)


def min_value(dtype, span=None):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The minimum value of dtype.
    """
    return _ffi_api.min_value(dtype, span)  # type: ignore


def max_value(dtype: str, span: Span | None = None) -> Any:
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The maximum value of dtype.
    """
    return _ffi_api.max_value(dtype, span)  # type: ignore


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


def reinterpret(dtype, value, span: Span | None = None) -> Any:
    """infinity value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    value : PrimExpr
        The input value.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The reinterpret cast value of dtype.
    """
    return _ffi_api.reinterpret(dtype, value, span)  # type: ignore


def exp(x):
    """Take exponential of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.exp", x)


def exp2(x):
    """Calculate 2**x

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.exp2", x)


def exp10(x):
    """Calculate 10**x

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.exp10", x)


def erf(x):
    """Take gauss error function of the input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.erf", x)


def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.tanh", x)


def sigmoid(x):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.sigmoid", x)


def log(x):
    """Take log of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.log", x)


def log2(x):
    """Take log2 of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.log2", x)


def log10(x):
    """Take log10 of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.log10", x)


def log1p(x):
    """Take log(x + 1) with respect to input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.log1p", x)


def tan(x):
    """Take tan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = _require_float_arg("tan", x)
    return call_intrin(x.dtype, "tirx.tan", x)


def cos(x):
    """Take cos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = _require_float_arg("cos", x)
    return call_intrin(x.dtype, "tirx.cos", x)


def cosh(x):
    """Take cosh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.cosh", x)


def acos(x):
    """Take acos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.acos", x)


def acosh(x):
    """Take acos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.acosh", x)


def sin(x):
    """Take sin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = _require_float_arg("sin", x)
    return call_intrin(x.dtype, "tirx.sin", x)


def sinh(x):
    """Take sinh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.sinh", x)


def asin(x):
    """Take asin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.asin", x)


def asinh(x):
    """Take asinh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.asinh", x)


def atan(x):
    """Take atan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.atan", x)


def atanh(x):
    """Take atanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.atanh", x)


def atan2(x1, x2):
    """Take arctan2(x1, x2).

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(x1.dtype, "tirx.atan2", x1, x2)


def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.sqrt", x)


def rsqrt(x):
    """Take reciprocal of square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.rsqrt", x)


def clz(x):
    """Count leading zero bits of an integer x.

    Parameters
    ----------
    x : PrimExpr
        Input 32 or 64 bit integer.
        The result is undefined if the input is 0.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin("int32", "tirx.clz", x)


def floor(x: PrimExprWithOp, span=None):
    """Take floor of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.floor(x, span)  # type: ignore


def ceil(x, span=None):
    """Take ceil of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.ceil(x, span)  # type: ignore


def trunc(x, span=None):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.trunc(x, span)  # type: ignore


def abs(x, span=None):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.abs(x, span)  # type: ignore


def bitwise_and(x, y, span=None):
    """Take bitwise and of two values

    Parameters
    ----------
    x : PrimExpr
        Left operand

    y : PrimExpr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _ffi_api.bitwise_and(x, y, span)


def bitwise_not(x, span=None):
    """Take bitwise not of input value

    Parameters
    ----------
    x : PrimExpr
        Input operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _ffi_api.bitwise_not(x, span)


def bitwise_or(x, y, span=None):
    """Take bitwise or of two values

    Parameters
    ----------
    x : PrimExpr
        Left operand

    y : PrimExpr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _ffi_api.bitwise_or(x, y, span)


def bitwise_xor(x, y, span=None):
    """Take bitwise xor of two values

    Parameters
    ----------
    x : PrimExpr
        Left operand

    y : PrimExpr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : PrimExpr
        The result.
    """
    return _ffi_api.bitwise_xor(x, y, span)


def round(x, span=None):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
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
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.nearbyint(x, span)  # type: ignore


def nextafter(x1, x2):
    """Return the next floating-point value after x1 towards x2.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(x1.dtype, "tirx.nextafter", x1, x2)  # type: ignore


def hypot(x1, x2):
    """Equivalent to sqrt(x1**2 + x2**2), element-wise.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(x1.dtype, "tirx.hypot", x1, x2)  # type: ignore


def copysign(x1, x2):
    """Change the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(x1.dtype, "tirx.copysign", x1, x2)  # type: ignore


def ldexp(x1, x2):
    """Returns x1 * (2 ** x2).

    Parameters
    ----------
    x1 : PrimExpr
        Input argument.

    x2 : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x1 = tir.convert(x1)
    x2 = tir.convert(x2)
    return call_intrin(x1.dtype, "tirx.ldexp", x1, x2)  # type: ignore


def likely(cond, span=None):
    """Mark condition as likely.

    Parameters
    ----------

    cond : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The marked expression.
    """
    return _ffi_api.likely(cond, span)  # type: ignore


def filter(var, pred, *, span=None):  # pylint: disable=redefined-builtin
    """Thread-set filter escape hatch.

    Use this wrapper only when the predicate is *not* in the canonical
    thread-filter grammar (see ``src/tirx/analysis/filter_canonical.h``).
    Canonical predicates -- pure conjunctions of ``scopeid_var <op> const``
    comparisons plus bare ``Tx.ptx.elect_sync()`` calls -- are recognized by
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
    return call_intrin("bool", "tirx.filter", var, pred, span=span)


def selector(var, pred, span=None):
    """Analysis-only active-thread selector.

    ``selector(var, pred)`` denotes the unique value of ``var`` in the current
    active domain for which ``pred`` is true. It is intended for compiler
    metadata and should not survive to executable codegen.
    """
    return call_intrin(var.dtype, "tirx.selector", var, pred, span=span)


def isnan(x, span=None):
    """Check if input value is Nan.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.isnan(x, span)  # type: ignore


def isnullptr(x, span=None):
    """Check if input value is nullptr.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin("bool", "tirx.isnullptr", x, span=span)  # type: ignore


def isfinite(x, span=None):
    """Check if input value is finite.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.isfinite(x, span)  # type: ignore


def isinf(x, span=None):
    """Check if input value is infinite.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.isinf(x, span)  # type: ignore


def power(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _ffi_api._OpPow(x, y, span)  # type: ignore


def pow(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _ffi_api._OpPow(x, y, span)  # type: ignore


def popcount(x):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    x = tir.convert(x)
    return call_intrin(x.dtype, "tirx.popcount", x)


def q_multiply_shift(x, y, q, s):
    """Execute a multiplication between two Q-numbers x and y
    followed by a right shift s. The mathematical expression is:

       out = round(x*y*2^-s)

    More about Q-numbers here: https://en.wikipedia.org/wiki/Q_(number_format)
    The rounding rule is to the nearest value, rounding half up
    (i.e., round(x.1) = x and round (x.5) = x+1)

    Parameters
    ----------
    x : PrimExpr
        First Q-number
    y : PrimExpr
        Second Q-number
    q : PrimExpr
        Number of fractional bits in x and y. Needs to be > 0
    s : PrimExpr
        Integer shift

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_intrin("int32", "tirx.q_multiply_shift", x, y, q, s)


def q_multiply_shift_per_axis(
    x: PrimExpr,
    y: PrimExpr,
    ls: PrimExpr,
    rs: PrimExpr,
    q: IntImm,
    is_lshift_required: IntImm,
    is_rshift_required: IntImm,
):
    """Execute a multiplication between two Q-numbers x and y

    Parameters
    ----------
    x : PrimExpr
        First Q-number.
    y : PrimExpr
        Second Q-number.
    ls : PrimExpr
         Integer left shift.
    rs : PrimExpr
         Integer right shift.
    q : IntImm
        Number of fractional bits in x and y. Needs to be > 0.
    is_lshift_required : IntImm
                         Whether we need to do left shift or not.
    is_rshift_required : IntImm
                         Whether we need to do right shift or not.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return call_intrin(
        "int32",
        "tirx.q_multiply_shift_per_axis",
        x,
        y,
        ls,
        rs,
        q,
        is_lshift_required,
        is_rshift_required,
    )


def shift_left(x, y, span=None):
    """Return the result of x left shifted by y bits.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _ffi_api.left_shift(x, y, span)


def shift_right(x, y, span=None):
    """Return the result of x right shifted by y bits.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _ffi_api.right_shift(x, y, span)


def fmod(x, y):
    """Return the remainder of x divided by y with the same sign as x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.
    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    x = tir.convert(x)
    y = tir.convert(y)
    return call_intrin(x.dtype, "tirx.fmod", x, y)


def if_then_else(cond, t, f, span=None):
    """Conditional selection expression.

    Parameters
    ----------
    cond : PrimExpr
        The condition

    t : PrimExpr
        The result expression if cond is true.

    f : PrimExpr
        The result expression if cond is false.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    result : Node
        The result of conditional expression.

    Note
    ----
    Unlike Select, if_then_else will not execute
    the branch that does not satisfy the condition.
    You can use it to guard against out of bound access.
    Unlike Select, if_then_else cannot be vectorized
    if some lanes in the vector have different conditions.
    """
    return _ffi_api._OpIfThenElse(cond, t, f, span)  # type: ignore


def div(a, b, span=None):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b, span).
    """
    return _ffi_api._OpDiv(a, b, span)  # type: ignore


def indexdiv(a, b, span=None):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexDiv(a, b, span)  # type: ignore


def indexmod(a, b, span=None):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexMod(a, b, span)  # type: ignore


def truncdiv(a, b, span=None):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncDiv(a, b, span)  # type: ignore


def truncmod(a, b, span=None):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncMod(a, b, span)  # type: ignore


def floordiv(a, b, span=None):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorDiv(a, b, span)  # type: ignore


def logaddexp(a, b, span=None):
    """Compute the logaddexp of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpLogAddExp(a, b, span)  # type: ignore


def floormod(a, b, span=None):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorMod(a, b, span)  # type: ignore


def ceildiv(lhs, rhs, span=None):
    """Generic ceildiv operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of ceildiv operaton.
    """
    return _ffi_api._OpCeilDiv(lhs, rhs, span)  # type: ignore


def comm_reducer(fcombine, fidentity, name="reduce"):
    """Create a commutative reducer for reduction.

    Parameters
    ----------
    fcombine : function(Expr -> Expr -> Expr)
        A binary function which takes two Expr as input to return a Expr.

    fidentity : function(str -> Expr)
        A function which takes a type string as input to return a const Expr.

    Returns
    -------
    reducer : function
        A function which creates a reduce expression over axis.
        There are two ways to use it:

        1. accept (expr, axis, where) to produce an Reduce Expr on
           specified axis;
        2. simply use it with multiple Exprs.

    Example
    -------
    .. code-block:: python

        n = te.var("n")
        m = te.var("m")
        mysum = te.comm_reducer(lambda x, y: x+y,
            lambda t: tvm.tirx.const(0, dtype=t), name="mysum")
        A = te.placeholder((n, m), name="A")
        k = te.reduce_axis((0, m), name="k")
        B = te.compute((n,), lambda i: mysum(A[i, k], axis=k), name="B")
    """

    def _reduce_directly(*args):
        num = len(args)
        # process `where` is None
        if num == 3 and args[2] is None:
            num = 2
        res = args[0]
        for i in range(num - 1):
            res = fcombine(res, args[i + 1])
        return res

    def _make_reduce(expr, axis, where=None, init=None):
        code = fcombine.__code__
        assert fcombine.__code__.co_argcount == 2
        expr = tir.convert(expr)
        if init is not None:
            init = tir.convert(init)
        if isinstance(expr, Array):
            size = len(expr)
            lhs = []
            rhs = []
            dtypes = []
            for i in range(size):
                dtype = expr[i].dtype
                dtypes.append(dtype)
                lname = code.co_varnames[0] + "_" + str(i)
                lhs.append(Var(lname, dtype))
                rname = code.co_varnames[1] + "_" + str(i)
                rhs.append(Var(rname, dtype))
            if init is None:
                init = []
            result = fcombine(lhs, rhs)
            id_elem = fidentity(*dtypes)
        else:
            assert isinstance(expr, tvm.ir.PrimExpr)
            size = 1
            dtype = expr.dtype
            lvar = Var(code.co_varnames[0], dtype)
            rvar = Var(code.co_varnames[1], dtype)
            result = [fcombine(lvar, rvar)]
            id_elem = [fidentity(dtype)]
            lhs = [lvar]
            rhs = [rvar]
            expr = [expr]
            if init is not None:
                init = [init]
        combiner = CommReducer(lhs, rhs, result, id_elem)
        if not isinstance(axis, list | tuple | tvm.ir.Array):
            axis = [axis]
        if where is None:
            where = tir.convert(True)
        if init is None:
            outputs = tuple(
                tvm.tirx.Reduce(combiner, expr, axis, where, i, []) for i in range(size)
            )
        else:
            outputs = tuple(
                tvm.tirx.Reduce(combiner, expr, axis, where, i, init) for i in range(size)
            )
        return outputs[0] if size == 1 else outputs

    # pylint: disable=keyword-arg-before-vararg
    def reducer(expr, axis, where=None, init=None, *args):
        if isinstance(axis, tvm.tirx.IterVar | list | tuple):
            assert not args
            return _make_reduce(expr, axis, where, init)

        if where is None:
            assert not args
            assert init is None
            return _reduce_directly(expr, axis)
        elif init is None:
            assert not args
            return _reduce_directly(expr, axis, where)
        else:
            return _reduce_directly(expr, axis, where, init, *args)

    doc_str = """Create a {0} expression over axis.

              Parameters
              ----------
              expr : PrimExpr
                  The source expression.
              axis : IterVar
                  The reduction IterVar axis
              where : optional, Expr
                  Filtering predicate of the reduction.
              Returns
              -------
              value : PrimExpr
                  The result value.

              Example
              -------
              .. code-block:: python

                m = te.var("m")
                n = te.var("n")
                A = te.placeholder((m, n), name="A")
                k = te.reduce_axis((0, n), name="k")

                # there are two way to use this {0} reducer:
                # mode 1, accept (expr, axis, where) to produce an Reduce Expr
                # tvm.{0} represents tvm.te.{0} or tvm.tirx.{0}.
                B = te.compute((m,), lambda i: tvm.{0}(A[i, k], axis=k), name="B")

                # mode 2, simply use it with multiple Exprs:
                {0}_res = tvm.{0}(m, n)
              """
    reducer.__doc__ = doc_str.format(name)
    return reducer


def TVMBackendAllocWorkspace(device_type, device_id, nbytes, dtype_code_hint, dtype_bits_hint):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle",
        "tirx.TVMBackendAllocWorkspace",
        device_type,
        device_id,
        nbytes,
        dtype_code_hint,
        dtype_bits_hint,
    )


def TVMBackendFreeWorkspace(device_type, device_id, ptr):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int32", "tirx.TVMBackendFreeWorkspace", device_type, device_id, ptr)


def anylist_getitem(list_handle, index):
    """Returns an item from any list.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.anylist_getitem", list_handle, index)


def anylist_resetitem(list_handle, index):
    """Reset an item from any list.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int", "tirx.anylist_resetitem", list_handle, index)


def anylist_setitem_call_packed(list_handle, index, func_name, *args):
    """Set anylist item by result of packed call.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    func_name: str
        The name of the function to be called.
    args:
        Extra arguments
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "int", "tirx.anylist_setitem_call_packed", list_handle, index, func_name, *args
    )


def anylist_setitem_call_cpacked(list_handle, index, func_name, *args):
    """Set anylist item by result of packed call.
    list_handle: Var
        The handle to anylist
    index : int
        The index
    func_name: str
        The name of the function to be called.
    args:
        Extra arguments
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "int", "tirx.anylist_setitem_call_cpacked", list_handle, index, func_name, *args
    )


def vscale():
    """Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
    (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
    Returns
    -------
    call : PrimExpr
        Call to the vscale intrinsic
    """
    return call_intrin("int32", "tirx.vscale")


def get_active_lane_mask(dtype, base, limit):
    """
    Calculate a predicate mask given an upper bound (limit) and a current value (base).

    It will be lowered to the llvm.get.active.lane.mask intrinsic.
    (https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics)

    Parameters
    ----------
    dtype : str
        The data type of the result.

    base : PrimExpr
        An expression reprsenting the base.

    limit : PrimExpr
        An expression representing the limit.
    """
    return call_intrin(dtype, "tirx.get_active_lane_mask", base, limit)


def get_vscale_expr(dtype: str | tvm_ffi.dtype, min_size: int = 128) -> PrimExpr:
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


def ignore_loop_partition(predicate) -> PrimExpr:
    """
    Annotate a predicate not be considered as target condition of loop partition.

    Parameters
    ----------
    predicate : PrimExpr
        The annotated predicate expression.
    """
    return call_intrin("bool", "tirx.ignore_loop_partition", predicate)


# pylint: disable=unnecessary-lambda
sum = comm_reducer(lambda x, y: x + y, lambda t: const(0, dtype=t), name="sum")
min = comm_reducer(lambda x, y: _ffi_api._OpMin(x, y, None), max_value, name="min")  # type: ignore
max = comm_reducer(lambda x, y: _ffi_api._OpMax(x, y, None), min_value, name="max")  # type: ignore


########################################################
# CUDA native builtins
########################################################


def cuda_func_call(func_name, *args, source_code, return_type="void"):
    """TVM intrinsic to call a CUDA function. Source code is provided as a string.

    Parameters
    ----------
    func_name: str
        The name of the CUDA function.

    args: PrimExpr
        The arguments to the CUDA function.

    source_code: str
        The source code of the CUDA function.

    return_type: str
        The return type of the CUDA function.
    """
    return call_intrin(return_type, "tirx.cuda_func_call", func_name, *args, source_code)


def cuda_warp_reduce(value, op, width=32):
    """Warp-level butterfly shuffle-XOR reduction.

    Reduces ``value`` across ``width`` adjacent lanes using the specified
    operation.  Codegen emits ``log2(width)`` steps of
    ``__shfl_xor_sync(0xFFFFFFFF, val, mask)`` with descending XOR masks.

    Parameters
    ----------
    value : PrimExpr
        The per-thread scalar value to reduce.

    op : str
        Reduction operation: ``"sum"``, ``"max"``, or ``"min"``.

    width : int
        Number of lanes participating in each reduction group.
        Must be a power of two in [2, 32].  Defaults to 32 (full warp).

    Returns
    -------
    call : PrimExpr
        The reduced value (same dtype as *value*).
    """
    return call_intrin(value.dtype, "tirx.cuda_warp_reduce", value, op, width)


def cuda_warp_sum(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "sum", width)``."""
    return cuda_warp_reduce(value, "sum", width)


def cuda_warp_max(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "max", width)``."""
    return cuda_warp_reduce(value, "max", width)


def cuda_warp_min(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "min", width)``."""
    return cuda_warp_reduce(value, "min", width)


def cuda_cta_reduce(value, op, num_warps, scratch):
    """CTA-wide reduction via warp shuffle + shared memory.

    Two-step reduction: (1) intra-warp shuffle reduction, (2) warp-0
    collects per-warp partials from ``scratch``, reduces, broadcasts via
    ``__syncthreads()``.  All CTA threads must participate.

    Parameters
    ----------
    value : PrimExpr
        Per-thread scalar value to reduce.

    op : str
        Reduction operation: ``"sum"``, ``"max"``, or ``"min"``.

    num_warps : int
        Number of warps in the CTA.  Must be a power of two in [1, 32].

    scratch : Var
        Data pointer to shared-memory scratch space (>= num_warps elements).

    Returns
    -------
    call : PrimExpr
        The reduced value broadcast to all threads (same dtype as *value*).
    """
    return call_intrin(value.dtype, "tirx.cuda_cta_reduce", value, op, num_warps, scratch)


def cuda_cta_sum(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "sum", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "sum", num_warps, scratch)


def cuda_cta_max(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "max", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "max", num_warps, scratch)


def cuda_cta_min(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "min", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "min", num_warps, scratch)


def cuda_copy_bytes(dst, src, num_bytes):
    """Typed load/store copy of ``num_bytes`` bytes.

    Copies ``num_bytes`` bytes from ``src`` to ``dst`` using a single
    typed load/store instruction.  Codegen selects the appropriate C++
    vector type (``uint4``, ``uint2``, ``unsigned int``, etc.).

    Parameters
    ----------
    dst : Var
        Destination pointer.

    src : Var
        Source pointer.

    num_bytes : int
        Number of bytes to copy.  Must be one of {1, 2, 4, 8, 16}.

    Returns
    -------
    call : PrimExpr
        A void call expression.
    """
    return call_intrin("void", "tirx.cuda_copy_bytes", dst, src, num_bytes)


def cuda_copy_128b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 16)`` — copies 128 bits."""
    return cuda_copy_bytes(dst, src, 16)


def cuda_copy_64b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 8)`` — copies 64 bits."""
    return cuda_copy_bytes(dst, src, 8)


def cuda_copy_32b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 4)`` — copies 32 bits."""
    return cuda_copy_bytes(dst, src, 4)


def cuda_copy_16b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 2)`` — copies 16 bits."""
    return cuda_copy_bytes(dst, src, 2)


def cuda_copy_8b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 1)`` — copies 8 bits."""
    return cuda_copy_bytes(dst, src, 1)


def cuda_warp_sync():
    """TVM intrinsic to synchronize threads within the current warp.

    This lowers to a CUDA `__syncwarp()` call.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_warp_sync")


def cuda_cta_sync():
    """TVM intrinsic to call CUDA syncthreads (block-wide barrier)

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_cta_sync")


def cuda_grid_sync():
    """TVM intrinsic to call CUDA grid-wide sync (cooperative groups)

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_grid_sync")


def cuda_cluster_sync():
    """TVM intrinsic to call CUDA cluster-wide barrier sync

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_cluster_sync")


def cuda_thread_rank():
    """TVM intrinsic that returns ``cooperative_groups::thread_rank()``
    for the enclosing CTA -- the linear thread index within the block.

    Useful for building "single thread of CTA" predicates without
    referencing user-declared scope_id vars. For example, the idiomatic
    mbarrier.init leader predicate is::

        Tx.cuda.thread_rank() == 0

    Returns
    -------
    call : PrimExpr
        The call expression (``int32``).
    """
    return call_intrin("int32", "tirx.cuda_thread_rank")


def cuda_half2float(src):
    """TVM intrinsic to convert half to float

    Parameters
    ----------
    src : PrimExpr
        Source pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("float32", "tirx.cuda_half2float", src)


def cuda_bfloat162float(src):
    """TVM intrinsic to convert bfloat16 to float

    Parameters
    ----------
    src : PrimExpr
        Source pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("float32", "tirx.cuda_bfloat162float", src)


def cuda_float22half2(dst, src):
    """TVM intrinsic to convert float2 to half2 with rounding

    Parameters
    ----------
    dst : PrimExpr
        Destination pointer.

    src : PrimExpr
        Source pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_float22half2", dst, src)


def cuda_trap_when_assert_failed(cond):
    """TVM intrinsic to trap when assertion failed (cond == false)

    Parameters
    ----------
    cond : PrimExpr
        Condition to check.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_trap_when_assert_failed", cond)


def cuda_runtime_instr_desc(desc, sf_id):
    """TVM intrinsic to update runtime instruction descriptor

    Parameters
    ----------
    desc : PrimExpr
        Pointer to the descriptor (uint32*).

    sf_id : PrimExpr
        The subfragment id.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_runtime_instr_desc", desc, sf_id)


def cuda_half8tofloat8(src_addr, dst_addr):
    """TVM intrinsic to convert 8 half2s to 8 float2s

    Parameters
    ----------
    src_addr : PrimExpr
        Source pointer.

    dst_addr : PrimExpr
        Destination pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_half8tofloat8", src_addr, dst_addr)


def cuda_float8tohalf8(src_addr, dst_addr):
    """TVM intrinsic to convert 8 float2s to 8 half2s

    Parameters
    ----------
    src_addr : PrimExpr
        Source pointer.

    dst_addr : PrimExpr
        Destination pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_float8tohalf8", src_addr, dst_addr)


def tvm_load_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle", "tirx.tvm_load_matrix_sync", fragment, m, n, k, index, buffer_ptr, stride, layout
    )


def tvm_mma_sync(
    fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle",
        "tirx.tvm_mma_sync",
        fragment_d,
        index_d,
        fragment_a,
        index_a,
        fragment_b,
        index_b,
        fragment_c,
        index_c,
    )


def tvm_bmma_sync(
    fragment_d, index_d, fragment_a, index_a, fragment_b, index_b, fragment_c, index_c
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle",
        "tirx.tvm_bmma_sync",
        fragment_d,
        index_d,
        fragment_a,
        index_a,
        fragment_b,
        index_b,
        fragment_c,
        index_c,
    )


def tvm_fill_fragment(fragment, m, n, k, index, value):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tirx.tvm_fill_fragment", fragment, m, n, k, index, value)


def tvm_store_matrix_sync(fragment, m, n, k, index, buffer_ptr, stride, layout):
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "handle", "tirx.tvm_store_matrix_sync", fragment, m, n, k, index, buffer_ptr, stride, layout
    )


def ptx_mma_sp(
    dtype,
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
    metadata,
    meta_index,
    sparse_selector,
    saturate,
):
    """TVM intrinsic for sparse tensor core ptx instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of multiplicand fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment B.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    metadata : Expr
        The metadata of operand.

    meta_index : Expr
        The metadata index of operand.

    sparse_selector : Expr
        The sparse selector indicating the thread that stores the metadata.

    saturate : bool
        The optional saturation at the output.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        "tirx.ptx_mma_sp",
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
        metadata,
        meta_index,
        sparse_selector,
        saturate,
    )


def mma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """TVM intrinsic for storing the result of PTX MMA into a destination pointer

    Parameters
    ----------
    dtype : str
        The data type of the result.

    m : IntImm
        The shape of mma fragment.

    n : IntImm
        The shape of mma fragment.

    dst_ptr : Var
        The destination pointer variable.

    src_ptr : Var
        The source pointer variable.

    src_offset : Expr
        The source offset.

    dst_stride : Var
        The destination stride.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(dtype, "tirx.mma_store", m, n, dst_ptr, src_ptr, src_offset, dst_stride)


def mma_store_legacy(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """mma_store with apache-style signature.

    ``dst_ptr`` is typically a ``tvm_access_ptr`` Call (so the caller can
    encode the destination's element dtype + base offset), and
    ``src_ptr + src_offset`` is the raw warp accumulator + element offset.
    Codegen does ``ptr + offset`` C pointer arithmetic; lower_warp_memory
    rewrites src_offset's group component to a thread-local index."""
    return call_intrin(
        dtype,
        "tirx.mma_store_legacy",
        m,
        n,
        dst_ptr,
        src_ptr,
        src_offset,
        dst_stride,
    )


def mma_fill(dtype, local_size, local_ptr, offset):
    """TVM intrinsic for zero-initalizing an MMA accumulation registor

    Parameters
    ----------
    dtype : str
        The data type of the result.

    local_size : IntImm
        The number of elements.

    local_ptr : Var
        The destination pointer variable.

    offset : Expr
        The destination offset.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(dtype, "tirx.mma_fill", local_size, local_ptr, offset)


def mma_fill_legacy(dtype, local_size, local_ptr, offset):
    """mma_fill with (ptr_var, offset). Codegen emits ``ptr + offset``
    C pointer arithmetic; lower_warp_memory rewrites the offset's group
    component to a thread-local index."""
    return call_intrin(dtype, "tirx.mma_fill_legacy", local_size, local_ptr, offset)


def ptx_cp_async_bulk(
    dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes, barrier_id
):
    """TVM intrinsic for ptx async copy from global to shared memory using cp.async.bulk
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

    Parameters
    ----------
    dtype : str
       The data type of the result.

    shared_ptr : Var
        The shared memory pointer variable.

    shared_offset : Expr
        The offset of shared memory pointer.

    global_ptr : Var
        The global memory pointer variable.

    global_offset : Expr
        The offset of global memory pointer.

    bytes : int
        The data size to copy.

    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        "tirx.ptx_cp_async_bulk",
        shared_ptr,
        shared_offset,
        global_ptr,
        global_offset,
        bytes,
        barrier_id,
    )


def ptx_cp_async_bulk_shared_to_cluster(dst_ptr, src_ptr, size, mbar):
    """PTX cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes

    Asynchronous bulk copy from executing CTA's shared memory to a remote
    CTA's shared memory within the same cluster.

    Parameters
    ----------
    dst_ptr : PrimExpr
        Destination pointer in shared::cluster address space (remote CTA).

    src_ptr : PrimExpr
        Source pointer in shared::cta address space (local CTA).

    size : PrimExpr
        Number of bytes to copy (must be multiple of 16).

    mbar : PrimExpr
        Mbarrier address in shared::cluster space for completion signaling,
        usually produced by ``Tx.ptx.map_shared_rank``.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_cp_async_bulk_shared_to_cluster", dst_ptr, src_ptr, size, mbar)


def ptx_cp_async_mbarrier_arrive(barrier_id):
    """TVM intrinsic for ptx async copy barrier using cp.async.mbarrier.arrive
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_cp_async_mbarrier_arrive", barrier_id)


def ptx_fence(sem: str, scope: str):
    """TVM intrinsic for PTX fence instruction.

    Generates: fence.{sem}.{scope};

    Parameters
    ----------
    sem : str
        The semantics of the fence. One of "sc", "acq_rel".
    scope : str
        The scope of the fence. One of "cta", "cluster", "gpu", "sys".

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("sem", sem, _FENCE_SEM)
    _choice("scope", scope, _FENCE_SCOPE)
    return call_intrin("", "tirx.ptx_fence", sem, scope)


def ptx_fence_proxy_async(space: str = ""):
    """TVM intrinsic for PTX fence.proxy.async instruction.

    Generates: fence.proxy.async[.{space}];

    Parameters
    ----------
    space : str
        The address space qualifier. One of "", "global", "shared::cta", "shared::cluster".
        Empty string means no qualifier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("space", space, _FENCE_PROXY_ASYNC_SPACE)
    return call_intrin("", "tirx.ptx_fence_proxy_async", space)


def ptx_mbarrier_init(bar, thread_count):
    """TVM intrinsic to call mbarrier.init.shared::cta.b64

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    thread_count : int
        The number of threads expected to arrive at the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_mbarrier_init", bar, thread_count)


def ptx_mbarrier_arrive(bar, cta_id=None, pred=None):
    """TVM intrinsic to call
        mbarrier.arrive.shared::cta.b64
    or
        @p mapa.shared::cluster.u32
        @p mbarrier.arrive.shared::cluster.b64

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    cta_id : Optional[PrimExpr]
        The cta id.

    pred : Optional[PrimExpr]
        The predicate to guard the operation.
    """
    if cta_id is None and pred is None:
        return call_intrin("", "tirx.ptx_mbarrier_arrive", bar)
    assert cta_id is not None and pred is not None
    return call_intrin("", "tirx.ptx_mbarrier_arrive", bar, cta_id, pred)


def ptx_mbarrier_arrive_expect_tx(bar, byte_count, cta_id=None, pred=None):
    """TVM intrinsic to call
        mbarrier.arrive_expect_tx.shared::cta.b64
    or
        @p mapa.shared::cluster.u32
        @p mbarrier.arrive_expect_tx.shared::cluster.b64

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    byte_count : int
        Increases the tx count of the mbarrier object to track completion of
        addtional async transactions.

    cta_id : Optional[PrimExpr]
        The cta id.

    pred : Optional[PrimExpr]
        The predicate to guard the operation.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if cta_id is None and pred is None:
        return call_intrin("", "tirx.ptx_mbarrier_arrive_expect_tx", bar, byte_count)
    assert cta_id is not None and pred is not None
    return call_intrin("", "tirx.ptx_mbarrier_arrive_expect_tx", bar, byte_count, cta_id, pred)


def ptx_mbarrier_try_wait(bar, phase):
    """TVM intrinsic to call mbarrier.try_wait.parity repeatedly until it returns true

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    phase : int
        The phase of the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_mbarrier_try_wait", bar, phase)


def ptx_mbarrier_try_wait_once(bar, phase, ticks):
    """TVM intrinsic for one-shot non-blocking ``mbarrier.try_wait.parity``.

    Returns ``1`` if the requested parity has been reached and ``0`` otherwise.
    This is intended for bounded debug waits; production waits should use
    :func:`ptx_mbarrier_try_wait`.
    """
    return call_intrin("uint32", "tirx.ptx_mbarrier_try_wait_once", bar, phase, ticks)


def ptx_bar_arrive(name_bar_id, thread_count):
    """TVM intrinsic to call bar.arrive a, b

    Parameters
    ----------
    name_bar_id : int
        The ID of the named barrier.

    thread_count : int
        The number of threads expected to arrive at the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_bar_arrive", name_bar_id, thread_count)


def ptx_bar_sync(name_bar_id, thread_count):
    """TVM intrinsic to call bar.sync a, {b}

    Parameters
    ----------
    name_bar_id : int
        The ID of the named barrier.

    thread_count : int
        The number of threads expected to arrive at the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_bar_sync", name_bar_id, thread_count)


def ptx_cp_async(
    dst_ptr,
    src_ptr,
    cp_size,
    *,
    cache_hint="",
    cache_policy=None,
    prefetch_size=-1,
    predicate=-1,
    fill_mode="",
):
    """TVM intrinsic for ptx async copy from global to shared memory using cp.async
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    Dispatches to one of three PTX-form-aligned ops:

    * ``ptx_cp_async_src_size`` for ``fill_mode == "zero"`` (zero-fill via
      ``src_size = pred ? cp_size : 0``).
    * ``ptx_cp_async_ignore_src`` for a non-empty ``predicate`` with no
      fill_mode (``setp+@p`` guards the asm).
    * ``ptx_cp_async_plain`` for the no-predicate / no-fill_mode case.

    Parameters
    ----------
    shared_ptr : PrimExpr
        The pointer to the shared memory.

    global_ptr : PrimExpr
        The pointer to the global memory.

    cp_size : int
        The data size to copy.

    cache_hint : str["evict_last", "evict_first", "evict_normal", ""]
        The cache hint.

    prefetch_size : int[-1, 64, 128, 256]
        The prefetch size.

    predicate : PrimExpr
        The predicate to guard the operation.

    fill_mode : str["zero", ""]
        The fill mode.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    _choice("prefetch_size", prefetch_size, _CP_ASYNC_PREFETCH_SIZE)
    _choice("fill_mode", fill_mode, _CP_ASYNC_FILL_MODE)
    return call_intrin(
        "",
        "tirx.ptx_cp_async",
        dst_ptr,
        src_ptr,
        cp_size,
        cache_policy,
        int(has_cache_policy),
        prefetch_size,
        predicate,
        fill_mode,
    )


def ptx_cp_async_legacy(*all_args):
    """Legacy ``ptx_cp_async`` API taking explicit src/dst offsets.

    Signature: ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size)``.
    Offsets are folded into the pointers via ``tvm_access_ptr`` then
    dispatched to fork-native :func:`ptx_cp_async`.

    ``T.ptx.cp_async_legacy`` runs through ``_dtype_forward`` which
    prepends a ``dtype=`` kwarg as a leading positional. The dtype names
    the *element* type of the buffer (offsets are in elements of that
    dtype, not bytes), so this function accepts either 5 or 6 positional
    args.
    """
    args = list(all_args)
    elem_dtype = "int8"
    if len(args) == 6:
        # Leading positional is the buffer element dtype, used to scale
        # offsets correctly when folding via ``tvm_access_ptr``.
        elem_dtype = args.pop(0)
    if len(args) != 5:
        raise ValueError(
            f"ptx_cp_async_legacy expects 5 args (or 6 with dtype= kwarg "
            f"prepended); got {len(all_args)}"
        )
    dst_ptr, dst_offset, src_ptr, src_offset, cp_size = args
    dst_ptr = tvm_access_ptr(elem_dtype, dst_ptr, dst_offset, 1, 1)
    src_ptr = tvm_access_ptr(elem_dtype, src_ptr, src_offset, 1, 1)
    return ptx_cp_async(dst_ptr, src_ptr, cp_size)


def ptx_cp_async_commit_group():
    """TVM intrinsic for ptx async copy commit
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_cp_async_commit_group")


def ptx_cp_async_wait_group(num=0):
    """TVM intrinsic for ptx async copy wait
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group

    Parameters
    ----------
    num : int, optional
        The number of the most recent uncommitted pending cp.async groups to wait.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_cp_async_wait_group", num)


def ptx_cp_async_bulk_tensor_global_to_cluster(
    dim, dst_ptr, bar, tensormap_addr, cta_mask, cta_group, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.async.bulk.tensor.dim.shared::cluster.global.tile.mbarrier::complete_tx::bytes

    Parameters
    ----------
    dim : int
        The dimension of the source tensor.

    dst_ptr : PrimExpr
        The destination pointer to the shared memory.

    bar : PrimExpr
        The pointer to mbarrier variable.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cta_mask : int
        The mask of the cta for multicast.

    cta_group : int
        Must be either 1 or 2.
        If set to 1, mbarrier must be in the shared memory of the same CTA as the shared memory destination
        If set to 2, mbarrier can be in shared memory of either the same CTA as the shared memory destination
                     or the shared memory of the peer CTA.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        specifies the starting coordinates in the tensor data in the global memory

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx_cp_async_bulk_tensor_global_to_cluster",
            dim,
            dst_ptr,
            bar,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_tensor_global_to_cluster",
        dim,
        dst_ptr,
        bar,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster(
    dim, dst_ptr, bar, tensormap_addr, cta_mask, cta_group, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call
    cp.async.bulk.tensor.dim.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes

    Parameters
    ----------
    dim : int
        The dimension of the source tensor.

    dst_ptr : PrimExpr
        The destination pointer to the shared memory.

    bar : PrimExpr
        The pointer to mbarrier variable.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cta_mask : int
        The mask of the cta for multicast.

    cta_group : int
        Must be either 1 or 2.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        The TMA coordinates followed by the 4 gather row indices.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster",
            dim,
            dst_ptr,
            bar,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster",
        dim,
        dst_ptr,
        bar,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_shared_to_global(
    dim, src_ptr, tensormap_addr, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.async.bulk.tensor.dim.global.shared::cta.tile.bulk_group

    Parameters
    ----------
    dim : int
        The dimension of the copy tensor.

    src_ptr : PrimExpr
        The source pointer to the shared memory.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        specifies the starting coordinates in the tensor data in the global memory

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx_cp_async_bulk_tensor_shared_to_global",
            dim,
            src_ptr,
            tensormap_addr,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_tensor_shared_to_global",
        dim,
        src_ptr,
        tensormap_addr,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_global_to_cluster_prefetch(
    dim, tensormap_addr, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.async.bulk.prefetch.tensor.dim.L2.global.tile

    Parameters
    ----------
    dim : int
        The dimension of the source tensor.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        specifies the starting coordinates in the tensor data in the global memory

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch",
            dim,
            tensormap_addr,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch",
        dim,
        tensormap_addr,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_shared_to_global_reduce(
    dim, src_ptr, tensormap_addr, cache_hint, red_op, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.reduce.async.bulk.tensor.dim.dst.src.redOp

    Parameters
    ----------
    dim : int
        The dimension of the copy tensor.

    src_ptr : PrimExpr
        The source pointer to the shared memory.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cache_hint: str
        The cache hint.

    red_op: str
        The reduction operator.

    coords: List[PrimExpr]
        The coordinates of the tensor.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy = red_op
        red_op, *coords = coords
        _choice("red_op", red_op, _CP_ASYNC_BULK_RED_OP)
        return call_intrin(
            "",
            "tirx.ptx_cp_async_bulk_tensor_shared_to_global_reduce",
            dim,
            src_ptr,
            tensormap_addr,
            cache_hint,
            has_cache_policy,
            red_op,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    _choice("red_op", red_op, _CP_ASYNC_BULK_RED_OP)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_tensor_shared_to_global_reduce",
        dim,
        src_ptr,
        tensormap_addr,
        cache_policy,
        int(has_cache_policy),
        red_op,
        *coords,
    )


def ptx_cp_async_bulk_commit_group():
    """TVM intrinsic to call cp.async.bulk.tensor.commit_group

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_cp_async_bulk_commit_group")


def ptx_cp_async_bulk_wait_group(n=0, read=True):
    """TVM intrinsic to call cp.async.bulk.tensor.wait_group

    Parameters
    ----------
    n : int
        The number of the most recent uncommitted pending cp.async groups to wait.

    read : bool
        Whether the wait is for read.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_cp_async_bulk_wait_group", n, read)


def ptx_barrier_cluster_arrive(sem="", aligned=True):
    """TVM intrinsic to call barrier.cluster.arrive{.sem}{.aligned}

    Parameters
    ----------
    sem : str
        Either release or relaxed or empty string.

    aligned : bool
        Whether all threads in the warp must execute the same instruction.
    """
    _choice("sem", sem, _CLUSTER_BARRIER_SEM)
    return call_intrin("", "tirx.ptx_barrier_cluster_arrive", sem, aligned)


def ptx_barrier_cluster_wait(acquire=False, aligned=True):
    """TVM intrinsic to call barrier.cluster.wait{.acquire}{.aligned}

    Parameters
    ----------
    acquire : bool
        The memory synchronization

    aligned : bool
        Whether all threads in the warp must execute the same instruction.
    """
    return call_intrin("", "tirx.ptx_barrier_cluster_wait", acquire, aligned)


def ptx_elect_sync():
    """TVM intrinsic to call elect.sync"""
    return call_intrin("uint32", "tirx.ptx_elect_sync")


def ptx_fence_mbarrier_init():
    """TVM intrinsic for PTX fence.mbarrier_init.release.cluster instruction.

    Generates: fence.mbarrier_init.release.cluster;

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_fence_mbarrier_init")


def ptx_fetch_register(bits, reg_name):
    """TVM intrinsic to tvm instrinsics to fetch PTX pre-defined registers

    Parameters
    ----------
    bits : int
        The number of bits of the register.

    reg_name : str
        The name of the register.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int" + str(bits), "tirx.ptx_fetch_register", bits, reg_name)


def ptx_mma(
    shape,
    a_layout,
    b_layout,
    d_type,
    a_type,
    b_type,
    c_type,
    d_ptrs,
    a_ptrs,
    b_ptrs,
    c_ptrs=None,
    saturate=False,
    bit_op=None,
):
    """TVM intrinsic for ptx tensor core mma instructions.
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

    Each per-thread register of every operand is addressed by its OWN pointer
    (one ``void*`` per b32/f32 register), so the register fragments need not be
    contiguous in the register file. ``d_ptrs`` / ``a_ptrs`` / ``b_ptrs`` /
    ``c_ptrs`` are lists of one pointer per 32-bit register (b32 for
    fp16/bf16/tf32/int8 multiplicands, f32/f64 for the accumulator), enumerated
    in the fixed PTX register order (see the gemm dispatch /
    ``tests/python/tirx-base/test_tir_ptx_mma.py``).

    Within one b32 register the packed elements (e.g. 2 fp16 along k_pack)
    must stay contiguous (stride 1); only the b32 registers themselves may be
    scattered.

    Parameters
    ----------
    shape : str
        The shape of mma fragment.

    a_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    b_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    d_type : str
        The data type of result fragment D.

    a_type : str
        The data type of multiplicand fragment A.

    b_type : str
        The data type of multiplicand fragment B.

    c_type : str
        The data type of accumulator fragment C.

    d_ptrs : List[PrimExpr]
        One pointer per result-fragment D register, in PTX order.

    a_ptrs : List[PrimExpr]
        One pointer per multiplicand-A register, in PTX order.

    b_ptrs : List[PrimExpr]
        One pointer per multiplicand-B register, in PTX order.

    c_ptrs : Optional[List[PrimExpr]]
        One pointer per accumulator-C register, in PTX order. ``None`` (the
        default) means the accumulator is not used (beta == 0): codegen feeds
        a literal 0 for each C slot.

    saturate : bool
        The optional saturation at the output.

    bit_op : Optional[Literal["xor", "and"]]
        The 1-bit operator (for the b1 subbyte form). ``None`` means unused.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    d_ptrs = list(d_ptrs)
    a_ptrs = list(a_ptrs)
    b_ptrs = list(b_ptrs)
    has_c = c_ptrs is not None
    c_ptrs = list(c_ptrs) if has_c else []

    # Encode group register counts as leading attrs so codegen can slice the
    # flat pointer tail. ``no_c_ptr`` mirrors the legacy IntImm(0) sentinel.
    no_c_ptr = not has_c
    # Flattened pointer list: D regs, A regs, B regs, then C regs (if any).
    ptrs = [*d_ptrs, *a_ptrs, *b_ptrs, *c_ptrs]

    base = [
        "",
        "tirx.ptx_mma",
        shape,
        a_layout,
        b_layout,
        d_type,
        a_type,
        b_type,
        c_type,
        len(d_ptrs),
        len(a_ptrs),
        len(b_ptrs),
        len(c_ptrs),
        no_c_ptr,
        *ptrs,
        saturate,
    ]
    if bit_op is None:
        return call_intrin(*base)
    return call_intrin(*base, bit_op)


def ptx_mma_legacy(*all_args, operator=None):
    """Legacy ``ptx_mma`` API.

    Signature: ``(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
    multiplicand_a, a_index, multiplicand_b, b_index, accumulator,
    c_index, saturate, operator=None)``. The accumulator is reused as
    both input and output (no separate ``d``/``c`` slot), unlike
    fork-native :func:`ptx_mma` which distinguishes them. Translation:

    * ``a_dtype, b_dtype, c_dtype`` → fork ``a_type, b_type, c_type``
      (and reuse ``c_dtype`` as fork ``d_type`` since the accumulator
      dtype is the output dtype here).
    * ``(a_ptr, a_offset)`` and ``(b_ptr, b_offset)`` → folded via
      :func:`tvm_access_ptr`.
    * ``(accumulator, c_index)`` → folded; passed for both ``d_ptr`` and
      ``c_ptr`` since the accumulator is reused as the output.

    ``T.ptx.mma.legacy`` runs through ``_dtype_forward`` which prepends a
    ``dtype=`` kwarg as a leading positional, so this function accepts
    either 13 or 14 positional args.
    """
    args = list(all_args)
    # ``T.ptx.mma.legacy(..., dtype="...")`` has the dtype prepended by
    # ``_dtype_forward``; strip it here.
    if len(args) in (14, 15):
        _ = args.pop(0)
    if len(args) == 14:
        # operator passed positionally as the trailing arg.
        operator = args.pop()
    if len(args) != 13:
        raise ValueError(
            f"ptx_mma_legacy expects 13-15 positional args (with optional "
            f"leading ``call_dtype`` from dtype= kwarg and optional trailing "
            f"``operator``); got {len(all_args)}"
        )
    (
        shape,
        a_layout,
        b_layout,
        a_dtype,
        b_dtype,
        c_dtype,
        a_ptr,
        a_offset,
        b_ptr,
        b_offset,
        acc_ptr,
        c_offset,
        saturate,
    ) = args
    # Emit tirx.ptx_mma_legacy directly with separate (ptr_var, offset)
    # pairs. codegen_cuda.cc uses C pointer arithmetic ``ptr + offset``
    # so element offsets stay element-accurate, and lower_warp_memory
    # rewrites the offset's group component to a thread-local index.
    call_args = [
        shape,
        a_layout,
        b_layout,
        a_dtype,
        b_dtype,
        c_dtype,
        a_ptr,
        a_offset,
        b_ptr,
        b_offset,
        acc_ptr,
        c_offset,
        saturate,
    ]
    if operator is not None:
        call_args.append(operator)
    return call_intrin("", "tirx.ptx_mma_legacy", *call_args)


def ptx_mma_sp_legacy(*all_args):
    """Legacy ``ptx_mma_sp`` API.

    Signature: ``(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
    multiplicand_a, a_index, multiplicand_b, b_index, accumulator,
    c_index, metadata, meta_index, sparse_selector, saturate)``.

    ``T.ptx.mma_sp.legacy`` runs through ``_dtype_forward`` which prepends
    a ``dtype=`` kwarg as a leading positional, so this function accepts
    either 16 or 17 positional args.
    """
    args = list(all_args)
    if len(args) == 17:
        _ = args.pop(0)
    if len(args) != 16:
        raise ValueError(
            f"ptx_mma_sp_legacy expects 16 args (or 17 with dtype= kwarg "
            f"prepended); got {len(all_args)}"
        )
    (
        shape,
        a_layout,
        b_layout,
        a_dtype,
        b_dtype,
        c_dtype,
        a_ptr,
        a_offset,
        b_ptr,
        b_offset,
        acc_ptr,
        c_offset,
        meta_ptr,
        meta_offset,
        sparse_selector,
        saturate,
    ) = args
    return ptx_mma_sp(
        c_dtype,
        shape,
        a_layout,
        b_layout,
        a_dtype,
        b_dtype,
        c_dtype,
        a_ptr,
        a_offset,
        b_ptr,
        b_offset,
        acc_ptr,
        c_offset,
        meta_ptr,
        meta_offset,
        sparse_selector,
        saturate,
    )


def ptx_ldmatrix(trans, num, dtype, smem_ptr, *dst_handles):
    """TVM intrinsic for ldmatrix.sync.aligned.m8n8.x{num}{.trans}.shared.{dtype}.

    Mirrors the PTX ISA destination form: each output register is a separate
    operand. Pass ``Tx.address_of(buf[idx])`` (or ``buf.ptr_to([idx])``) for
    each destination — the slots may be non-contiguous.

    Parameters
    ----------
    trans : bool
        Apply the ``.trans`` modifier.
    num : int
        One of 1, 2, 4 — number of m8n8 fragments.
    dtype : str
        ``"b16"`` (4 bytes per fragment register) or ``"b8"`` (2 bytes per).
    smem_ptr : PrimExpr
        Generic pointer to source shared memory.
    *dst_handles : PrimExpr
        N pointer-to-uint32 destinations, where
        ``N = num if dtype == "b16" else num // 2``.

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
    """
    _choice("num", num, _LDMATRIX_NUM)
    _choice("dtype", dtype, _LDMATRIX_DTYPE)
    # _LDMATRIX_DTYPE entries carry leading dot (".b16" / ".b8").
    dtype_bare = dtype.lstrip(".") if isinstance(dtype, str) else dtype
    n_regs = int(num) if dtype_bare == "b16" else int(num) // 2
    if len(dst_handles) != n_regs:
        raise ValueError(
            f"ldmatrix .x{int(num)}.{dtype_bare} expects {n_regs} destination "
            f"handles, got {len(dst_handles)}"
        )
    return call_intrin("", "tirx.ptx_ldmatrix", trans, num, dtype, smem_ptr, *dst_handles)


_PTX_TO_NUMPY_DTYPE = {
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "bf16": "bfloat16",
    "tf32": "float32",
    "s8": "int8",
    "u8": "uint8",
    "s32": "int32",
    "s4": "int4",
    "u4": "uint4",
    "b1": "int1",
    "b16": "uint16",
    "e4m3": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
}


def _ptx_to_numpy_dtype(dtype_str):
    """Map a PTX-abbreviation or numpy dtype string to a numpy dtype string
    suitable for ``tvm_access_ptr`` (which scales the offset by the element
    bit width). Unknown strings pass through unchanged so a caller may also
    pass an already-numpy dtype."""
    s = dtype_str if isinstance(dtype_str, str) else str(dtype_str)
    return _PTX_TO_NUMPY_DTYPE.get(s, s)


def _wrap_or_fold_access_ptr(ptr, offset, elem_dtype):
    """Wrap ``ptr`` with ``tvm_access_ptr`` unless it already is one.

    Several s_tir tensor intrinsics already pass ``buffer.access_ptr(...)``
    (an ``tvm_access_ptr`` Call) for the pointer argument. Naively wrapping
    that again yields a nested ``tvm_access_ptr(... access_ptr(...) ...)``
    whose ``args[1]`` is a Call rather than a Var, which crashes the
    lowering rule (Downcast<Var> at intrin_rule.cc) and several s_tir
    passes that assume a raw buffer var. Detect that case and fold the
    outer offset into the inner one.
    """
    from tvm.ir import Op  # local import to avoid cycles

    is_access_ptr_call = (
        isinstance(ptr, Call) and isinstance(ptr.op, Op) and ptr.op.name == "tirx.tvm_access_ptr"
    )
    if is_access_ptr_call:
        # Inner Call already wraps the buffer var. Reuse its inner var and
        # inner element dtype (the marker type_annotation), and add the
        # outer offset (which is in `elem_dtype` units, same convention as
        # the inner since both come from the same buffer).
        inner_args = ptr.args
        inner_marker = inner_args[0]
        inner_var = inner_args[1]
        inner_offset = inner_args[2]
        rw_mask = inner_args[4]
        return call_intrin(
            "handle",
            "tirx.tvm_access_ptr",
            inner_marker,
            inner_var,
            inner_offset + offset,
            1,
            rw_mask,
        )
    return tvm_access_ptr(elem_dtype, ptr, offset, 1, 1)


def ptx_ldmatrix_legacy(*all_args):
    """Legacy ``ptx_ldmatrix`` API taking explicit offsets.

    Signature: ``(trans, num, dtype, local_ptr, local_offset, smem_ptr,
    smem_offset)``. Offsets are folded into the pointers via
    ``tvm_access_ptr`` and dispatched to the fork-native
    :func:`ptx_ldmatrix`.

    ``T.ptx.ldmatrix_legacy`` runs through ``_dtype_forward`` which
    prepends a ``dtype=`` kwarg as a leading positional naming the buffer
    element type — offsets are in elements of that dtype, not bytes, so
    we forward it to ``tvm_access_ptr`` for correct scaling.
    """
    if len(all_args) == 8:
        elem_dtype, trans, num, dtype, local_ptr, local_offset, smem_ptr, smem_offset = all_args
    elif len(all_args) == 7:
        trans, num, dtype, local_ptr, local_offset, smem_ptr, smem_offset = all_args
        elem_dtype = "int8"
    else:
        raise ValueError(
            f"ptx_ldmatrix_legacy expects 7 args (or 8 with dtype= kwarg "
            f"prepended); got {len(all_args)}"
        )
    # Call.dtype carries the buffer element type so codegen can pick the
    # int8+trans manual-loop fallback (ldmatrix can't transpose int8).
    return call_intrin(
        elem_dtype,
        "tirx.ptx_ldmatrix_legacy",
        trans,
        num,
        dtype,
        local_ptr,
        local_offset,
        smem_ptr,
        smem_offset,
    )


def ptx_stmatrix(trans, num, dtype, smem_ptr, *src_handles, shape="m8n8", space="shared"):
    """TVM intrinsic for ``stmatrix.sync.aligned.shape.x{num}{.trans}.space.{dtype}``.

    Mirrors :func:`ptx_ldmatrix`: each source register is a separate operand.
    Pass ``Tx.address_of(buf[idx])`` (or ``buf.ptr_to([idx])``) for each
    source — the slots may be non-contiguous.

    Parameters
    ----------
    trans : bool
        Apply the ``.trans`` modifier (required for ``shape == "m16n8"``).
    num : int
        One of 1, 2, 4 — number of m8n8 fragments per warp.
    dtype : str
        ``".b16"`` (4 bytes per fragment register) or ``".b8"`` (2 bytes per).
    smem_ptr : PrimExpr
        Destination pointer in shared memory.
    *src_handles : PrimExpr
        ``num`` pointer-to-uint32 sources.
    shape : str, keyword-only, default "m8n8"
        ``"m8n8"`` or ``"m16n8"``.
    space : str, keyword-only, default "shared"
        ``"shared"`` or ``"shared::cta"``.

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix
    """
    _choice("num", num, _LDMATRIX_NUM)
    _choice("dtype", dtype, _LDMATRIX_DTYPE)
    if shape not in ("m8n8", "m16n8"):
        raise ValueError(f"Unsupported stmatrix shape {shape!r}")
    if space not in ("shared", "shared::cta"):
        raise ValueError(f"Unsupported stmatrix state space {space!r}")
    if shape == "m16n8" and not trans:
        raise ValueError("stmatrix .m16n8 requires .trans")
    n_regs = int(num)
    if len(src_handles) != n_regs:
        dtype_bare = dtype.lstrip(".") if isinstance(dtype, str) else dtype
        raise ValueError(
            f"stmatrix .x{int(num)}.{dtype_bare} expects {n_regs} source "
            f"handles, got {len(src_handles)}"
        )
    return call_intrin(
        "", "tirx.ptx_stmatrix", trans, num, dtype, shape, space, smem_ptr, *src_handles
    )


def ptx_wgmma_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    """TVM intrinsic to create memory descriptor for wgmma instructions

    Parameters
    ----------
    desc : PrimExpr
        The pointer to the shared memory descriptor.

    addr : PrimExpr
        The address of the matrix.

    ldo : PrimExpr
        The leading dimension offset.

    sdo : PrimExpr
        The stride dimension offset.

    swizzle : int
        The swizzle value (CUtensorMapSwizzle_enum).
    """
    return call_intrin("", "tirx.ptx_wgmma_encode_matrix_descriptor", desc, addr, ldo, sdo, swizzle)


def ptx_wgmma_noop_barrier(reg):
    """TVM intrinsic to call "" : "+{format}"(reg)::"memory"

    Parameters
    ----------
    reg : PrimExpr
        The register to fence.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_wgmma_noop_barrier", reg)


def ptx_wgmma_mma_async_ss(
    descA, descB, *accums, M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD
):
    """TVM intrinsic to call wgmma.mma_async.sync.aligned.shape.dtype.atype.btype over 2 smem operators

    Parameters
    ----------
    M : int
        The number of rows in matrix A and D.

    N : int
        The number of columns in matrix B and D.

    K : int
        The number of columns in matrix A and rows in matrix B.

    in_dtype : str
        The data type of the input matrices.

    out_type : str
        The data type of the output matrices.

    transA : bool
        True for M/N major, False for K major.

    transB : bool
        True for M/N major, False for K major.

    scaleA : float
        The scaling factor for matrix A.

    scaleB : float
        The scaling factor for matrix B.

    scaleD : PrimExpr
        True: D = A * B + D, False: D = A * B.

    descA : PrimExpr
        The SMEM descriptor of matrix A

    descB : PrimExpr
        The SMEM descriptor of matrix B

    accums : list
        The accumulators registers.
    """  # noqa: E501
    return call_intrin(
        "",
        "tirx.ptx_wgmma_mma_async_ss",
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        transA,
        transB,
        scaleA,
        scaleB,
        scaleD,
        descA,
        descB,
        *accums,
    )


def ptx_wgmma_mma_async_rs(
    descB, *reg_list, M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD
):
    """TVM intrinsic to call wgmma.mma_async.sync.aligned.shape.dtype.atype.btype
        When A is in register and B is in shared memory

    Parameters
    ----------
    M : int
        The number of rows in matrix A and D.

    N : int
        The number of columns in matrix B and D.

    K : int
        The number of columns in matrix A and rows in matrix B.

    in_dtype : str
        The data type of the input matrices.

    out_type : str
        The data type of the output matrices.

    transA : bool
        True for M/N major, False for K major.

    transB : bool
        True for M/N major, False for K major.

    scaleA : float
        The scaling factor for matrix A.

    scaleB : float
        The scaling factor for matrix B.

    scaleD : PrimExpr
        True: D = A * B + D, False: D = A * B.

    descB : PrimExpr
        The SMEM descriptor of matrix B

    reg_list : list
        The A registers and accumulators registers.
    """
    return call_intrin(
        "",
        "tirx.ptx_wgmma_mma_async_rs",
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        transA,
        transB,
        scaleA,
        scaleB,
        scaleD,
        descB,
        *reg_list,
    )


def ptx_wgmma_fence():
    """TVM intrinsic to call wgmma.fence.sync.aligned

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_wgmma_fence")


def ptx_wgmma_commit_group():
    """TVM intrinsic to call wgmma.commit_group.sync.aligned

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_wgmma_commit_group")


def ptx_wgmma_wait_group(n):
    """TVM intrinsic to call wgmma.wait_group.sync.aligned

    Parameters
    ----------
    n : int
        The number of the most recent uncommitted pending wgmma groups to wait.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_wgmma_wait_group", n)


def ptx_setmaxnreg(inc: bool, reg_count):
    """TVM intrinsic to call setmaxnreg.action.sync.aligned.u32 imm-reg-count

    Parameters
    ----------
    inc : bool
        True to increase the register count, False to decrease.

    reg_count : int
        The register count.
    """
    return call_intrin("", "tirx.ptx_setmaxnreg", inc, reg_count)


def ptx_tcgen05_alloc(dst_ptr, n_cols, cta_group=1):
    """TVM intrinsic to call tcgen05.alloc.cta_group.sync.aligned
        Dynamically allocates the number of cols in tensor memory, and write
        the address of allocated memory to shared memory.

    Parameters
    ----------
    dst_ptr : Var
        The pointer to the destination shared memory.

    n_cols : int
        The number of columns to allocate in tensor memory.
        Must be a multiple of 32 and a power of 2, and within the range [32, 512].

    cta_group : int
        The number of CTA groups involved in the allocation.
        If cta_group=1, one warp from CTA performs the allocation. Else, if cta_group=2,
        one warp from each of the peer CTAs perform the allocation.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx_tcgen05_alloc", dst_ptr, n_cols, cta_group)


def ptx_tcgen05_dealloc(taddr, n_cols, cta_group=1):
    """TVM intrinsic to call tcgen05.dealloc.cta_group.sync.aligned
        Deallocates the tensor memory specified by the tensor memory address taddr.

    Parameters
    ----------
    taddr : PrimExpr
        The address of previously allocated tensor memory, should be uint32_t.

    n_cols : int
        The number of columns to deallocate in tensor memory.
        Must be a multiple of 32 and a power of 2, and within the range [32, 512].

    cta_group : int
        The number of CTA groups involved in the deallocation.
        If cta_group=1, one warp from CTA performs the deallocation. Else, if cta_group=2,
        one warp from each of the peer CTAs perform the deallocation.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx_tcgen05_dealloc", taddr, n_cols, cta_group)


def ptx_tcgen05_relinquish_alloc_permit(cta_group=1):
    """TVM intrinsic to call tcgen05.relinquish_alloc_permit.cta_group.sync.aligned
        The CTA of the executing thread is relinquishing the right to allocate
        Tensor Memory after calling this op.

    Parameters
    ----------
    cta_group : int
        The number of CTA groups involved in relinquishing.
        If cta_group=1, one warp from CTA performs the relinquishing. Else, if cta_group=2,
        one warp from each of the peer CTAs perform the relinquishing.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx_tcgen05_relinquish_alloc_permit", cta_group)


def ptx_tcgen05_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    """TVM intrinsic to create memory descriptor for tcgen05 instructions

    Parameters
    ----------
    desc : PrimExpr
        The pointer to the shared memory descriptor.

    addr : PrimExpr
        The address of the matrix.

    ldo : PrimExpr
        The leading dimension offset.

    sdo : PrimExpr
        The stride dimension offset.

    swizzle : int
        The swizzle value (CUtensorMapSwizzle_enum).
    """
    return call_intrin(
        "", "tirx.ptx_tcgen05_encode_matrix_descriptor", desc, addr, ldo, sdo, swizzle
    )


def ptx_tcgen05_encode_instr_descriptor(
    desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_groups=1,
    neg_a=False,
    neg_b=False,
    sat_d=False,
    is_sparse=False,
):
    """TVM intrinsic to create instruction descriptor for tcgen05 MMA without block scaling

    Parameters
    ----------
    desc : PrimExpr
        The pointer to the instruction descriptor.

    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    M : int
        The size of non-reduction dimension of Matrix A.

    N : int
        The size of non-reduction dimension of Matrix B.

    K : int
        The size of reduction dimension of Matrix A/B.

    trans_a : bool
        Whether the multiplicand matrix A is transposed.
        True for M/N major, False for K major.

    trans_b : bool
        Whether the multiplicand matrix B is transposed.
        True for M/N major, False for K major.

    n_cta_groups : int
        The number of CTA groups involved in the MMA operation.

    neg_a : bool
        Whether to negate the multiplicand matrix A.

    neg_b : bool
        Whether to negate the multiplicand matrix B.

    sat_d : bool
        Whether to saturate the resultant matrix D.

    is_sparse : bool
        Whether the MMA operation is sparse.
    """
    _choice("n_cta_groups", n_cta_groups, _TCGEN05_CTA_GROUP)
    return call_intrin(
        "",
        "tirx.ptx_tcgen05_encode_instr_descriptor",
        desc,
        d_dtype,
        a_dtype,
        b_dtype,
        M,
        N,
        K,
        trans_a,
        trans_b,
        n_cta_groups,
        neg_a,
        neg_b,
        sat_d,
        is_sparse,
    )


def ptx_tcgen05_encode_instr_descriptor_block_scaled(
    desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    sfa_tmem_addr,
    sfb_tmem_addr,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_groups=1,
    neg_a=False,
    neg_b=False,
    is_sparse=False,
):
    """TVM intrinsic to create instruction descriptor for tcgen05 MMA with block scaling

    Parameters
    ----------
    desc : PrimExpr
        The pointer to the instruction descriptor.

    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    sfa_dtype : str
        The datatype of scale factor matrix A.

    sfb_dtype : str
        The datatype of scale factor matrix B.

    sfa_tmem_addr : PrimExpr
        The address of the scale factor matrix A in tensor memory, should be uint32_t.

    sfb_tmem_addr : PrimExpr
        The address of the scale factor matrix B in tensor memory, should be uint32_t.

    M : int
        The size of non-reduction dimension of Matrix A.

    N : int
        The size of non-reduction dimension of Matrix B.

    K : int
        The size of reduction dimension of Matrix A/B.

    trans_a : bool
        Whether the multiplicand matrix A is transposed.
        True for M/N major, False for K major.

    trans_b : bool
        Whether the multiplicand matrix B is transposed.
        True for M/N major, False for K major.

    n_cta_groups : int
        The number of CTA groups involved in the MMA operation.

    neg_a : bool
        Whether to negate the multiplicand matrix A.

    neg_b : bool
        Whether to negate the multiplicand matrix B.

    is_sparse : bool
        Whether the MMA operation is sparse.
    """
    _choice("n_cta_groups", n_cta_groups, _TCGEN05_CTA_GROUP)
    return call_intrin(
        "",
        "tirx.ptx_tcgen05_encode_instr_descriptor_block_scaled",
        desc,
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        sfa_tmem_addr,
        sfb_tmem_addr,
        M,
        N,
        K,
        trans_a,
        trans_b,
        n_cta_groups,
        neg_a,
        neg_b,
        is_sparse,
    )


def ptx_tcgen05_mma(
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    *disable_output_lane,
    d_dtype,
    a_dtype,
    b_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
    scale_input_d=0,
    pred=None,
):
    """TVM intrinsic to call tcgen05.mma.cta_group.kind without block scaling.

    Parameters
    ----------
    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. The inline asm tests
        `enable_input_d != 0`: zero means D = A*B, non-zero means D = A*B + D.

    scale_input_d : int
        The optional scaling factor to scale input matrix D.
        D = A*B+D * (2 ^ - scale-input-d)

    disable_output_lane : list
        The lanes that should not be updated in the resultant matrix D.

    pred : Optional[PrimExpr]
        Runtime ``uint32`` instruction-level predicate. When given, emit
        ``@p_issue tcgen05.mma...`` with ``p_issue = (pred != 0)``. Preserves
        PTX-level predicate semantics (single predicated SASS instruction).
    """

    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)

    # default value for disable_output_lane
    if len(disable_output_lane) == 0:
        disable_output_lane = [0] * (4 if cta_group == 1 else 8)

    args = [
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
    ]
    if pred is not None:
        args.append(pred)
    return call_intrin("", "tirx.ptx_tcgen05_mma", *args)


def ptx_tcgen05_mma_block_scale(
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
):
    """TVM intrinsic to call tcgen05.mma.cta_group.kind.block_scale
        Performs matrix multiplication with block scaling:
        (A * scale_A)  * (B * scale_B) + D

    Parameters
    ----------
    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    sfa_dtype : str
        The datatype of scale factor matrix A.

    sfb_dtype : str
        The datatype of scale factor matrix B.

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    sfa_tmem_addr : PrimExpr
        The address of the scale factor matrix A in tensor memory, should be uint32_t.

    sfb_tmem_addr : PrimExpr
        The address of the scale factor matrix B in tensor memory, should be uint32_t.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. Zero means D = A*B,
        non-zero means D = A*B + D.
    """

    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin(
        "",
        "tirx.ptx_tcgen05_mma_block_scale",
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
    )


def ptx_tcgen05_mma_sp(
    d_tmem_addr,
    a_operand,
    b_desc,
    sp_tmem_addr,
    i_desc,
    *disable_output_lane,
    d_dtype,
    a_dtype,
    b_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
    scale_input_d=0,
):
    """TVM intrinsic to call tcgen05.mma.sp.cta_group.kind without block scaling.

    Parameters
    ----------
    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    sp_tmem_addr : PrimExpr
        The address of the metadata of sparse matrix in tensor memory, should be uint32_t.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. The inline asm tests
        `enable_input_d != 0`: zero means D = A*B, non-zero means D = A*B + D.

    scale_input_d : int
        The optional scaling factor to scale input matrix D.
        D = A*B+D * (2 ^ - scale-input-d)

    disable_output_lane : list
        The lanes that should not be updated in the resultant matrix D.
    """

    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)

    # default value for disable_output_lane
    if len(disable_output_lane) == 0:
        disable_output_lane = [0] * (4 if cta_group == 1 else 8)

    return call_intrin(
        "",
        "tirx.ptx_tcgen05_mma_sp",
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sp_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
    )


def ptx_tcgen05_mma_sp_block_scale(
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    sp_tmem_addr,
    i_desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
):
    """TVM intrinsic to call tcgen05.mma.sp.cta_group.kind.block_scale
        Performs sparse matrix multiplication with block scaling:
        (A * scale_A)  * (B * scale_B) + D

    Parameters
    ----------
    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    sfa_dtype : str
        The datatype of scale factor matrix A.

    sfb_dtype : str
        The datatype of scale factor matrix B.

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    sfa_tmem_addr : PrimExpr
        The address of the scale factor matrix A in tensor memory, should be uint32_t.

    sfb_tmem_addr : PrimExpr
        The address of the scale factor matrix B in tensor memory, should be uint32_t.

    sp_tmem_addr : PrimExpr
        The address of the metadata of sparse matrix in tensor memory, should be uint32_t.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. Zero means D = A*B,
        non-zero means D = A*B + D.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin(
        "",
        "tirx.ptx_tcgen05_mma_sp_block_scale",
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        sp_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
    )


def ptx_tcgen05_fence_before_thread_sync():
    """TVM intrinsic to call tcgen05.fence::before_thread_sync
    Orders all prior asynchronous tcgen05 operations relative to subsequent operations.
    """
    return call_intrin("", "tirx.ptx_tcgen05_fence_before_thread_sync")


def ptx_tcgen05_fence_after_thread_sync():
    """TVM intrinsic to call tcgen05.fence::after_thread_sync
    Orders all subsequent asynchronous tcgen05 operations relative to previous operations.
    """
    return call_intrin("", "tirx.ptx_tcgen05_fence_after_thread_sync")


def _choice(name: str, value, options):
    """Validate `value` is one of `options`. Raise a clear ValueError otherwise.

    Symbolic values (Var, non-constant PrimExpr) are accepted without
    validation; specialization later replaces them with concrete values
    that the C-side intrinsic body re-checks.
    """
    # Concrete int / IntImm value: validate.
    try:
        concrete = int(value)
    except (TypeError, ValueError):
        return  # symbolic; defer check
    if concrete not in options:
        raise ValueError(f"invalid {name}={concrete!r}; expected one of {tuple(options)}")


# See top-of-file imports for `_FENCE_SEM` etc. (re-exported from _common).
# Note: TCGEN05_LDST_SHAPES values must stay in sync with the shape branches
# of codegen_ptx_tcgen05_ld/_st in intrinsics/cuda/tcgen05.py.


def ptx_tcgen05_cp(
    taddr, src_desc, *, shape, cta_group=1, multicast="", decompress="", row=0, col=0
):
    """TVM intrinsic for the Blackwell `tcgen05.cp` PTX instruction.

    The emitted PTX is::

        tcgen05.cp.cta_group::{cta_group}.{shape}[.{multicast}][.{decompress}] [taddr], src_desc;

    Each keyword argument maps 1:1 to a PTX token: read the call and you
    know what instruction is emitted.

    Parameters
    ----------
    taddr : PrimExpr
        Destination tensor-memory address (uint32). Callers typically pass
        ``tmem_base + column_offset_in_uint32s`` directly. Use the optional
        ``row`` / ``col`` keyword arguments only when the address needs
        runtime row/col composition via ``get_tmem_addr`` (high 16 bits row,
        low 16 bits col).

    src_desc : PrimExpr
        The 64-bit shared-memory matrix descriptor.

    shape : str
        One of ``"32x128b"``, ``"4x256b"``, ``"128x128b"``, ``"128x256b"``,
        ``"64x128b"``.

    cta_group : int
        1 or 2.

    multicast : str
        One of ``""``, ``"warpx4"``, ``"warpx2::02_13"``, ``"warpx2::01_23"``.
        ``"32x128b"`` requires ``"warpx4"``; ``"64x128b"`` requires one of the
        ``warpx2::*`` values; other shapes require ``""``.

    decompress : str
        Trailing PTX suffix for fp4/fp6 → fp8 on-the-fly decompression.
        One of ``""``, ``"b8x16.b4x16_p64"``, ``"b8x16.b6x16_p32"``.

    row, col : PrimExpr
        Optional row/col offsets added to ``taddr`` at runtime. Default 0.
    """
    _choice("shape", shape, _TCGEN05_CP_SHAPES)
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    _choice("multicast", multicast, _TCGEN05_CP_MULTICAST)
    _choice("decompress", decompress, _TCGEN05_CP_DECOMPRESS)
    if shape == "32x128b" and multicast != "warpx4":
        raise ValueError(f"shape=32x128b requires multicast='warpx4', got {multicast!r}")
    if shape == "64x128b" and multicast not in ("warpx2::02_13", "warpx2::01_23"):
        raise ValueError(f"shape=64x128b requires multicast in warpx2::*, got {multicast!r}")
    if shape in ("128x128b", "128x256b", "4x256b") and multicast != "":
        raise ValueError(f"shape={shape} requires multicast='', got {multicast!r}")

    return call_intrin(
        "",
        "tirx.ptx_tcgen05_cp",
        taddr,
        src_desc,
        shape,
        cta_group,
        multicast,
        decompress,
        row,
        col,
    )


def ptx_tcgen05_shift(taddr, cta_group=1):
    """TVM intrinsic to call tcgen05.shift.cta_group.down
        Asynchronously shift down the rows of the matrix in Tensor Memory for a warp.

    Parameters
    ----------
    taddr : PrimExpr
        The address of matrix in tensor memory, should be uint32_t.

    cta_group : int
        The number of CTA groups involved in the shift.
        If cta_group=1, shift operation is performed in the Tensor Memory of current CTA.
        Else, shift operation is performed in the Tensor Memory of both the current CTA and
        the peer CTA.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx_tcgen05_shift", taddr, cta_group)


def ptx_tcgen05_ld(src_addr, *regs, shape, num, row=0, col=0, pack=False):
    """TVM intrinsic for tcgen05.ld.sync.aligned — async collective load from TMEM.

    Emits ``tcgen05.ld.sync.aligned.{shape}.x{num}[.pack::16b].b32 {regs}, [addr];``

    Parameters
    ----------
    src_addr : PrimExpr
        Tensor-memory source address (uint32).

    regs : list[PrimExpr]
        Destination registers. Count depends on shape x num.

    shape : str
        One of ``"16x32bx2"``, ``"16x64b"``, ``"16x128b"``, ``"16x256b"``, ``"32x32b"``.

    num : int
        Repeat factor along the columns. Power-of-two in [1, 128].

    row, col : PrimExpr
        Optional TMEM row/col offsets added to ``src_addr`` at runtime (row must be
        a multiple of 32). Default 0.

    pack : bool
        Pack two 16-bit chunks into a single 32-bit register.
    """
    _choice("shape", shape, _TCGEN05_LDST_SHAPES)
    return call_intrin("", "tirx.ptx_tcgen05_ld", src_addr, row, col, shape, num, pack, *regs)


def ptx_tcgen05_st(dst_addr, *regs, shape, num, row=0, col=0, unpack=False):
    """TVM intrinsic for tcgen05.st.sync.aligned — async collective store to TMEM.

    Emits ``tcgen05.st.sync.aligned.{shape}.x{num}[.unpack::16b].b32 [addr], {regs};``

    Parameters
    ----------
    dst_addr : PrimExpr
        Tensor-memory destination address (uint32).

    regs : list[PrimExpr]
        Source registers. Count depends on shape x num.

    shape : str
        One of ``"16x32bx2"``, ``"16x64b"``, ``"16x128b"``, ``"16x256b"``, ``"32x32b"``.

    num : int
        Repeat factor along the columns. Power-of-two in [1, 128].

    row, col : PrimExpr
        Optional TMEM row/col offsets added to ``dst_addr`` at runtime (row must be
        a multiple of 32). Default 0.

    unpack : bool
        Unpack a 32-bit register into two 16-bit chunks.
    """
    _choice("shape", shape, _TCGEN05_LDST_SHAPES)
    return call_intrin("", "tirx.ptx_tcgen05_st", dst_addr, row, col, shape, num, unpack, *regs)


def ptx_tcgen05_wait_ld():
    """TVM intrinsic to call tcgen05.wait::ld.sync.aligned
    Wait for the completion of all prior async tcgen05.ld operations.
    """
    return call_intrin("", "tirx.ptx_tcgen05_wait_ld")


def ptx_tcgen05_wait_st():
    """TVM intrinsic to call tcgen05.wait::st.sync.aligned
    Wait for the completion of all prior async tcgen05.st operations.
    """
    return call_intrin("", "tirx.ptx_tcgen05_wait_st")


def ptx_tcgen05_commit(bar, cta_group=1, cta_mask=0, *, pred=None):
    """TVM intrinsic to call tcgen05.commit.cta_group

    Parameters
    ----------
    bar : PrimExpr
        The pointer to mbarrier variable.

    cta_group: int
        The number of CTA groups involved in previous tcgen05 operations.

    cta_mask : int
        The mask of the CTAs in the cluster, used for multicast.

    pred : Optional[PrimExpr]
        Runtime ``uint32`` predicate. When given, emit
        ``@p tcgen05.commit...`` with ``p = (pred != 0)``. This preserves
        PTX-level instruction predicate semantics (single predicated
        instruction in SASS), distinct from a C-level ``if`` branch.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    args = [bar, cta_group, cta_mask]
    if pred is not None:
        args.append(pred)
    return call_intrin("", "tirx.ptx_tcgen05_commit", *args)


def print_buffer(buffer_var, dtype, is_string, is_scalar, dim_num, *shape):
    """Print out buffer memory (tensor, string, or scalar) during runtime on cuda.
    This print function allows printing out buffer in tvm during runtime without
    dumping all the cuda code.
    Parameters
    ----------
    buffer_var : Var
        The data pointer of the buffer that needs to be printed out.
    dtype : DataType
        The data type of the buffer.
    is_string: Bool
        Whether the buffer is a string (dtype is Int8 by default in the backend).
    is_scalar: Bool
        Whether the buffer is a scalar.
    dim_num : Int
        The number of dimensions of the buffer
    *shape : Tuple
        The dimensions of the buffer in order.
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    final_shape_args = []
    if len(shape) == 1 and isinstance(shape[0], tuple | list | tvm.ir.Array):
        # Case 1: Called as print_buffer(..., dim, (s1, s2, ...))
        # The user provided a tuple/list as the single shape argument.
        final_shape_args = list(shape[0])
    else:
        # Case 2: Called as print_buffer(..., dim, s1, s2, ...)
        # This is how TVMScript parser will call it.
        final_shape_args = list(shape)

    return _ffi_api.print_buffer(
        buffer_var, dtype, is_string, is_scalar, dim_num, *final_shape_args
    )


def timer_init_cuda(profiler_buffer, profiler_tag, profiler_write_offset, num_groups, group_id):
    """TVM intrinsic for initializing the CUDA profiler, and store profiling result in a buffer.

    Parameters
    ----------
    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    num_groups: int
        The number of groups in the profiler.

    group_id: PrimExpr
        The group id of the current thread.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin(
        "handle",
        "tirx.timer_init_cuda",
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        num_groups,
        group_id,
    )


def timer_start_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
):
    """TVM intrinsic for starting the timer for profiling a specific event, and storing profiling result in a buffer.

    Parameters
    ----------
    event_type: Enum
        The event to profile.

    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    profiler_write_stride: int
        The stride to advance in buffer in the next write.

    leader_cond: PrimExpr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "handle",
        "tirx.timer_start_cuda",
        event_type.value,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
    )


def timer_end_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
):
    """TVM intrinsic for ending the timer for profiling a specific event, and storing profiling result in a buffer.

    Parameters
    ----------
    event_type: Enum
        The event to profile.

    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    profiler_write_stride: int
        The stride to advance in buffer in the next write.

    leader_cond: PrimExpr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "handle",
        "tirx.timer_end_cuda",
        event_type.value,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
    )


def timer_finalize_cuda(
    profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride, leader_cond
):
    """TVM intrinsic for finalizing the CUDA profiler, and store profiling result in a buffer.

    Parameters
    ----------
    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    profiler_write_stride: int
        The stride to advance in buffer in the next write.

    leader_cond: PrimExpr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin(
        "handle",
        "tirx.timer_finalize_cuda",
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
    )


def cuda_atomic_add(res_addr, value):
    """TVM intrinsic to call cuda atomic add instruction

    Parameters
    ----------
    res_addr : PrimExpr
        The result address.

    value: PrimExpr
        The value to add.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    value = tir.convert(value)
    return call_intrin(value.dtype, "tirx.cuda_atomic_add", res_addr, value)


def cuda_thread_fence():
    """TVM intrinsic to call cuda thread fence instruction

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_thread_fence")


def cuda_warpgroup_sync(bar_no):
    """TVM intrinsic to synchronize a CUDA warpgroup via a named barrier.

    Parameters
    ----------
    bar_no : PrimExpr
        The named barrier id to use for the warpgroup.

    Notes
    -----
    Synchronizes 128 threads in a warpgroup using `bar.sync bar_no, 128`.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_warpgroup_sync", bar_no)


def cuda_syncthreads_and(cond):
    """TVM intrinsic to call cuda syncthreads_and instruction

    Parameters
    ----------
    cond: PrimExpr
        The condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int64", "tirx.cuda_syncthreads_and", cond)


def cuda_syncthreads_or(cond):
    """TVM intrinsic to call cuda syncthreads_or instruction

    Parameters
    ----------
    cond: PrimExpr
        The condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int64", "tirx.cuda_syncthreads_or", cond)


def cuda_nano_sleep(time):
    """TVM intrinsic to call cuda nano sleep instruction

    Parameters
    ----------
    time: PrimExpr
        The time to sleep.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_nano_sleep", time)


def cuda_printf(fmt, *args):
    """TVM intrinsic to call cuda printf instruction

    Parameters
    ----------
    fmt: str
        The format string.

    *args: list
        The arguments to the format string.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda_printf", fmt, *args)


def cuda_ldg(addr, dtype):
    """TVM intrinsic to call CUDA C++ __ldg() function

    Parameters
    ----------
    addr : PrimExpr
        The memory address to load.

    dtype : str
        The data type of the loaded value.

    Returns
    """
    return call_intrin(dtype, "tirx.cuda_ldg", addr, dtype)


def cuda_get_tmem_addr(addr, row_offset, col_offset):
    """TVM intrinsic to call cuda tmem address calculation

    Parameters
    ----------
    addr: PrimExpr
        The memory address to calculate.

    row_offset: PrimExpr
        The row offset to calculate.

    col_offset: PrimExpr
        The column offset to calculate.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("uint32", "tirx.cuda_get_tmem_addr", addr, row_offset, col_offset)


def cuda_cvta_generic_to_shared(ptr):
    """Convert a generic pointer to a shared-memory address (uint32).

    Wraps ``__cvta_generic_to_shared(ptr)``. Used by op-wrappers that
    precompute the shared-memory address at the wrapper layer instead of
    inside the asm helper body.
    """
    return call_intrin("uint32", "tirx.cuda_cvta_generic_to_shared", ptr)


def cuda_smem_addr_from_uint64(cluster_addr):
    """Narrow a 64-bit cluster-mapped SMEM address to a 32-bit SMEM address.

    Wraps ``static_cast<unsigned int>(cluster_addr)``. Used by
    cp.async.bulk.shared::cluster.* op-wrappers.
    """
    return call_intrin("uint32", "tirx.cuda_smem_addr_from_uint64", cluster_addr)


def cuda_sm100_tma_2sm_mbarrier_addr(bar):
    """Compute the SM100 2SM TMA mbarrier shared-address operand."""
    return bitwise_and(cuda_cvta_generic_to_shared(bar), const(0xFEFFFFFF, dtype="uint32"))


def ptx_exp2(x):
    """TVM intrinsic for PTX fast exp2 approximation (ex2.approx.ftz.f32)

    Parameters
    ----------
    x : PrimExpr
        The float32 input value.

    Returns
    -------
    call : PrimExpr
        The call expression returning 2^x (approximate).
    """
    return call_intrin("float32", "tirx.ptx_exp2", x)


def ptx_rcp(x):
    """TVM intrinsic for PTX fast reciprocal approximation (rcp.approx.ftz.f32)

    Parameters
    ----------
    x : PrimExpr
        The float32 input value.

    Returns
    -------
    call : PrimExpr
        The call expression returning 1/x (approximate).
    """
    return call_intrin("float32", "tirx.ptx_rcp", x)


def ptx_any_sync(mask, pred):
    """TVM intrinsic for PTX warp-wide any predicate (__any_sync)

    Parameters
    ----------
    mask : PrimExpr
        The thread mask (uint32).
    pred : PrimExpr
        The predicate value (int32).

    Returns
    -------
    call : PrimExpr
        The call expression returning 1 if any thread in mask has pred != 0.
    """
    return call_intrin("int32", "tirx.ptx_any_sync", mask, pred)


def ptx_reduce3_max_f32(a, b, c):
    """TVM intrinsic to call 3-input max.f32 PTX instruction (sm_100a+)

    Parameters
    ----------
    a, b, c : PrimExpr
        The three float32 values to compare.

    Returns
    -------
    call : PrimExpr
        The call expression returning max(a, b, c).
    """
    return call_intrin("float32", "tirx.ptx_reduce3_max_f32", a, b, c)


def ptx_reduce3_min_f32(a, b, c):
    """TVM intrinsic to call 3-input min.f32 PTX instruction (sm_100a+)

    Parameters
    ----------
    a, b, c : PrimExpr
        The three float32 values to compare.

    Returns
    -------
    call : PrimExpr
        The call expression returning min(a, b, c).
    """
    return call_intrin("float32", "tirx.ptx_reduce3_min_f32", a, b, c)


def _ptx_binary_arith(op_name, dtype, d, a, b, *, rounding="rn", ftz=False, sat=False):
    """Shared helper for add/sub/mul over (f32 | f32x2 | f64), DPS form."""
    _choice("rounding", rounding, _F32X2_ROUND)
    if dtype == "f64" and (ftz or sat):
        raise ValueError(f"PTX {op_name}.f64 does not accept .ftz or .sat")
    if dtype == "f32x2" and sat:
        raise ValueError(f"PTX {op_name}.f32x2 does not accept .sat")
    return call_intrin(
        "",
        f"tirx.ptx_{op_name}_{dtype}",
        d,
        a,
        b,
        rounding,
        int(ftz),
        int(sat),
    )


def _ptx_fma(dtype, d, a, b, c, *, rounding="rn", ftz=False, sat=False):
    """Shared helper for fma over (f32 | f32x2 | f64), DPS form."""
    _choice("rounding", rounding, _F32X2_ROUND)
    if dtype == "f64" and (ftz or sat):
        raise ValueError("PTX fma.f64 does not accept .ftz or .sat")
    if dtype == "f32x2" and sat:
        raise ValueError("PTX fma.f32x2 does not accept .sat")
    return call_intrin(
        "",
        f"tirx.ptx_fma_{dtype}",
        d,
        a,
        b,
        c,
        rounding,
        int(ftz),
        int(sat),
    )


def ptx_add_f32(d_addr, a, b, *, rounding="rn", ftz=False, sat=False):
    """PTX ``add{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("add", "f32", d_addr, a, b, rounding=rounding, ftz=ftz, sat=sat)


def ptx_add_f32x2(d_addr, a, b, *, rounding="rn", ftz=False):
    """PTX ``add{.rnd}{.ftz}.f32x2 [d_addr], a, b`` — DPS form.

    a, b are packed-as-uint64 register operands (2 fp32 each).
    """
    return _ptx_binary_arith("add", "f32x2", d_addr, a, b, rounding=rounding, ftz=ftz)


def ptx_add_f64(d_addr, a, b, *, rounding="rn"):
    """PTX ``add{.rnd}.f64 [d_addr], a, b`` — DPS form (no .ftz / .sat)."""
    return _ptx_binary_arith("add", "f64", d_addr, a, b, rounding=rounding)


def ptx_sub_f32(d_addr, a, b, *, rounding="rn", ftz=False, sat=False):
    """PTX ``sub{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("sub", "f32", d_addr, a, b, rounding=rounding, ftz=ftz, sat=sat)


def ptx_sub_f32x2(d_addr, a, b, *, rounding="rn", ftz=False):
    """PTX ``sub{.rnd}{.ftz}.f32x2 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("sub", "f32x2", d_addr, a, b, rounding=rounding, ftz=ftz)


def ptx_sub_f64(d_addr, a, b, *, rounding="rn"):
    """PTX ``sub{.rnd}.f64 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("sub", "f64", d_addr, a, b, rounding=rounding)


def ptx_mul_f32(d_addr, a, b, *, rounding="rn", ftz=False, sat=False):
    """PTX ``mul{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("mul", "f32", d_addr, a, b, rounding=rounding, ftz=ftz, sat=sat)


def ptx_mul_f32x2(d_addr, a, b, *, rounding="rn", ftz=False):
    """PTX ``mul{.rnd}{.ftz}.f32x2 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("mul", "f32x2", d_addr, a, b, rounding=rounding, ftz=ftz)


def ptx_mul_f64(d_addr, a, b, *, rounding="rn"):
    """PTX ``mul{.rnd}.f64 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("mul", "f64", d_addr, a, b, rounding=rounding)


def ptx_fma_f32(d_addr, a, b, c, *, rounding="rn", ftz=False, sat=False):
    """PTX ``fma{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b, c`` — DPS form."""
    return _ptx_fma("f32", d_addr, a, b, c, rounding=rounding, ftz=ftz, sat=sat)


def ptx_fma_f32x2(d_addr, a, b, c, *, rounding="rn", ftz=False):
    """PTX ``fma{.rnd}{.ftz}.f32x2 [d_addr], a, b, c`` — DPS form.

    a, b, c are packed-as-uint64 register operands.
    """
    return _ptx_fma("f32x2", d_addr, a, b, c, rounding=rounding, ftz=ftz)


def ptx_fma_f64(d_addr, a, b, c, *, rounding="rn"):
    """PTX ``fma{.rnd}.f64 [d_addr], a, b, c`` — DPS form."""
    return _ptx_fma("f64", d_addr, a, b, c, rounding=rounding)


def ptx_max_f32(a, b, *, ftz=False, nan=False):
    """TVM intrinsic for PTX ``max{.ftz}{.NaN}.f32 d, a, b``.

    2-operand form (distinct from :func:`ptx_reduce3_max_f32` which is the
    3-operand SM_100+ form). ``.NaN`` qualifier propagates NaN inputs to
    the output; without it, NaN inputs are silently ignored.

    Parameters
    ----------
    a, b : PrimExpr
        Float32 inputs.
    ftz : bool
        If True, flush subnormals to zero (``.ftz``).
    nan : bool
        If True, propagate NaN inputs (``.NaN``).
    """
    return call_intrin("float32", "tirx.ptx_max_f32", a, b, int(ftz), int(nan))


def ptx_griddepcontrol_wait():
    """TVM intrinsic for PTX ``griddepcontrol.wait`` (sm_90+).

    Blocks the current grid until prerequisite grids signalled via
    :func:`ptx_griddepcontrol_launch_dependents` have finished. Acts as a
    full memory barrier.
    """
    return call_intrin("", "tirx.ptx_griddepcontrol_wait")


def ptx_griddepcontrol_launch_dependents():
    """TVM intrinsic for PTX ``griddepcontrol.launch_dependents`` (sm_90+).

    Signals that the current grid has reached a point where dependent
    grids may begin execution.
    """
    return call_intrin("", "tirx.ptx_griddepcontrol_launch_dependents")


_PTX_LD_SCOPE = {"cta", "cluster", "gpu", "sys"}
_PTX_LD_SPACE = {"global", "shared", "shared::cta", "shared::cluster", "local"}
_PTX_LD_VOLATILE_SPACE = _PTX_LD_SPACE | {"const"}
_PTX_LD_TYPE = {"b32", "u32", "u64", "s32", "f32"}
_PTX_LD_COP = {"", "ca", "cg", "cs", "lu", "cv"}
_PTX_MEM_SCOPE = {"", "cta", "cluster", "gpu", "sys"}
_PTX_MEM_SPACE = {"global", "shared", "shared::cta", "shared::cluster"}
_PTX_SCALAR_TYPE = {"b32", "b64", "u32", "u64", "s32", "s64", "f32", "f64"}
_PTX_RED_OP = {"and", "or", "xor", "add", "inc", "dec", "min", "max"}
_PTX_ATOM_OP = {"and", "or", "xor", "exch", "add", "inc", "dec", "min", "max"}
_PTX_ST_VEC = {"", "v2", "v4", "v8"}
_PTX_ST_COP = {"", "wb", "cg", "cs", "wt"}
_PTX_PREFETCH_TENSORMAP_SPACE = {"", "const", "param"}
_PTX_SCALAR_RETURN_TYPE = {
    "b32": "uint32",
    "u32": "uint32",
    "s32": "int32",
    "b64": "uint64",
    "u64": "uint64",
    "s64": "int64",
    "f32": "float32",
    "f64": "float64",
}
_PTX_CACHE_POLICY = {
    "evict_normal": 0x1000000000000000,
    "evict_first": 0x12F0000000000000,
    "evict_last": 0x14F0000000000000,
}


def _resolve_cache_policy(cache_hint, cache_policy, choices=_CP_ASYNC_BULK_CACHE_HINT):
    _choice("cache_hint", cache_hint, choices)
    if cache_policy is not None:
        return cache_policy, True
    if cache_hint:
        if cache_hint not in _PTX_CACHE_POLICY:
            raise ValueError(
                f"Unsupported built-in cache policy {cache_hint!r}; pass cache_policy explicitly"
            )
        return const(_PTX_CACHE_POLICY[cache_hint], dtype="uint64"), True
    return const(0, dtype="uint64"), False


def ptx_ld_acquire(addr, return_type, ptx_type, *, scope="gpu", space="global"):
    """TVM intrinsic for scalar PTX ``ld.acquire.scope{.ss}.type`` loads.

    This wrapper covers the scalar no-cache-policy/no-vector instances of the
    PTX ISA ``ld.acquire`` form. ``scope``, state ``space``, PTX ``type`` and
    TVM ``return_type`` are explicit so callers can request either raw-bit or
    typed loads.

    Parameters
    ----------
    addr : PrimExpr
        The memory address to load.

    return_type : str
        TVM dtype returned by the load.

    ptx_type : str
        PTX type suffix such as ``"b32"``, ``"u64"``, or ``"s32"``.

    scope : str
        PTX memory scope: ``"cta"``, ``"cluster"``, ``"gpu"``, or ``"sys"``.

    space : str
        PTX state space suffix.

    Returns
    -------
    call : PrimExpr
        The loaded value.
    """
    _choice("scope", scope, _PTX_LD_SCOPE)
    _choice("space", space, _PTX_LD_SPACE)
    _choice("ptx_type", ptx_type, _PTX_LD_TYPE)
    return call_intrin(
        return_type, "tirx.ptx_ld_acquire", addr, return_type, ptx_type, scope, space
    )


def ptx_ld(
    addr,
    return_type,
    ptx_type,
    *,
    weak=False,
    space="global",
    cop="",
    cache_hint="",
    cache_policy=None,
):
    """TVM intrinsic for scalar PTX ``ld{.weak}{.ss}{.cop}{.level::cache_hint}.type``.

    This wrapper covers scalar no-prefetch/no-vector instances of the weak
    generic load form.
    """
    _choice("space", space, _PTX_LD_SPACE | {"const", "param::entry", "param::func"})
    _choice("cop", cop, _PTX_LD_COP)
    _choice("ptx_type", ptx_type, _PTX_LD_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        return_type,
        "tirx.ptx_ld",
        addr,
        cache_policy,
        return_type,
        int(bool(weak)),
        space,
        cop,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_ld_volatile(addr, return_type, ptx_type, *, space="global"):
    """TVM intrinsic for scalar PTX ``ld.volatile{.ss}.type`` loads.

    This wrapper covers scalar no-prefetch/no-vector instances.
    """
    _choice("space", space, _PTX_LD_VOLATILE_SPACE)
    _choice("ptx_type", ptx_type, _PTX_LD_TYPE)
    return call_intrin(return_type, "tirx.ptx_ld_volatile", addr, return_type, ptx_type, space)


def ptx_ld_global_acquire(res, addr):
    """TVM intrinsic to call the legacy ptx ld.global.acquire helper.

    Parameters
    ----------
    res : PrimExpr
        The result of the load.

    addr : PrimExpr
        The memory address to load.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx_ld_global_acquire", res, addr)


def ptx_red_scalar(
    address,
    value,
    *,
    sem="",
    scope="",
    space="global",
    op,
    ptx_type,
    cache_hint="",
    cache_policy=None,
):
    _choice("scope", scope, _PTX_MEM_SCOPE)
    _choice("space", space, _PTX_MEM_SPACE)
    _choice("op", op, _PTX_RED_OP)
    _choice("ptx_type", ptx_type, _PTX_SCALAR_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(
        cache_hint, cache_policy, _CP_ASYNC_CACHE_HINT
    )
    if sem not in ("", "relaxed", "release"):
        raise ValueError(f"Unsupported PTX red sem {sem!r}")
    return call_intrin(
        "",
        "tirx.ptx_red_scalar",
        address,
        value,
        cache_policy,
        sem,
        scope,
        space,
        op,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_atom_scalar(
    address,
    value,
    *,
    sem="",
    scope="",
    space="global",
    op,
    ptx_type,
    cache_hint="",
    cache_policy=None,
):
    _choice("scope", scope, _PTX_MEM_SCOPE)
    _choice("space", space, _PTX_MEM_SPACE)
    _choice("op", op, _PTX_ATOM_OP)
    _choice("ptx_type", ptx_type, _PTX_SCALAR_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    if sem not in ("", "relaxed", "acquire", "release", "acq_rel"):
        raise ValueError(f"Unsupported PTX atom sem {sem!r}")
    return call_intrin(
        _PTX_SCALAR_RETURN_TYPE[ptx_type],
        "tirx.ptx_atom_scalar",
        address,
        value,
        cache_policy,
        sem,
        scope,
        space,
        op,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_st(
    address,
    *values,
    weak=False,
    space="shared",
    cop="",
    vec="",
    ptx_type,
    cache_hint="",
    cache_policy=None,
):
    _choice("space", space, _PTX_MEM_SPACE | {"local", "param::func"})
    _choice("cop", cop, _PTX_ST_COP)
    _choice("vec", vec, _PTX_ST_VEC)
    _choice("ptx_type", ptx_type, _PTX_SCALAR_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_st",
        address,
        *values,
        cache_policy,
        int(bool(weak)),
        space,
        cop,
        vec,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_st_bulk(ptr, num_bytes, *, weak=False, space="shared::cta"):
    if space not in ("", "shared::cta"):
        raise ValueError(f"Unsupported PTX st.bulk space {space!r}")
    return call_intrin("", "tirx.ptx_st_bulk", ptr, num_bytes, int(bool(weak)), space)


def ptx_prefetch_tensormap(tensormap_addr, space=""):
    _choice("space", space, _PTX_PREFETCH_TENSORMAP_SPACE)
    return call_intrin("", "tirx.ptx_prefetch_tensormap", tensormap_addr, space)


def ptx_mbarrier_test_wait_parity(barrier, phase, *, sem="", scope="", space="shared::cta"):
    if sem not in ("", "acquire", "relaxed"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity sem {sem!r}")
    if scope not in ("", "cta", "cluster"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity scope {scope!r}")
    if bool(sem) != bool(scope):
        raise ValueError("mbarrier.test_wait.parity sem and scope must be set together")
    if space not in ("shared", "shared::cta"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity space {space!r}")
    return call_intrin(
        "uint32", "tirx.ptx_mbarrier_test_wait_parity", barrier, phase, sem, scope, space
    )


def ptx_cp_async_bulk_g2s_cta(
    dst_ptr,
    src_ptr,
    num_bytes,
    mbarrier_ptr,
    *,
    cache_hint="",
    cache_policy=None,
    ignore_oob=False,
    ignore_bytes_left=0,
    ignore_bytes_right=0,
):
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_g2s_cta",
        dst_ptr,
        src_ptr,
        num_bytes,
        ignore_bytes_left,
        ignore_bytes_right,
        mbarrier_ptr,
        cache_policy,
        int(has_cache_policy),
        int(bool(ignore_oob)),
    )


def ptx_cp_async_bulk_g2s_cluster(
    dst_ptr,
    src_ptr,
    num_bytes,
    mbarrier_ptr,
    *,
    cache_hint="",
    cache_policy=None,
    multicast=False,
    cta_mask=0,
):
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_g2s_cluster",
        dst_ptr,
        src_ptr,
        num_bytes,
        mbarrier_ptr,
        cta_mask,
        cache_policy,
        int(has_cache_policy),
        int(bool(multicast)),
    )


def ptx_cp_async_bulk_s2s_cluster(dst_ptr, src_ptr, num_bytes, mbarrier):
    return call_intrin(
        "", "tirx.ptx_cp_async_bulk_s2s_cluster", dst_ptr, src_ptr, num_bytes, mbarrier
    )


def ptx_cp_async_bulk_s2g(
    dst_ptr, src_ptr, num_bytes, *, cache_hint="", cache_policy=None, cp_mask=False, byte_mask=0
):
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx_cp_async_bulk_s2g",
        dst_ptr,
        src_ptr,
        num_bytes,
        byte_mask,
        cache_policy,
        int(has_cache_policy),
        int(bool(cp_mask)),
    )


def ptx_fns_b32(mask, base, offset):
    return call_intrin("uint32", "tirx.ptx_fns_b32", mask, base, offset)


def ptx_add_rn_f32_bf16(acc, x):
    return call_intrin("float32", "tirx.ptx_add_rn_f32_bf16", acc, x)


def cuda_uint_as_float(bits):
    return call_intrin("float32", "tirx.cuda_uint_as_float", bits)


def cuda_float_as_uint(x):
    return call_intrin("uint32", "tirx.cuda_float_as_uint", x)


def cuda_ballot_sync(mask, pred):
    return call_intrin("uint32", "tirx.cuda_ballot_sync", mask, pred)


def cuda_ffs_u32(value):
    return call_intrin("int32", "tirx.cuda_ffs_u32", value)


def cuda_reduce_add_sync_u32(mask, value):
    return call_intrin("uint32", "tirx.cuda_reduce_add_sync_u32", mask, value)


def cuda_reduce_min_sync_u32(mask, value):
    return call_intrin("uint32", "tirx.cuda_reduce_min_sync_u32", mask, value)


def cuda_clock64():
    return call_intrin("uint64", "tirx.cuda_clock64")


def cuda_make_float2(x, y):
    return call_intrin("uint64", "tirx.cuda_make_float2", x, y)


def cuda_float2_x(packed):
    return call_intrin("float32", "tirx.cuda_float2_x", packed)


def cuda_float2_y(packed):
    return call_intrin("float32", "tirx.cuda_float2_y", packed)


def cuda_fmul2_rn(a, b):
    return call_intrin("uint64", "tirx.cuda_fmul2_rn", a, b)


def cuda_fadd2_rn(a, b):
    return call_intrin("uint64", "tirx.cuda_fadd2_rn", a, b)


def cuda_float22bfloat162_rn(v0, v1):
    return call_intrin("uint32", "tirx.cuda_float22bfloat162_rn", v0, v1)


def cuda_float22bfloat162_rn_from_float2(packed):
    return call_intrin("uint32", "tirx.cuda_float22bfloat162_rn_from_float2", packed)


def cuda_bfloat1622float2(packed):
    return call_intrin("uint64", "tirx.cuda_bfloat1622float2", packed)


def cuda_hmin2(a, b):
    return call_intrin("uint32", "tirx.cuda_hmin2", a, b)


def cuda_hmax2(a, b):
    return call_intrin("uint32", "tirx.cuda_hmax2", a, b)


def cuda_fp8x4_e4m3_from_float4(x, y, z, w):
    return call_intrin("uint32", "tirx.cuda_fp8x4_e4m3_from_float4", x, y, z, w)


def ptx_map_shared_rank(ptr, rank):
    """TVM intrinsic to call ptx map_shared_rank instruction

    Parameters
    ----------
    ptr: PrimExpr
        The generic pointer to the local shared memory, handle type

    rank: int
        The rank of the distributed shared memory.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return ptx_mapa(ptr, rank, space="", ptx_type="u64", return_type="uint64")


def ptx_mapa(ptr, rank, *, space="", ptx_type="u64", return_type="uint64"):
    """TVM intrinsic for PTX ``mapa{.space}.type d, a, b``."""
    if space not in ("", "shared::cluster"):
        raise ValueError(f"Unsupported mapa space {space!r}")
    if ptx_type not in ("u32", "u64"):
        raise ValueError(f"Unsupported mapa type {ptx_type!r}")
    return call_intrin(return_type, "tirx.ptx_mapa", ptr, rank, space, ptx_type, return_type)


def cuda_atomic_cas(ptr, old_val, new_val):
    """TVM intrinsic to call cuda atomic cas instruction

    Parameters
    ----------
    ptr: PrimExpr
        The pointer to the memory location.

    old_val: PrimExpr
        The old value.

    new_val: PrimExpr
        The new value.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    old_val = tir.convert(old_val)
    return call_intrin(old_val.dtype, "tirx.cuda_atomic_cas", ptr, old_val, new_val)


def thread_return():
    """TVM intrinsic to call thread_return()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.thread_return")


def continue_loop(span=None):
    """Create a tir intrinsic call to represent continue expression

    Parameters
    ----------
    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    ret : PrimExpr
        The continue expression
    """

    return _ffi_api.continue_loop(span)


def break_loop(span=None):
    """Create a tir intrinsic call to represent break expression

    Parameters
    ----------
    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    ret : PrimExpr
        The break expression
    """

    return _ffi_api.break_loop(span)


########################################################
# NVSHMEM builtins
########################################################


def nvshmem_my_pe():
    """TVM intrinsic to call nvshmem_my_pe()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("int32", "tirx.nvshmem_my_pe")


def nvshmem_n_pes():
    """TVM intrinsic to call nvshmem_n_pes()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("int32", "tirx.nvshmem_n_pes")


def nvshmem_getmem_nbi(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_getmem_nbi()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: PrimExpr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per thread.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin("", "tirx.nvshmem_getmem_nbi", dst, src, nelems, pe)


def nvshmem_putmem_nbi(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_putmem_nbi()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the destination data object.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per thread.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem_putmem_nbi", dst, src, nelems, pe)


def nvshmem_getmem_nbi_warp(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_getmem_nbi_warp()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: PrimExpr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per warp.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin("", "tirx.nvshmem_getmem_nbi_warp", dst, src, nelems, pe)


def nvshmem_putmem_nbi_warp(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_putmem_nbi_warp()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the destination data object.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per warp.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem_putmem_nbi_warp", dst, src, nelems, pe)


def nvshmem_getmem_nbi_block(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_getmem_nbi_block()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: PrimExpr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per block.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin("", "tirx.nvshmem_getmem_nbi_block", dst, src, nelems, pe)


def nvshmem_putmem_nbi_block(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_putmem_nbi_block()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the destination data object.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per block.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem_putmem_nbi_block", dst, src, nelems, pe)


def nvshmem_signal_op(sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_signal_op()

    Parameters
    ----------
    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal word to be updated, must be uint64_t*.

    signal: uint64_t
        The value used to update sig_addr.

    sig_op: str
        Operation used to update sig_addr with signal, typical sig_op values are "set" and "add".

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    _choice("sig_op", sig_op, _NVSHMEM_SIG_OP)
    return call_intrin("", "tirx.nvshmem_signal_op", sig_addr, signal, sig_op, pe)


def nvshmem_wait_until(ivar, cmp, cmp_value, type="uint64_t"):
    """TVM intrinsic to call nvshmem_wait_until()

    Parameters
    ----------
    ivar: PrimExpr
        The pointer to the symmetric address of a remotely accessible data object, must be TYPE*.

    cmp: str
        The compare operator that compares ivar with cmp_value.

    cmp_value: TYPE
        The value to be compared with ivar.

    type: str
        The TYPE of ivar and cmp_value.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    _choice("cmp", cmp, _NVSHMEM_CMP)
    return call_intrin("", "tirx.nvshmem_wait_until", ivar, cmp, cmp_value, type)


def nvshmem_quiet():
    """TVM intrinsic to call nvshmem_quiet()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem_quiet")


def nvshmem_putmem_signal_nbi(dst, src, nelems, sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per thread.

    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "", "tirx.nvshmem_putmem_signal_nbi", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_putmem_signal_nbi_warp(dst, src, nelems, sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi_warp()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per warp.

    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "", "tirx.nvshmem_putmem_signal_nbi_warp", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_putmem_signal_nbi_block(dst, src, nelems, sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi_block()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per block.

    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "", "tirx.nvshmem_putmem_signal_nbi_block", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_fence():
    """TVM intrinsic to call nvshmem_fence()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem_fence")


def nvshmem_barrier_all():
    """TVM intrinsic to call nvshmem_barrier_all()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem_barrier_all")


########################################################
# NKI builtins
########################################################


def nki_load(res, data):
    """TVM intrinsic to call nki load instruction

    Parameters
    ----------
    res : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_load", res, data)


def nki_store(res, data):
    """TVM intrinsic to call nki store instruction

    Parameters
    ----------
    res : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_store", res, data)


def nki_tensor_copy(res, data):
    """TVM intrinsic to call nki tensor copy instruction

    Parameters
    ----------
    res : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_tensor_copy", res, data)


def nki_matmul(res, lhs, rhs, accum=True):
    """TVM intrinsic to call nki matmul instruction

    Parameters
    ----------
    res : BufferLoad
        The result buffer.

    lhs: BufferLoad
        The left hand side buffer.

    rhs: BufferLoad
        The right hand side buffer.

    accum: bool
        Whether to accumulate the result.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_matmul", res, lhs, rhs, accum)


def nki_activation(result, data, opcode, bias=0.0, scale=1.0):
    """TVM intrinsic to call nki activation instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    opcode: str
        The opcode.

    bias: PrimExpr
        The bias.

    scale: PrimExpr
        The scale.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_activation", result, data, opcode, bias, scale)


def nki_reciprocal(result, data):
    """TVM intrinsic to call nki reciprocal instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_reciprocal", result, data)


def nki_tensorreduce(result, data, opcode, negate, *axes):
    """TVM intrinsic to call nki tensorreduce instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    opcode: str
        The opcode.

    negate: bool
        Whether to negate the result.

    axes: Tuple[int]
        The axes to reduce over.


    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_tensorreduce", result, data, opcode, negate, *axes)


def nki_tensortensor(result, operand0, operand1, opcode):
    """TVM intrinsic to call nki tensortensor instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    operand0: BufferLoad
        The first operand buffer.

    operand1: BufferLoad
        The second operand buffer.

    opcode: str
        The opcode.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_tensortensor", result, operand0, operand1, opcode)


def nki_tensorscalar(result, operand0, operand1, opcode, reverse=False):
    """TVM intrinsic to call nki tensorscalar instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    operand0: BufferLoad
        The first operand buffer.

    operand1: PrimExpr
        The second operand scalar.

    opcode: str
        The opcode.

    reverse: bool
        Whether to reverse the operands.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_tensorscalar", result, operand0, operand1, opcode, reverse)


def nki_memset(result, value):
    """TVM intrinsic to call nki memset instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    value: PrimExpr
        The value to set.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_memset", result, value)


def nki_activation_reduce(reduce_res, act_res, data, opcode, reduce_opcode, bias=0.0, scale=1.0):
    """TVM intrinsic to call nki activation reduce instruction

    act_res = act_op(data * scale + bias)
    reduce_res = reduce_op(act_res)

    Parameters
    ----------
    reduce_res : BufferLoad
        The result buffer of reduction.

    act_res : BufferLoad
        The result buffer of activation.

    data: BufferLoad
        The data buffer.

    opcode: str
        The opcode.

    reduce_opcode: str
        The reduce opcode.

    bias: PrimExpr
        The bias.

    scale: PrimExpr
        The scale.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "",
        "tirx.nki_activation_reduce",
        reduce_res,
        act_res,
        data,
        opcode,
        reduce_opcode,
        bias,
        scale,
    )


def nki_tensorscalar_reduce(
    reduce_res, tensorscalar_res, operand0, operand1, opcode, reduce_opcode, reverse=False
):
    """TVM intrinsic to call nki tensorscalar reduce instruction

    tensorscalar_res = tensorscalar_op(operand0, operand1)
    reduce_res = reduce_op(tensorscalar_res)

    Parameters
    ----------
    reduce_res : BufferLoad
        The result buffer of reduction.

    tensorscalar_res : BufferLoad
        The result buffer of tensorscalar.

    operand0: BufferLoad
        The first operand buffer.

    operand1: PrimExpr
        The second operand scalar.

    opcode: str
        The opcode.

    reduce_opcode: str
        The reduce opcode.

    reverse: bool
        Whether to reverse the operands of tensorscalar.
    """
    return call_intrin(
        "",
        "tirx.nki_tensorscalar_reduce",
        reduce_res,
        tensorscalar_res,
        operand0,
        operand1,
        opcode,
        reduce_opcode,
        reverse,
    )


def nki_identity(result, size):
    """TVM intrinsic to call nki identity instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    size: PrimExpr
        The size of the identity tensor.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_identity", result, size)


def nki_scalar_tensor_tensor(
    result, data, operand0, operand1, opcode0, opcode1, reverse0=False, reverse1=False
):
    """TVM intrinsic to call nki scalar tensor tensor instruction
    (data op0 operand0) op1 (operand1) , where op0 is tensor-scalar and op1 is tensor-tensor

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    operand0: PrimExpr
        The first operand scalar.

    operand1: BufferLoad
        The second operand buffer.

    opcode0: str
        The first opcode.

    opcode1: str
        The second opcode.

    reverse0: bool
        Whether to reverse the first operand.

    reverse1: bool
        Whether to reverse the second operand.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "",
        "tirx.nki_scalar_tensor_tensor",
        result,
        data,
        operand0,
        operand1,
        opcode0,
        opcode1,
        reverse0,
        reverse1,
    )


def nki_scalar_tensor_scalar(
    result, data, operand0, operand1, opcode0, opcode1, reverse0=False, reverse1=False
):
    """TVM intrinsic to call nki scalar tensor scalar instruction
    (data op0 operand0) op1 (operand1) , where op0 and op1 are tensor-scalar

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    data: BufferLoad
        The data buffer.

    operand0: PrimExpr
        The first operand scalar.

    operand1: PrimExpr
        The second operand scalar.

    opcode0: str
        The first opcode.

    opcode1: str
        The second opcode.

    reverse0: bool
        Whether to reverse the first operand.

    reverse1: bool
        Whether to reverse the second operand.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        "",
        "tirx.nki_scalar_tensor_scalar",
        result,
        data,
        operand0,
        operand1,
        opcode0,
        opcode1,
        reverse0,
        reverse1,
    )


def nki_affine_select(result, pred, true_value, false_value):
    """TVM intrinsic to call nki affine select instruction

    Parameters
    ----------
    result : BufferLoad
        The result buffer.

    pred: PrimExpr
        The predicate.

    true_value: PrimExpr
        The true value.

    false_value: PrimExpr
        The false value.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.nki_affine_select", result, pred, true_value, false_value)
