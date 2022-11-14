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
# pylint: disable=redefined-builtin, invalid-name
"""Operators used in TIR expression."""
import warnings
from typing import Any, Optional

import tvm._ffi
from tvm.ir import Array, Op, PrimExpr
from tvm.ir.base import Span
from tvm.runtime import const, convert

from . import _ffi_api
from .buffer import Buffer
from .expr import Call, CommReducer, IntImm, PrimExprWithOp, StringImm, Var


def _pack_buffer(buf, span=None):
    """Build intrinsics that packs the buffer."""
    shape = Call("handle", "tir.tvm_stack_make_shape", buf.shape, span)
    strides = Call("handle", "tir.tvm_stack_make_shape", buf.strides, span) if buf.strides else 0
    pack_args = [
        buf.data,
        shape,
        strides,
        len(buf.shape),
        const(0, dtype=buf.dtype),
        buf.elem_offset,
    ]
    return Call("handle", Op.get("tir.tvm_stack_make_array"), pack_args, span)


def call_packed_lowered(*args, span=None):
    """Lowered version of call packed.
    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.
    When the argument is Buffer, the corresponding PackedFunc
    will recieve an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is NDArray.

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
    return Call("int32", Op.get("tir.tvm_call_packed_lowered"), call_args, span)


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
    return Call("int32", Op.get("tir.tvm_call_cpacked_lowered"), call_args, span)


def call_packed(*args, span=None):
    """Build expression by call an external packed function.

    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.

    When the argument is Buffer, the corresponding PackedFunc
    will receive an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is NDArray.

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
    return Call("int32", Op.get("tir.tvm_call_packed"), call_args, span)


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
    return Call("int32", Op.get("tir.tvm_call_cpacked"), call_args, span)


def call_intrin(dtype, func_name, *args, span=None):
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

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return Call(dtype, func_name, convert(args), span)


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
    return Call(
        dtype, Op.get("tir.call_pure_extern"), convert((StringImm(func_name),) + args), span
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
    call : PrimExpr
        The call expression.
    """
    return Call(
        dtype, Op.get("tir.call_extern"), convert((StringImm(func_name),) + args), span=span
    )


def call_llvm_intrin(dtype, name, *args, span=None):
    """Build expression by calling a llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Poistional arguments.

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
        warnings.warn(f"Unknown llvm intrinsic function {name}, falling back to 0")
    return call_intrin(
        dtype,
        Op.get("tir.call_llvm_intrin"),
        tvm.tir.const(llvm_id, "uint32"),
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
       Poistional arguments.

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
        warnings.warn(f"Unknown llvm intrinsic function {name}, falling back to 0")
    return call_intrin(
        dtype,
        Op.get("tir.call_llvm_pure_intrin"),
        tvm.tir.const(llvm_id, "uint32"),
        *args,
        span=span,
    )


def tvm_check_return(expected, return_unexpected, nested_call):
    """Return new on stack dtype[num]
    Parameters
    ----------
    expected : int
        The expected return code.
    return_unexpected : int
        The unexpected return code.
    nested_call : PrimExpr
        The call expression to check return.
    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int32", "tir.tvm_check_return", expected, return_unexpected, nested_call)


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
    return call_intrin("handle", "tir.tvm_stack_alloca", dtype_str, num)


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
    return call_intrin("handle", "tir.tvm_stack_make_shape", *args)


def tvm_stack_make_array(data, shape, strides, ndim, arr_dtype, elem_offset):
    """Allocate a NDArray(DLTensor) on stack, return the handle

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
        "handle", "tir.tvm_stack_make_array", data, shape, strides, ndim, arr_dtype, elem_offset
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
    return call_intrin("bool", "tir.assume", cond)


def undef():
    """Returns an initialized but arbitrary value

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int32", "tir.undef")


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
    return call_intrin("handle", "tir.start_profile_intrinsic", id)


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
    return call_intrin("handle", "tir.end_profile_intrinsic", id)


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
    return call_intrin("handle", "tir.tvm_tuple", *value)


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
    return call_intrin(dtype, "tir.tvm_struct_get", arr, index, field)


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
    return call_intrin("handle", "tir.tvm_struct_set", arr, index, field, value)


def address_of(buffer_load, span=None):
    """Returns the address of an element in the buffer

    Parameters
    ----------
    buffer_load: BufferLoad
        The buffer load.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tir.address_of", buffer_load, span=span)


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
    return call_intrin("handle", "tir.lookup_param", param_name, span=span)


def tvm_thread_allreduce(*freduce_args):
    """
    Parameters
    ----------
    freduce_args : Expr
        The args.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("handle", "tir.tvm_thread_allreduce", *freduce_args)


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
    return call_intrin(dtype, "tir.type_annotation")


def tvm_access_ptr(ptype, data, offset, extent, rw_mask):
    """Get head access address with memory access pattern info

    Parameters
    ----------
    ptype : Expr
        The data type of pointer.

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
    return call_intrin("handle", "tir.tvm_access_ptr", ptype, data, offset, extent, rw_mask)


def tvm_throw_last_error():
    """Throw TVMGetLastError()

    Returns
    -------
    ret : PrimExpr
        The return expression
    """
    return call_intrin("handle", "tir.tvm_throw_last_error")


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
        "handle",
        "tir.tvm_load_matrix_sync",
        fragment,
        m,
        n,
        k,
        index,
        buffer_ptr,
        stride,
        layout,
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
        "tir.tvm_mma_sync",
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
        "tir.tvm_bmma_sync",
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
    return call_intrin(
        "handle",
        "tir.tvm_fill_fragment",
        fragment,
        m,
        n,
        k,
        index,
        value,
    )


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
        "handle",
        "tir.tvm_store_matrix_sync",
        fragment,
        m,
        n,
        k,
        index,
        buffer_ptr,
        stride,
        layout,
    )


def ptx_mma(
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
    saturate,
    operator=None,
):
    """TVM intrinsic for ptx tensor core mma instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

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
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment A.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    saturate : bool
        The optional saturation at the output.


    operator : Optional[Literal["xor", "and"]]
        The 1-bit operator.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if operator is None:
        return call_intrin(
            dtype,
            "tir.ptx_mma",
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
            saturate,
        )
    return call_intrin(
        dtype,
        "tir.ptx_mma",
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
        saturate,
        operator,
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
        "tir.ptx_mma_sp",
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
    return call_intrin(
        dtype,
        "tir.mma_store",
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
    return call_intrin(
        dtype,
        "tir.mma_fill",
        local_size,
        local_ptr,
        offset,
    )


def ptx_ldmatrix(dtype, trans, num, type, local_ptr, local_offset, smem_ptr, smem_offset):
    """TVM intrinsic for ptx load matrix from shared memory
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Parameters
    ----------
    dtype : str
       The data type of the result.

    trans : bool
        The matrix is loaded in column-major format.

    num : IntImm
        The number of matrices.

    type : Literal[".b16"]
        The data type of the matrices.

    local_ptr : Var
        The local pointer variable.

    local_offset : Expr
        The offset of local pointer.

    smem_ptr : Var
        The shared memory pointer variable.

    smem_offset : Expr
        The offset of shared memort pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        "tir.ptx_ldmatrix",
        trans,
        num,
        type,
        local_ptr,
        local_offset,
        smem_ptr,
        smem_offset,
    )


def ptx_cp_async(dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes):
    """TVM intrinsic for ptx async copy from global to shared memory
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

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

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype, "tir.ptx_cp_async", shared_ptr, shared_offset, global_ptr, global_offset, bytes
    )


def ptx_commit_group():
    """TVM intrinsic for ptx async copy commit
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tir.ptx_commit_group")


def ptx_wait_group(num):
    """TVM intrinsic for ptx async copy wait
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group

    Parameters
    ----------
    num : int
        The number of the most recent uncommitted pending cp.async groups to wait.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tir.ptx_wait_group", num)


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
    return call_intrin(dtype, "tir.vectorlow", vec)


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
    return call_intrin(dtype, "tir.vectorhigh", vec)


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
    return call_intrin(dtype, "tir.vectorcombine", vec1, vec2)


def ret(val):
    """Create a tir return expression

    Parameters
    ----------
    val : Expr
        The returned tir expression, whose data type is int, float or void pointer.

    Returns
    -------
    ret : PrimExpr
        The return expression
    """
    return call_intrin(val.dtype, "tir.ret", val)


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


@tvm._ffi.register_func("tvm.default_trace_action")
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
    tvm.tir.call_packed : Creates packed function.
    """
    if not isinstance(args, list):
        raise Exception("tvm.tir.trace consumes the args as list type")
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    call_args.insert(0, trace_action)
    return tvm.tir.Call(args[-1].dtype, Op.get("tir.tvm_call_trace_packed"), call_args)


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


def max_value(dtype: str, span: Optional[Span] = None) -> Any:
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


def infinity(dtype: str, span: Optional[Span] = None) -> Any:
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


def reinterpret(dtype, value) -> Any:
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
    return call_intrin(dtype, "tir.reinterpret", value)


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
    return call_intrin(x.dtype, "tir.exp", x)


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
    return call_intrin(x.dtype, "tir.exp2", x)


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
    return call_intrin(x.dtype, "tir.exp10", x)


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
    return call_intrin(x.dtype, "tir.erf", x)


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
    return call_intrin(x.dtype, "tir.tanh", x)


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
    return call_intrin(x.dtype, "tir.sigmoid", x)


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
    return call_intrin(x.dtype, "tir.log", x)


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
    return call_intrin(x.dtype, "tir.log2", x)


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
    return call_intrin(x.dtype, "tir.log10", x)


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
    return call_intrin(x.dtype, "tir.log1p", x)


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
    return call_intrin(x.dtype, "tir.tan", x)


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
    return call_intrin(x.dtype, "tir.cos", x)


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
    return call_intrin(x.dtype, "tir.cosh", x)


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
    return call_intrin(x.dtype, "tir.acos", x)


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
    return call_intrin(x.dtype, "tir.acosh", x)


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
    return call_intrin(x.dtype, "tir.sin", x)


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
    return call_intrin(x.dtype, "tir.sinh", x)


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
    return call_intrin(x.dtype, "tir.asin", x)


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
    return call_intrin(x.dtype, "tir.asinh", x)


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
    return call_intrin(x.dtype, "tir.atan", x)


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
    return call_intrin(x.dtype, "tir.atanh", x)


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
    return call_intrin(x1.dtype, "tir.atan2", x1, x2)


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
    return call_intrin(x.dtype, "tir.sqrt", x)


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
    return call_intrin(x.dtype, "tir.rsqrt", x)


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
    return call_intrin("int32", "tir.clz", x)


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
    return call_intrin(x1.dtype, "tir.nextafter", x1, x2)  # type: ignore


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
    return call_intrin(x1.dtype, "tir.hypot", x1, x2)  # type: ignore


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
    return call_intrin(x1.dtype, "tir.copysign", x1, x2)  # type: ignore


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
    return call_intrin(x1.dtype, "tir.ldexp", x1, x2)  # type: ignore


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
    return call_intrin("bool", "tir.isnullptr", x, span=span)  # type: ignore


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
    return _ffi_api._OpPow(convert(x), convert(y), span)  # type: ignore


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
    return call_intrin(x.dtype, "tir.popcount", x)


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
    return call_intrin("int32", "tir.q_multiply_shift", x, y, q, s)


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
        "tir.q_multiply_shift_per_axis",
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
    return call_intrin(x.dtype, "tir.fmod", x, y)


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
    return _ffi_api._OpIfThenElse(convert(cond), convert(t), convert(f), span)  # type: ignore


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
            lambda t: tvm.tir.const(0, dtype=t), name="mysum")
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
        expr = convert(expr)
        if init is not None:
            init = convert(init)
        if isinstance(expr, Array):
            size = len(expr)
            larr = []
            rarr = []
            dtypes = []
            for i in range(size):
                dtype = expr[i].dtype
                dtypes.append(dtype)
                lname = code.co_varnames[0] + "_" + str(i)
                larr.append(Var(lname, dtype))
                rname = code.co_varnames[1] + "_" + str(i)
                rarr.append(Var(rname, dtype))
            if init is not None:
                init = convert(init)
                assert isinstance(init, Array)
                assert len(init) == size
                for init_i in range(size):
                    init_i = convert(init_i)
                    assert isinstance(
                        init_i, (tvm.tir.ProducerLoad, tvm.tir.IntImm, tvm.tir.FloatImm)
                    )
            else:
                init = convert([])
            lhs = convert(larr)
            rhs = convert(rarr)
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
            lhs = convert([lvar])
            rhs = convert([rvar])
            expr = convert([expr])
            if init is not None:
                assert isinstance(init, (tvm.tir.ProducerLoad, tvm.tir.IntImm, tvm.tir.FloatImm))
                init = convert([init])
        result = convert(result)
        id_elem = convert(id_elem)
        combiner = CommReducer(lhs, rhs, result, id_elem)
        axis = convert(axis if isinstance(axis, (list, tuple)) else [axis])
        if where is None:
            where = convert(True)
        if init is None:
            outputs = tuple(
                tvm.tir.Reduce(combiner, expr, axis, where, i, convert([])) for i in range(size)
            )
        else:
            outputs = tuple(
                tvm.tir.Reduce(combiner, expr, axis, where, i, init) for i in range(size)
            )
        return outputs[0] if size == 1 else outputs

    # pylint: disable=keyword-arg-before-vararg
    def reducer(expr, axis, where=None, init=None, *args):
        if isinstance(axis, (tvm.tir.IterVar, list, tuple)):
            assert not args
            return _make_reduce(expr, axis, where, init)
        if where is None:
            assert not args
            return _reduce_directly(expr, axis)
        return _reduce_directly(expr, axis, where, *args)

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
                # tvm.{0} represents tvm.te.{0} or tvm.tir.{0}.
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
        "tir.TVMBackendAllocWorkspace",
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
    return call_intrin("int32", "tir.TVMBackendFreeWorkspace", device_type, device_id, ptr)


# pylint: disable=unnecessary-lambda
sum = comm_reducer(lambda x, y: x + y, lambda t: const(0, dtype=t), name="sum")
min = comm_reducer(lambda x, y: _ffi_api._OpMin(x, y, None), max_value, name="min")  # type: ignore
max = comm_reducer(lambda x, y: _ffi_api._OpMax(x, y, None), min_value, name="max")  # type: ignore
