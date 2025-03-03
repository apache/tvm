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
# pylint: disable=invalid-name
"""Legalization functions for DLTensor inspection."""

import enum

from tvm.script import tir as T

from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


class TVMStructFieldKind(enum.IntEnum):
    """Equivalent to tvm::tir::builtin::TVMStructFieldKind

    This does not use `enum.auto()` to define the values, because
    `enum.auto()` starts from 1, and this must match the C++
    definition which starts from 0.
    """

    kArrAddr = 0
    kArrData = 1
    kArrShape = 2
    kArrStrides = 3
    kArrNDim = 4
    kArrTypeCode = 5
    kArrTypeBits = 6
    kArrTypeLanes = 7
    kArrByteOffset = 8
    kArrDeviceId = 9
    kArrDeviceType = 10
    kArrKindBound_ = 11
    kTVMValueContent = 12
    kTVMValueKindBound_ = 13


@register_legalize("relax.inspect.tensor_stride_i")
def _tensor_stride_i(bb: BlockBuilder, call: Call) -> Expr:
    @T.prim_func(private=True)
    def _get_tensor_stride_i(dlpack_handle: T.handle, axis: T.int64) -> T.int64:
        T.func_attr({"tir.is_host": T.bool(True), "tir.is_scheduled": T.bool(True)})
        assert T.int64(0) <= axis, "Specified axis may not be negative"
        ndim: T.int32 = T.tvm_struct_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kArrNDim), "int32"
        )
        assert axis < T.Cast(
            "int64", ndim
        ), "Specified axis may not be larger than the tensor's dimensionality"
        stride_ptr: T.handle("int64") = T.tvm_struct_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kArrStrides), "handle"
        )

        if T.isnullptr(stride_ptr):
            shape_ptr: T.handle("int64") = T.tvm_struct_get(
                dlpack_handle, 0, int(TVMStructFieldKind.kArrShape), "handle"
            )
            shape = T.decl_buffer(ndim, "int64", data=shape_ptr)

            product = T.decl_buffer([], "int64")
            product[()] = 1

            # TODO(Lunderberg): Add a TIR lowering pass to allow
            # ranges to start somewhere other than zero.  This loop
            # could then iterate on `range(axis+1, ndim)`.
            for dim_offset in range(ndim - (axis + 1)):
                dim = dim_offset + (axis + 1)
                product[()] = product[()] * shape[dim]

            return product[()]
        else:
            strides = T.decl_buffer(ndim, "int64", data=stride_ptr)
            stride: T.int64 = strides[axis]
            return stride

    gvar = bb.add_func(_get_tensor_stride_i, "_get_tensor_stride_i")
    return Call(gvar, call.args)


@register_legalize("relax.inspect.tensor_byte_offset")
def _tensor_byte_offset(bb: BlockBuilder, call: Call) -> Expr:
    @T.prim_func(private=True)
    def _get_tensor_byte_offset(dlpack_handle: T.handle) -> T.int64:
        T.func_attr({"tir.is_host": T.bool(True), "tir.is_scheduled": T.bool(True)})
        byte_offset: T.uint64 = T.tvm_struct_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kArrByteOffset), "uint64"
        )
        return byte_offset

    gvar = bb.add_func(_get_tensor_byte_offset, "_get_tensor_byte_offset")
    return Call(gvar, call.args)


@register_legalize("relax.inspect.tensor_elem_offset")
def _tensor_elem_offset(bb: BlockBuilder, call: Call) -> Expr:
    @T.prim_func(private=True)
    def _get_tensor_elem_offset(dlpack_handle: T.handle) -> T.int64:
        T.func_attr({"tir.is_host": T.bool(True), "tir.is_scheduled": T.bool(True)})
        byte_offset: T.uint64 = T.tvm_struct_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kArrByteOffset), "uint64"
        )
        scalar_bits: T.uint8 = T.tvm_struct_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kArrTypeBits), "uint8"
        )
        lanes: T.uint16 = T.tvm_struct_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kArrTypeLanes), "uint16"
        )
        bytes_per_element = T.ceildiv(scalar_bits.astype("uint64") * lanes.astype("uint64"), 8)
        elem_offset = byte_offset // bytes_per_element
        return elem_offset

    gvar = bb.add_func(_get_tensor_elem_offset, "_get_tensor_elem_offset")
    return Call(gvar, call.args)
