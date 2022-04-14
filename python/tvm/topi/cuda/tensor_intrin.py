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
# pylint: disable=invalid-name, unnecessary-lambda, too-many-arguments
"""Tensor intrinsics on CUDA."""
import tvm
from tvm import te
from ..utils import is_target


def dp4a(x_scope="local", y_scope="local", z_scope="local", dtypes=("int8", "int8")):
    """
    Int8 dot product reduced by every 4 elements using __dp4a

    Parameters
    ----------
    x_scope : str, optional
        The storage scope of buffer for lhs
    y_scope : str, optional
        The storage scope of buffer for rhs
    z_scope : str, optional
        The storage scope of buffer for result
    dtypes:  tuple of strs, optional
        The dtype of x and y

    Returns
    -------
    intrin : TensorIntrin
        The dp4a TensorIntrin that can be used in tensorizing schedule.
    """

    n = 4  # dp4a requires operands packed by 4
    result_dtype = "int32" if dtypes[1] == "int8" else "uint32"

    x = te.placeholder((n,), name="x", dtype=dtypes[0])
    y = te.placeholder((n,), name="y", dtype=dtypes[1])

    k = te.reduce_axis((0, n), name="rc")

    z = te.compute(
        (1,), lambda i: te.sum(x[k].astype(result_dtype) * y[k].astype(result_dtype), axis=[k])
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            xx, yy = ins
            zz = outs[0]
            zz_dtype = zz.dtype

            if index == 1:
                return zz.vstore(0, tvm.tir.const(0, zz_dtype))

            ib = tvm.tir.ir_builder.create()

            vec_x_dtype = "int8x4" if xx.dtype == "int8" else "uint8x4"
            vec_y_dtype = "int8x4" if yy.dtype == "int8" else "uint8x4"

            vec_x = xx.vload(0, dtype=vec_x_dtype)
            vec_y = yy.vload(0, dtype=vec_y_dtype)
            prev_z = 0 if index == 0 else zz.vload(0)

            if is_target("rocm"):
                # TODO(masahi): Here we are assuming that we are compiling for gfx10 or later
                # We can refine the specification for dot product on rocm if needed later.

                # We can just use "llvm.amdgcn.udot4" for u8u8u32, but it is not tested.
                assert (
                    dtypes[0] == "int8" and dtypes[0] == "int8"
                ), "u8u8u32 dot product for rocm not supported yet"

                new_z = tvm.tir.call_llvm_pure_intrin(
                    zz_dtype,
                    "llvm.amdgcn.sdot4",
                    tvm.tir.const(4, "uint32"),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_x),
                    tvm.tir.call_intrin("int32", "tir.reinterpret", vec_y),
                    prev_z,
                    True,
                )
            else:
                new_z = tvm.tir.call_pure_extern(zz_dtype, "__dp4a", vec_x, vec_y, prev_z)

            ib.emit(zz.vstore(0, new_z))

            return ib.get()

        return _instr(0), _instr(1), _instr(2)  # body, reset, update

    default_buffer_params = {"data_alignment": 4, "offset_factor": 1}
    scopes = {x: x_scope, y: y_scope, z: z_scope}
    binds = {
        t: tvm.tir.decl_buffer(
            t.shape, t.dtype, t.op.name, scope=scopes[t], **default_buffer_params
        )
        for t in [x, y, z]
    }

    return te.decl_tensor_intrin(
        z.op, _intrin_func, binds=binds, default_buffer_params=default_buffer_params
    )


def intrin_wmma_load_matrix_A(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype):
    """Intrin function for loading data from shared memory to wmma.matrix_a"""
    wmma_m, wmma_n, wmma_k = shape

    A = te.placeholder(A_shape, name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", strides=strides_from, data_alignment=32, offset_factor=8
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_a",
        strides=strides_dst,
        data_alignment=32,
        offset_factor=8,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_k
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BA.access_ptr("r"),
                strides_from[0],
                layout,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_W(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype):
    """Intrin function for loading data from shared memory to wmma.matrix_b"""
    wmma_m, wmma_n, wmma_k = shape

    A = te.placeholder(A_shape, name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", strides=strides_from, data_alignment=32, offset_factor=8
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_b",
        strides=strides_dst,
        data_alignment=32,
        offset_factor=8,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_n * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_n
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BA.access_ptr("r"),
                strides_from[0],
                layout,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_store_matrix(strides_dst, strides_from, shape, out_dtype, A_shape, C_shape):
    """Intrin function for storing the results from wmma.accumulator to shared"""
    wmma_m, wmma_n, wmma_k = shape
    A = te.placeholder(A_shape, name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="wmma.accumulator",
        strides=strides_from,
        data_alignment=32,
        offset_factor=8,
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="shared", strides=strides_dst, data_alignment=32, offset_factor=8
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_n
        warp_index = BA.elem_offset // row + BA.elem_offset % row // wmma_n
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BC.access_ptr("w"),
                strides_dst[0],
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(AL_gemm, WL_gemm, CL_compute, strides_A, strides_W, strides_Conv, shape):
    """Intrin for wmma fill_fragment and mma_sync

    Parameters
    ----------
    AL_gemm : tvm.te.placeholder
        wmma matrix A
    WL_gemm : tvm.te.placeholder
        wmma matrix B
    CL_compute : tvm.te.compute
        The definition of wmma gemm
    """
    wmma_m, wmma_n, wmma_k = shape
    A = AL_gemm
    B = WL_gemm
    C = CL_compute

    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope="wmma.matrix_a",
        data_alignment=32,
        offset_factor=8,
        strides=strides_A,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope="wmma.matrix_b",
        data_alignment=32,
        offset_factor=8,
        strides=strides_W,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=32,
        offset_factor=8,
        strides=strides_Conv,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def warp_idnex(offset, row, col):
            row = row * col
            return offset // row + offset % row // col

        warp_index_A = warp_idnex(BA.elem_offset, wmma_m, wmma_k)
        warp_index_B = warp_idnex(BB.elem_offset, wmma_k, wmma_n)
        warp_index_C = warp_idnex(BC.elem_offset, wmma_m, wmma_n)

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    wmma_m,
                    wmma_n,
                    wmma_k,
                    warp_index_C,
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    warp_index_C,
                    BA.data,
                    warp_index_A,
                    BB.data,
                    warp_index_B,
                    BC.data,
                    warp_index_C,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})
