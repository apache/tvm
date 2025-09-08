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
# pylint: disable=invalid-name,line-too-long
"""Intrinsics for RISCV tensorization"""

import logging
import tvm_ffi

from tvm.runtime import DataType
from tvm.script import tir as T
from tvm.target.codegen import llvm_get_vector_width, target_has_features, Target
from .. import TensorIntrin

logger = logging.getLogger(__name__)


def get_max_elems(vlen: int, lmul: int, sew: int) -> int:
    """Returns number of elements of a given data type (SEW)
    that fits multiple (LMUL) of the vector registers (VLEN).

    Args:
        vlen (int): VLEN vector length in bits
        lmul (int): LMUL vector lenght multiplier
        sew (int): SEW standard (single) element width

    Returns:
        int: Number of elements
    """
    return (vlen // sew) * lmul


def rvv_vec_dot_product_kernels(
    n_elems: int,
    n_lanes: int,
    data_dtype: str,
    weight_dtype: str,
    out_dtype: str,
    lmul: int,
):
    """Dot product of vector and matrix rows using RISC-V vector instructions.

    These kernels takes two arrays A[ELEMS] and B[ELEMS][MACS] and computes
    dot product of A[ELEMS] with each row of B[LANES], accumulating results
    with C[LANES].

    The pseudo code is as follows:
    .. code-block:: c
        void vec_dot_prod(A[ELEMS], B[LANES][ELEMS], C[LANES]){
            for (j = 0; j < LANES; j++) {
                for (k = 0; k < ELEMS; k++) {
                    C[j] += A[k] * B[j][k]
                }
            }
        }
    """

    @T.prim_func
    def rvv_vec_dot_prod_desc(
        A: T.Buffer((n_elems,), data_dtype, offset_factor=1),
        B: T.Buffer((n_lanes, n_elems), weight_dtype, offset_factor=1),
        C: T.Buffer((n_lanes,), out_dtype, offset_factor=1),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:n_lanes], A[0:n_elems], B[0:n_lanes, 0:n_elems])
            T.writes(C[0:n_lanes])
            for j in T.serial(0, n_lanes):
                for k in T.serial(0, n_elems):
                    with T.block("update"):
                        vj, vk = T.axis.remap("SR", [j, k])
                        C[vj] = C[vj] + T.cast(A[vk], out_dtype) * T.cast(B[vj, vk], out_dtype)

    # LLVM only supports ELEN=32 or ELEN=64
    # https://llvm.org/docs//RISCV/RISCVVectorExtension.html
    d_dtype_lanes = (64 // DataType(data_dtype).bits) * lmul
    w_dtype_lanes = (64 // DataType(weight_dtype).bits) * lmul
    # reduction lanes narrows
    o_dtype_lanes = (64 // DataType(out_dtype).bits) * lmul // n_lanes
    # data type widening case
    o_dtype_lanes = max(o_dtype_lanes, 2)

    mask_args = () if data_dtype[0] in ("i", "u") else (T.uint64(7),)

    wide_dtype = out_dtype
    if DataType(out_dtype).bits > DataType(data_dtype).bits:
        wide_dtype = "".join(c for c in data_dtype if not c.isdigit())
        wide_dtype += str(DataType(data_dtype).bits * 2)

    # fmt: off
    @T.prim_func
    def rvv_vec_dot_prod_impl(
        A: T.Buffer((n_elems,), data_dtype, offset_factor=1),
        B: T.Buffer((n_lanes, n_elems), weight_dtype, offset_factor=1),
        C: T.Buffer((n_lanes,), out_dtype, offset_factor=1),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:n_lanes], A[0:n_elems], B[0:n_lanes, 0:n_elems])
            T.writes(C[0:n_lanes])

            vec_A = T.call_llvm_intrin(
                f"{data_dtype}xvscalex{d_dtype_lanes}",
                "llvm.riscv.vle",
                T.broadcast(T.Cast(data_dtype, 0), T.vscale() * d_dtype_lanes),
                T.tvm_access_ptr(T.type_annotation(data_dtype), A.data, 0, n_elems, 1),
                T.int64(n_elems))

            for i in range(n_lanes):
                with T.block("update"):
                    T.reads(B[i, 0:n_elems])
                    T.writes(C[i])

                    vec_B_row = T.call_llvm_intrin(
                        f"{weight_dtype}xvscalex{w_dtype_lanes}",
                        "llvm.riscv.vle",
                        T.broadcast(T.Cast(data_dtype, 0), T.vscale() * w_dtype_lanes),
                        T.tvm_access_ptr(T.type_annotation(weight_dtype), B.data, i * n_elems, n_elems, 1),
                        T.int64(n_elems))

                    product = T.call_llvm_intrin(
                        f"{wide_dtype}xvscalex{w_dtype_lanes}",
                        "llvm.riscv.vfmul" if out_dtype[0] == "f" else \
                        "llvm.riscv.vwmulsu" if (data_dtype[0] != weight_dtype[0]) else \
                        "llvm.riscv.vwmul",
                        T.broadcast(T.Cast(wide_dtype, 0), T.vscale() * w_dtype_lanes),
                        vec_B_row,
                        vec_A,
                        *mask_args,
                        T.uint64(n_elems))

                    ini_acc = T.call_llvm_intrin(
                        f"{out_dtype}xvscalex{o_dtype_lanes}",
                        "llvm.riscv.vle",
                        T.broadcast(T.Cast(out_dtype, 0), T.vscale() * o_dtype_lanes),
                        T.tvm_access_ptr(T.type_annotation(out_dtype), C.data, i, 1, 1),
                        T.int64(1))

                    red_sum = T.call_llvm_intrin(
                        f"{out_dtype}xvscalex{o_dtype_lanes}",
                        "llvm.riscv.vfredusum" if out_dtype[0] == "f" else \
                        "llvm.riscv.vwredsum",
                        T.broadcast(T.Cast(out_dtype, 0), T.vscale() * o_dtype_lanes),
                        product,
                        ini_acc,
                        *mask_args,
                        T.uint64(n_elems))

                    C[i] = T.call_llvm_intrin(
                        out_dtype,
                        "llvm.riscv.vfmv.f.s" if out_dtype[0] == "f" else \
                        "llvm.riscv.vmv.x.s",
                        red_sum)
    # fmt: on
    return rvv_vec_dot_prod_desc, rvv_vec_dot_prod_impl


@tvm_ffi.register_global_func("tir.tensor_intrin.register_rvv_isa_intrinsics")
def register_rvv_isa_intrinsics(target: Target, inventory_only=False) -> dict():
    """Register RISCV V (vector) intrinsics
    [x] Implementation follows version 1.0 vector specifications:
        https://github.com/riscvarchive/riscv-v-spec/releases/tag/v1.0

    Args:
        target (Target): TVM target
        inventory_only (bool): No registration inventory only

    Returns:
        dict(): A catalog with registered kernel names and properties
    """
    if not target_has_features("v", target):
        raise RuntimeError("Current target does not support `v` extension.")

    vlen = llvm_get_vector_width(target)
    # get maximum reduction lanes (without grouping)
    n_lanes = get_max_elems(vlen, lmul=1, sew=32)

    kernels_inventory = {}

    data_dtype = ["uint8", "int8", "float16", "float32"]
    weight_dtype = ["int8", "int8", "float16", "float32"]
    output_dtype = ["int32", "int32", "float16", "float32"]

    for d_dtype, w_dtype, o_dtype in zip(data_dtype, weight_dtype, output_dtype):
        # max elements to grouped registers
        max_elems = get_max_elems(vlen, lmul=8, sew=DataType(d_dtype).bits)
        # data widening halves available vector registers
        if DataType(o_dtype).bits > DataType(d_dtype).bits:
            max_elems //= 2
        # compute optimal LMUL for full load
        lmul = max_elems // (vlen // DataType(d_dtype).bits)

        n_elems = max_elems
        while n_elems >= 4:

            dt = DataType(d_dtype)
            wt = DataType(w_dtype)
            ot = DataType(o_dtype)
            kernel_name = "rvv_dot"
            kernel_name += f"_{n_elems}{dt[0]}{dt.bits}"
            kernel_name += f"_{n_lanes}x{n_elems}{wt[0]}{wt.bits}"
            kernel_name += f"_{n_lanes}{ot[0]}{ot.bits}"
            kernels_inventory[kernel_name] = n_elems

            if not inventory_only:
                logger.debug(f"Registering kernel {kernel_name}")
                desc, impl = rvv_vec_dot_product_kernels(
                    n_elems, n_lanes, d_dtype, w_dtype, o_dtype, lmul
                )
                TensorIntrin.register(kernel_name, desc, impl, override=True)

            n_elems //= 2

    return kernels_inventory


def register_riscv_intrinsics(target: Target):
    """Register RISCV intrinsics

    Args:
        target (Target): TVM target
    """

    # RISCV `v` 1.0 extension templates
    _ = register_rvv_isa_intrinsics(target)
    logger.debug("Finished registering riscv intrinsics.")
