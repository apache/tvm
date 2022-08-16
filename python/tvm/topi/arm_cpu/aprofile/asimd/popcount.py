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

import tvm
from tvm import te


def popcount(m, k_i, w_b, x_b, unipolar):
    pack_dtype = "uint8"
    w = te.placeholder((w_b, m, k_i), dtype=pack_dtype, name="w")
    x = te.placeholder(
        (
            x_b,
            k_i,
        ),
        dtype=pack_dtype,
        name="x",
    )
    k = te.reduce_axis((0, k_i), name="k")
    bw = te.reduce_axis((0, w_b), name="bw")
    bx = te.reduce_axis((0, x_b), name="bx")
    if unipolar:
        dtype = "int16"
        z = te.compute(
            (m,),
            lambda i: te.sum(
                (
                    tvm.tir.popcount(w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype))
                    - tvm.tir.popcount(~w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype))
                )
                << (bw + bx).astype(dtype),
                axis=[bw, bx, k],
            ),
            name="z",
        )
    else:
        dtype = "uint16"
        z = te.compute(
            (m,),
            lambda i: te.sum(
                tvm.tir.popcount(w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype))
                << (bw + bx).astype(dtype),
                axis=[bw, bx, k],
            ),
            name="z",
        )
    Wb = tvm.tir.decl_buffer(
        w.shape, w.dtype, name="W", offset_factor=k_i, strides=[te.var("ldw"), te.var("ldw"), 1]
    )  # stride can be inferred
    Xb = tvm.tir.decl_buffer(
        x.shape, x.dtype, name="X", offset_factor=k_i, strides=[te.var("ldw"), 1]
    )
    Zb = tvm.tir.decl_buffer(z.shape, z.dtype, name="Z", offset_factor=1, strides=[1])

    def _intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]

        args_2 = tvm.tir.const(2, "uint32")

        if unipolar:
            vpadd = "llvm.arm.neon.vpadd.v8i8"
            vpadalu = "llvm.arm.neon.vpadals.v16i8.v8i16"
            full_dtype = "int8x16"
            half_dtype = "int8x8"
            return_dtype = "int16x8"
        else:
            vpadd = "llvm.arm.neon.vpadd.v8u8"
            vpadalu = "llvm.arm.neon.vpadalu.v16u8.v8u16"
            full_dtype = "uint8x16"
            half_dtype = "uint8x8"
            return_dtype = "uint16x8"

        def _instr(index):
            irb = tvm.tir.ir_builder.create()
            if index == 1:  # reduce reset
                irb.emit(zz.vstore(0, tvm.tir.const(0, return_dtype)))
                return irb.get()
            # body and reduce update
            cnts8 = [None] * 8
            cnts4 = [None] * 4
            cnts2 = [None] * 2
            for bw in range(w_b):
                for bx in range(x_b):
                    if k_i == 16:
                        for i in range(m):
                            w_ = ww.vload([bw, i, 0], "uint8x16").astype(full_dtype)
                            x_ = xx.vload([bx, 0], "uint8x16").astype(full_dtype)
                            if unipolar:
                                cnts = tvm.tir.popcount(w_ & x_) - tvm.tir.popcount(~w_ & x_)
                            else:
                                cnts = tvm.tir.popcount(w_ & x_)
                            upper_half = tvm.tir.call_intrin(half_dtype, "tir.vectorhigh", cnts)
                            lower_half = tvm.tir.call_intrin(half_dtype, "tir.vectorlow", cnts)
                            cnts8[i] = upper_half + lower_half
                        for i in range(m // 2):
                            cnts4[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts8[i * 2], cnts8[i * 2 + 1]
                            )
                        for i in range(m // 4):
                            cnts2[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts4[i * 2], cnts4[i * 2 + 1]
                            )
                        cnts = tvm.tir.call_intrin(
                            full_dtype, "tir.vectorcombine", cnts2[0], cnts2[1]
                        )
                        shifted_cnts = cnts << tvm.tir.const(bw + bx, pack_dtype)
                        out = tvm.tir.call_llvm_pure_intrin(
                            return_dtype, vpadalu, args_2, zz.vload(0, return_dtype), shifted_cnts
                        )
                    else:  # ki == 8
                        for i in range(m):
                            w_ = ww.vload([bw, i, 0], "uint8x8").astype(half_dtype)
                            x_ = xx.vload([bx, 0], "uint8x8").astype(half_dtype)
                            if unipolar:
                                cnts8[i] = tvm.tir.popcount(w_ & x_) - tvm.tir.popcount(~w_ & x_)
                            else:
                                cnts8[i] = tvm.tir.popcount(w_ & x_)
                        for i in range(m // 2):
                            cnts4[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts8[i * 2], cnts8[i * 2 + 1]
                            )
                        for i in range(m // 4):
                            cnts2[i] = tvm.tir.call_llvm_pure_intrin(
                                half_dtype, vpadd, args_2, cnts4[i * 2], cnts4[i * 2 + 1]
                            )
                        cnts = tvm.tir.call_intrin(
                            full_dtype, "tir.vectorcombine", cnts2[0], cnts2[1]
                        )
                        shifted_cnts = cnts << tvm.tir.const(bw + bx, pack_dtype)
                        out = tvm.tir.call_llvm_pure_intrin(
                            return_dtype, vpadalu, args_2, zz.vload(0, return_dtype), shifted_cnts
                        )
                    irb.emit(zz.vstore(0, out))
            return irb.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        z.op, _intrin_func, binds={w: Wb, x: Xb, z: Zb}, default_buffer_params=buffer_params
    )
