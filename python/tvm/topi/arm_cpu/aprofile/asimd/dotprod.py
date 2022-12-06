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


def dot_int8_int8_int32_neon():
    """
    Int8 dot product using vmlal instructions

    .. code-block:: c

        void dot_int8_int8_int32(int8 data[4], int8 kernel[4][4], int32 output[4]){
            for (int i = 0; i < 4; i++){
                out[i] = 0;
                for (int k = 0; k < 4; k++){
                    out[i] += data[k] * kernel[i][k]
                }
            }
        }

    We use the smull and saddlp instructions to compute the dot product.
    smull : int8x16 -> int8x16 -> int16x8 elementwise multiplication
    saddlp: int16x8 -> int32x4 pairwise addition of elements

    Data is broadcast across the register
    int8 elements
    |         data      |         data      |
    |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |

                      smull

    int8 elements
    |     kernel[i]     |     kernel[i+1]   |
    |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |

                        =

    int16 elements
    |               data * kernel[i]        |         data * kernel[i+1]            |
    |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |

                                          saddlp =

    int32 elements
    |    partial sum(data * kernel[i])      |  partial sum(data * kernel[i+1])      |
    |         0         |         1         |         2         |         3         |


    We apply the above kernel twice and use addp to compute the second set of pairwise additions

    int32 elements (narrowed for so they fit on a line)
    |    psum d*k[i]    |   psum d*k[i+1]   |           |   psum d*k[i+2]   |   psum d*k[i+3]   |
    |    0    |    1    |    2    |    3    |   addp    |    4    |    5    |    6    |    7    |
                                                 =
    |sum d*ki |sum d*ki1|sum d*ki2|sum d*ki3|
    |    0    |    1    |    2    |    3    |


    """
    int32_lanes = 4  # 4 int32 lanes = 128
    num_int8_elements = 4  # 4 int8 elements in int32
    data = te.placeholder((num_int8_elements,), dtype="int8", name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype="int8", name="kernel")
    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(data[k].astype("int32") * kernel[i, k].astype("int32"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype="int8", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape, dtype="int8", name="b_buffer", offset_factor=1, strides=[te.var("ldw"), 1]
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            int_8xl = "int8x8"
            int_32xl = "int32x4"
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, int_32xl)))
                return ib.get()

            # this broadcasts data to the vector size
            a_int8 = ins[0].vload([0], "int8x4")
            re_int32 = tvm.tir.call_intrin("int32", "tir.reinterpret", a_int8)
            vec_ai32 = re_int32.astype("int32x2")
            vec_a = tvm.tir.call_intrin(int_8xl, "tir.reinterpret", vec_ai32)

            vec_b = ins[1].vload([0, 0], "int8x16")

            def pairwise_add_mul(extract_half):
                vec_b_half = tvm.tir.call_intrin("int8x8", extract_half, vec_b)
                multiply = tvm.tir.call_llvm_pure_intrin(
                    "int16x8",
                    "llvm.aarch64.neon.smull.v8i16",  # saturating pairwise multiplication
                    tvm.tir.const(2, "uint32"),
                    vec_a,
                    vec_b_half,
                )
                pairwise_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int32x4",
                    "llvm.aarch64.neon.saddlp.v4i32.v8i16",
                    tvm.tir.const(1, "uint32"),
                    multiply,
                )
                return pairwise_reduction

            pair_1 = pairwise_add_mul("tir.vectorlow")
            pair_2 = pairwise_add_mul("tir.vectorhigh")
            quad_reduction = tvm.tir.call_llvm_pure_intrin(
                "int32x4",
                "llvm.aarch64.neon.addp.v4i32",
                tvm.tir.const(2, "uint32"),
                pair_1,
                pair_2,
            )
            if index == 0:
                ib.emit(outs[0].vstore(0, quad_reduction))
            else:
                ib.emit(outs[0].vstore(0, quad_reduction + outs[0].vload([0], int_32xl)))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={data: a_buffer, kernel: b_buffer},
        default_buffer_params=buffer_params,
    )
