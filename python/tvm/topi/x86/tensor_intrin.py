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
"""Core kernel of dot product of 4 Int8 operations"""
# pylint: disable=invalid-name,unused-variable
import tvm
from tvm import te
import tvm.target.codegen
from tvm.target.x86 import target_has_features, get_simd_32bit_lanes


def dot_16x1x16_uint8_int8_int32():
    """Dispatch the most optimized intrin depending on the target"""
    assert target_has_features(
        "sse4.2"
    ), "An old Intel machine that does not have fast Int8 support."
    if target_has_features("avx512vnni") or target_has_features("avxvnni"):
        # VNNI capable platform
        return dot_16x1x16_uint8_int8_int32_cascadelake()
    # vpmaddubsw/vpmaddwd fallback
    return dot_16x1x16_uint8_int8_int32_skylake()


def dot_16x1x16_uint8_int8_int32_skylake():
    """
    Int8 dot product by every 4 elements using AVX512 Skylake instructions.
    This function takes two arrays of uint8 and int8 datatype -- data[4] and
    kernel[16][4] -- and computes a dot product of data[4] with every
    4 elements of kernels, resulting in output[16] of int32 datatype.
    The pseudo code is as follows.
    .. code-block:: c
        void dot_16x1x16_uint8_int8_int32(uint8 data[4], int8 kernel[16][4],
                int32 output[16]){
            for (int i = 0; i < 16; i++){
                output[i] = 0;
                for (int k = 0; k < 4; k++){
                    output[i] += data[k] * kernel[i][k]
                }
            }
        }

    Physically, the kernel array sits in an AVX512 vector register and
    the data[4] is broadcasted to another AVX512 vector register. This
    function returns a TensorIntrin that can be used to tensorize
    a schedule.

    Returns
    -------
    intrin : TensorIntrin
        The Skylake int8 TensorIntrin that can be used in tensorizing schedule
    """

    int32_lanes = get_simd_32bit_lanes()
    num_int8_elements = 4  # 4 int8 elements in int32
    data = te.placeholder((num_int8_elements,), dtype="uint8", name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype="int8", name="kernel")
    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(data[k].astype("int32") * kernel[i, k].astype("int32"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype="uint8", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape, dtype="int8", name="b_buffer", offset_factor=1, strides=[te.var("ldw"), 1]
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            # int_lx32 - output datatype after pmaddubs - 16 bits to number of lanes
            # int_8xl - input datatype to pmaddubs - 8 bits to number of lanes
            # int_32xl - output datatype after pmaddw - 32 bits per number of lanes

            if int32_lanes == 4:
                int_lx32 = "int16x8"
                int_8xl = "int8x16"
                int_32xl = "int32x4"
                pmaddubs = "llvm.x86.ssse3.pmadd.ub.sw.128"
                pmaddw = "llvm.x86.sse2.pmadd.wd"
            elif int32_lanes == 8:
                int_lx32 = "int16x16"
                int_8xl = "int8x32"
                int_32xl = "int32x8"
                pmaddubs = "llvm.x86.avx2.pmadd.ub.sw"
                pmaddw = "llvm.x86.avx2.pmadd.wd"
            elif int32_lanes == 16:
                int_lx32 = "int16x32"
                int_8xl = "int8x64"
                int_32xl = "int32x16"
                pmaddubs = "llvm.x86.avx512.pmaddubs.w.512"
                pmaddw = "llvm.x86.avx512.pmaddw.d.512"

            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, int_32xl)))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x4")
            re_int32 = tvm.tir.call_intrin("int32", "tir.reinterpret", a_int8)
            vec_ai32 = re_int32.astype(int_32xl)
            vec_a = tvm.tir.call_intrin(int_8xl, "tir.reinterpret", vec_ai32)
            vec_b = ins[1].vload([0, 0], int_8xl)
            vec_one = tvm.tir.const(1, int_lx32)
            pair_reduction = tvm.tir.call_llvm_pure_intrin(
                int_lx32,
                pmaddubs,
                tvm.tir.const(2, "uint32"),
                vec_a,
                vec_b,
            )
            quad_reduction = tvm.tir.call_llvm_pure_intrin(
                int_32xl,
                pmaddw,
                tvm.tir.const(2, "uint32"),
                pair_reduction,
                vec_one,
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


def dot_16x1x16_uint8_int8_int16():
    """
    Int8 dot product by every 2 elements using AVX512 Skylake instructions.
    This function takes two arrays of uint8 and int8 datatype -- data[2] and
    kernel[4][32][2] -- and computes a dot product of data[2] with every
    2 elements of kernels, resulting in output[4][32] of int16 datatype.
    The pseudo code is as follows.
    .. code-block:: c
        void dot_16x1x16_uint8_int8_int16(uint8 data[2], int8 kernel[32*4][2],
                int16 output[32*4]){
            for (int i = 0; i< 4; i++){
                for (int j = 0; j < 32; j++){
                    output[i][i] = 0;
                    for (int k = 0; k < 2; k++){
                        output[i][j][k] += data[k] * kernel[i][j][k]
                    }
                }
            }
        }

    Physically, the kernel array sits in four AVX512 vector registers and
    the data[2] is broadcasted to another AVX512 vector register. This
    function returns a TensorIntrin that can be used to tensorize
    a schedule.

    Returns
    -------
    intrin : TensorIntrin
        The Skylake int8 TensorIntrin that can be used in tensorizing schedule
    """

    int16_lanes = 4 * 32  # 4*32 int32 lanes in 4 AVX512 vector registers
    num_int8_elements = 2  # 2 int8 elements in int16
    data = te.placeholder((num_int8_elements,), dtype="uint8", name="data")
    kernel = te.placeholder((int16_lanes, num_int8_elements), dtype="int8", name="kernel")
    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int16_lanes,),
        lambda i: te.sum(data[k].astype("int16") * kernel[i, k].astype("int16"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype="uint8", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(kernel.shape, dtype="int8", name="b_buffer", offset_factor=1)
    # strides=[te.var('ldw'), 1, 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                for i in range(4):
                    ib.emit(outs[0].vstore([i * 32], tvm.tir.const(0, "int16x32")))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x2")
            re_int16 = tvm.tir.call_intrin("int16", "tir.reinterpret", a_int8)
            vec_ai16 = re_int16.astype("int16x32")
            vec_a = tvm.tir.call_intrin("int8x64", "tir.reinterpret", vec_ai16)

            for i in range(4):
                vec_b = ins[1].vload([i * 32, 0], "int8x64")
                pair_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int16x32",
                    "llvm.x86.avx512.pmaddubs.w.512",
                    tvm.tir.const(2, "uint32"),
                    vec_a,
                    vec_b,
                )
                if index == 0:
                    ib.emit(outs[0].vstore([i * 32], pair_reduction))
                else:
                    ib.emit(
                        outs[0].vstore(
                            [i * 32], pair_reduction + outs[0].vload([i * 32], "int16x32")
                        )
                    )
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


def dot_16x1x16_uint8_int8_int32_cascadelake():
    """
    Int8 dot product by every 4 elements using AVX512VNNI Cascade Lake instructions.
    This function takes two arrays of uint8 and int8 datatype -- data[4] and
    kernel[16][4] -- and computes a dot product of data[4] with every
    4 elements of kernels, resulting in output[16] of int32 datatype.
    The pseudo code is as follows.
    .. code-block:: c
        void dot_16x1x16_uint8_int8_int32_cascadelake(uint8 data[4], int8 kernel[16][4],
                int32 output[16]){
            for (int i = 0; i < 16; i++){
                output[i] = 0;
                for (int k = 0; k < 4; k++){
                    output[i] += data[k] * kernel[i][k]
                }
            }
        }

    Physically, the kernel array sits in an AVX512 vector register and
    the data[4] is broadcasted to another AVX512 vector register. This
    function returns a TensorIntrin that can be used to tensorize
    a schedule.

    Returns
    -------
    intrin : TensorIntrin
        The Cascade Lake int8 TensorIntrin that can be used in tensorizing schedule
    """

    int32_lanes = 16  # 16 int32 lanes in AVX512
    num_int8_elements = 4  # 4 int8 elements in int32
    data = te.placeholder((num_int8_elements,), dtype="uint8", name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype="int8", name="kernel")
    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(data[k].astype("int32") * kernel[i, k].astype("int32"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype="uint8", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape, dtype="int8", name="b_buffer", offset_factor=1, strides=[te.var("ldw"), 1]
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, "int32x16")))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x4")
            re_int32 = tvm.tir.call_intrin("int32", "tir.reinterpret", a_int8)
            vec_ai32 = re_int32.astype("int32x16")
            vec_b = ins[1].vload([0, 0], "int8x64")

            vnni_inst_name = "llvm.x86.avx512.vpdpbusd.512"
            llvm_id = tvm.target.codegen.llvm_lookup_intrinsic_id(vnni_inst_name)

            if llvm_id != 0:  # VNNI is available for current LLVM version
                vec_bi32 = tvm.tir.call_intrin("int32x16", "tir.reinterpret", vec_b)
                vec_c = outs[0].vload([0], "int32x16")
                quad_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int32x16",
                    "llvm.x86.avx512.vpdpbusd.512",
                    tvm.tir.const(3, "uint32"),
                    vec_c,
                    vec_ai32,
                    vec_bi32,
                )
                ib.emit(outs[0].vstore(0, quad_reduction))
            else:  # Fall back to the normal AVX512
                vec_a = tvm.tir.call_intrin("int8x64", "tir.reinterpret", vec_ai32)
                vec_one = tvm.tir.const(1, "int16x32")
                pair_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int16x32",
                    "llvm.x86.avx512.pmaddubs.w.512",
                    tvm.tir.const(2, "uint32"),
                    vec_a,
                    vec_b,
                )
                quad_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int32x16",
                    "llvm.x86.avx512.pmaddw.d.512",
                    tvm.tir.const(2, "uint32"),
                    pair_reduction,
                    vec_one,
                )
                if index == 0:
                    ib.emit(outs[0].vstore(0, quad_reduction))
                else:
                    ib.emit(outs[0].vstore(0, quad_reduction + outs[0].vload([0], "int32x16")))
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


def dot_32x128x32_u8s8s32_sapphirerapids(LDA):
    """
    Int8 dot product by every 16x64 elements using AMX-TMUL Sapphire Rapids instructions.
    The tdpxxd instruction takes two tile of uint8 and int8 datatype -- data[16][64] and
    kernel[1][16][16][4] -- and computes a dot product of data[16][16] in int32 datatype.

    (Physically, to efficiently leveraging the tile register, we constructing a 2x2 tiles
    matmul which performs 32x128x32 in total)

    The pseudo code is as follows:
        for(k=0; k<2; k++){
            for(n=0; n<2; n++){
                tileload64(tmm_b, B)
                for(m=0; m<2; m++){
                    if(n==0)
                        tileload64(tmm_a, A)
                    tdpbusd(tmm_c, tmm_a, tmm_b)
                }
            }
        }

    Args:
        LDA (int): the stride of the matrix A, which is uint8 type and use it to determine
                    memory strides of macro reduce axis.

    Returns
    -------
    intrin : TensorIntrin
        The Sapphire Rapids AMX-TMUL int8 tdpbusd TensorIntrin that can be used in tensorizing
        schedule
    """
    A = te.placeholder((32, 128), name="A", dtype="uint8")
    B = te.placeholder((2, 32, 16, 4), name="B", dtype="int8")
    k = te.reduce_axis((0, 128), name="k")

    C = te.compute(
        (32, 32),
        lambda i, j: te.sum(
            A[i, k].astype("int32")
            * B[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(k, 4), j % 16, k % 4].astype("int32"),
            axis=k,
        ),
        name="C",
    )

    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, offset_factor=1, strides=[te.var("ldw"), 1], name="BA"
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        offset_factor=1,
        strides=[te.var("ldw"), te.var("ldw"), te.var("ldw"), 1],
        name="BB",
    )
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, offset_factor=1, strides=[te.var("ldw"), 1], name="BC", scope="amx.tmm"
    )

    def intrin_func(ins, outs):  # pylint: disable=unused-variable
        bufA = ins[0]
        bufB = ins[1]
        bufC = outs[0]

        assert LDA
        _strides_A = tvm.tir.const(LDA, dtype="uint64")
        _strides_B_tile = tvm.tir.const(LDA / 128, dtype="uint64")

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_llvm_intrin(
                    "int32",
                    "llvm.x86.tilezero",
                    tvm.tir.const(1, "uint8"),
                    tvm.tir.const(0, dtype="uint8"),
                )
            )  # tile C 0
            ib.emit(
                tvm.tir.call_llvm_intrin(
                    "int32",
                    "llvm.x86.tilezero",
                    tvm.tir.const(1, "uint8"),
                    tvm.tir.const(1, dtype="uint8"),
                )
            )  # tile C 1
            ib.emit(
                tvm.tir.call_llvm_intrin(
                    "int32",
                    "llvm.x86.tilezero",
                    tvm.tir.const(1, "uint8"),
                    tvm.tir.const(2, dtype="uint8"),
                )
            )  # tile C 2
            ib.emit(
                tvm.tir.call_llvm_intrin(
                    "int32",
                    "llvm.x86.tilezero",
                    tvm.tir.const(1, "uint8"),
                    tvm.tir.const(3, dtype="uint8"),
                )
            )  # tile C 3

            return ib.get()

        def body():  # load A, load B, dpbusd, store C
            ib = tvm.tir.ir_builder.create()

            for k_tile in range(2):  # reduced data blocks
                for n_acc in range(2):  # broadcast data blocks
                    tmm_B_ = tvm.tir.const(n_acc + 6, dtype="uint8")
                    ib.emit(
                        tvm.tir.call_llvm_intrin(
                            "int32",
                            "llvm.x86.tileloaddt164",  # load B: tmm6, tmm7
                            tvm.tir.const(3, "uint8"),
                            tmm_B_,
                            bufB.access_ptr(
                                "r", offset=64 * 16 * (n_acc * 2 * _strides_B_tile + k_tile)
                            ),
                            tvm.tir.const(64, dtype="uint64"),
                        )
                    )

                    for m_acc in range(2):  # loaded data blocks
                        tmm_A_ = tvm.tir.const(m_acc + 4, dtype="uint8")
                        if n_acc == 0:
                            ib.emit(
                                tvm.tir.call_llvm_intrin(
                                    "int32",
                                    "llvm.x86.tileloaddt164",  # load A: , tmm4, tmm5
                                    tvm.tir.const(3, "uint8"),
                                    tmm_A_,
                                    bufA.access_ptr(
                                        "r", offset=m_acc * 16 * _strides_A + k_tile * 64
                                    ),
                                    _strides_A,
                                )
                            )

                        tmm_C_ = tvm.tir.const(m_acc * 2 + n_acc, dtype="uint8")
                        ib.emit(
                            tvm.tir.call_llvm_intrin(
                                "int32",
                                "llvm.x86.tdpbusd",
                                tvm.tir.const(3, "uint8"),
                                tmm_C_,
                                tmm_A_,
                                tmm_B_,
                            )
                        )  # tdpxxd

            return ib.get()

        # body, reset, store
        return (
            body(),
            init(),
            body(),
        )

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def acc_32x32_int32_sapphirerapids(LDC):
    """
    Store the accumulated tile register in scope amx.tmm to global memory.
    (tmm0, tmm1, tmm2, tmm3 --> global 4 tiles)

    Args:
        LDC (int): the stride of the matrix C, which is int32 type and use it to
                    determine memory strides.

    Returns
    -------
    intrin : TensorIntrin
        The Sapphirerapids AMX-TMUL int8 tilestored64 TensorIntrin that can be used
        in tensorizing schedule
    """
    A = te.placeholder((32, 32), name="A", dtype="int32")
    bufA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="amx.tmm",
        name="a_buffer",
        offset_factor=1,
        strides=[te.var("ldw"), 1],
    )

    C = te.compute((32, 32), lambda i, j: A[i, j], name="C")
    bufC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="global",
        name="c_buffer",
        offset_factor=1,
        strides=[te.var("ldw"), 1],
    )

    assert LDC
    _strides_C = tvm.tir.const(4 * LDC, dtype="uint64")

    def intrin_func(ins, outs):  # pylint: disable=unused-variable
        ib = tvm.tir.ir_builder.create()
        bufA = ins[0]
        bufC = outs[0]
        for n_acc in range(2):  # broadcast data blocks
            for m_acc in range(2):  # loaded data blocks
                ib.emit(
                    tvm.tir.call_llvm_intrin(
                        "int32",
                        "llvm.x86.tilestored64",
                        tvm.tir.const(3, "uint8"),
                        tvm.tir.const(m_acc * 2 + n_acc, dtype="uint8"),
                        bufC.access_ptr("w", offset=n_acc * 16 + m_acc * 16 * _strides_C / 4),
                        _strides_C,
                    )
                )

        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: bufA, C: bufC})
