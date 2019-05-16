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
#pylint: disable=invalid-name
import tvm


def dot_16x1x16_int8_int8_int32():
    """
    Int8 dot product by every 4 elements using AVX2 Skylake instructions.
    This function takes two arrays of int8 datatype -- data[4] and
    kernel[16][4] -- and computes a dot product of data[4] with every
    4 elements of kernels, resulting in output[16] of int32 datatype.
    The pseudo code is as follows.
    .. code-block:: c
        void dot_16x1x16_int8_int8_int32(int8 data[4], int8 kernel[16][4],
                int32 output[16]){
            for (int i = 0; i < 16; i++){
                out[i] = 0;
                for (int k = 0; k < 4; k++){
                    out[i] += data[k] * kernel[i][k]
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

    int32_lanes = 16 # 16 int32 lanes in AVX512
    num_int8_elements = 4 # 4 int8 elements in int32
    data = tvm.placeholder((num_int8_elements,), dtype='uint8', name='data')
    kernel = tvm.placeholder((int32_lanes, num_int8_elements), dtype='int8', name='kernel')
    k = tvm.reduce_axis((0, num_int8_elements), name='k')
    C = tvm.compute((int32_lanes,),
                    lambda i: tvm.sum(data[k].astype('int32') *
                                      kernel[i, k].astype('int32'),
                                      axis=k),
                    name="C")

    a_buffer = tvm.decl_buffer(data.shape, dtype='uint8', name="a_buffer",
                               offset_factor=1,
                               strides=[1])
    b_buffer = tvm.decl_buffer(kernel.shape, dtype='int8', name="b_buffer",
                               offset_factor=1,
                               strides=[tvm.var('ldw'), 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.const(0, 'int32x16')))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x4")
            re_int32 = tvm.call_pure_intrin('int32', 'reinterpret', a_int8)
            vec_ai32 = re_int32.astype('int32x16')
            vec_a = tvm.call_pure_intrin('int8x64', 'reinterpret', vec_ai32)
            vec_b = ins[1].vload([0, 0], "int8x64")
            vec_one = tvm.const(1, "int16x32")
            pair_reduction = tvm.call_llvm_intrin('int16x32',
                                                  'llvm.x86.avx512.pmaddubs.w.512',
                                                  tvm.const(0, 'uint32'),
                                                  vec_a, vec_b)
            quad_reduction = tvm.call_llvm_intrin('int32x16',
                                                  'llvm.x86.avx512.pmaddw.d.512',
                                                  tvm.const(0, 'uint32'),
                                                  pair_reduction, vec_one)
            if index == 0:
                ib.emit(outs[0].vstore(0, quad_reduction))
            else:
                ib.emit(outs[0].vstore(0, quad_reduction + outs[0].vload([0], 'int32x16')))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={data:a_buffer, kernel:b_buffer})


def dot_16x1x16_int8_int8_int16():
    """
    Int8 dot product by every 2 elements using AVX2 Skylake instructions.
    This function takes two arrays of int8 datatype -- data[2] and
    kernel[4][32][2] -- and computes a dot product of data[2] with every
    2 elements of kernels, resulting in output[4][32] of int16 datatype.
    The pseudo code is as follows.
    .. code-block:: c
        void dot_16x1x16_int8_int8_int16(int8 data[2], int8 kernel[32*4][2],
                int16 output[32*4]){
        for (int i = 0; i< 4; i++){
                for (int j = 0; j < 32; j++){
                    out[i][i] = 0;
                    for (int k = 0; k < 2; k++){
                        out[i][j][k] += data[k] * kernel[i][j][k]
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

    num_int8_elements = 2 # 2 int8 elements in int32
    data = tvm.placeholder((num_int8_elements,), dtype='uint8', name='data')
    kernel = tvm.placeholder((128, num_int8_elements), dtype='int8', name='kernel')
    k = tvm.reduce_axis((0, num_int8_elements), name='k')
    C = tvm.compute((128, ),
                    lambda i: tvm.sum(data[k].astype('int16') *
                                      kernel[i, k].astype('int16'),
                                      axis=k),
                    name="C")

    a_buffer = tvm.decl_buffer(data.shape, dtype='uint8', name="a_buffer",
                               offset_factor=1,
                               strides=[1])
    b_buffer = tvm.decl_buffer(kernel.shape, dtype='int8', name="b_buffer",
                               offset_factor=1)
                               # strides=[tvm.var('ldw'), 1, 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.ir_builder.create()
            if index == 1:
                for i in range(4):
                    ib.emit(outs[0].vstore([i*32], tvm.const(0, 'int16x32')))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x2")
            re_int16 = tvm.call_pure_intrin('int16', 'reinterpret', a_int8)
            vec_ai16 = re_int16.astype('int16x32')
            vec_a = tvm.call_pure_intrin('int8x64', 'reinterpret', vec_ai16)

            for i in range(4):
                vec_b = ins[1].vload([i*32, 0], "int8x64")
                pair_reduction = tvm.call_llvm_intrin('int16x32',
                                                      'llvm.x86.avx512.pmaddubs.w.512',
                                                      tvm.const(0, 'uint32'),
                                                      vec_a, vec_b)
                if index == 0:
                    ib.emit(outs[0].vstore([i*32], pair_reduction))
                else:
                    ib.emit(outs[0].vstore([i*32], pair_reduction + outs[0].vload([i*32],
                                                                                  'int16x32')))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={data:a_buffer, kernel:b_buffer})
