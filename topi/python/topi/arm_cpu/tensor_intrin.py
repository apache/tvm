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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D int8 schedule on ARM"""

import tvm

def dot_2x1x2_uint8_uint8_uint32():
    """
    Int8 dot product by every 4 elements using ARM v8.2 udot.
    This function takes two arrays of int8 datatype -- data[4] and
    kernel[2][4] -- and computes a dot product of data[4] with every
    4 elements of kernels, resulting in output[2] of uint32 datatype.
    The pseudo code is as follows.
    .. code-block:: c
        void dot_2x1x2_uint8_uint8_uint32(int8 data[4], int8 kernel[16][4], uint32 output[16]){
            for (int i = 0; i < 2; i++){
                out[i] = 0;
                for (int k = 0; k < 4; k++){
                    out[i] += data[k] * kernel[i][k]
                }
            }
        }

    Physically, the kernel array sits in a vector register and
    the data[4] is broadcasted to another vector register. This
    function returns a TensorIntrin that can be used to tensorize
    a schedule.

    Returns
    -------
    intrin : TensorIntrin
        The ARM uint8 TensorIntrin that can be used in tensorizing schedule
    """
    int32_lanes = 2  # 2 uint32 lanes
    num_int8_elements = 4  # 4 uint8 elements in int32

    data = tvm.placeholder((num_int8_elements,), dtype='uint8', name='data')
    kernel = tvm.placeholder((int32_lanes, num_int8_elements), dtype='uint8', name='kernel')

    k = tvm.reduce_axis((0, num_int8_elements), name='k')
    C = tvm.compute((int32_lanes,),
                    lambda i: tvm.sum(data[k].astype('uint32') *
                                      kernel[i, k].astype('uint32'),
                                      axis=k), name="C")

    a_buffer = tvm.decl_buffer(data.shape, dtype='uint8', name="a_buffer",
                               offset_factor=1,
                               strides=[1])
    b_buffer = tvm.decl_buffer(kernel.shape, dtype='uint8', name="b_buffer",
                               offset_factor=1,
                               strides=[tvm.var('s'), 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.const(0, 'uint32x2')))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x4")
            re_int32 = tvm.call_pure_intrin('uint32', 'reinterpret', a_int8)
            vec_ai32 = re_int32.astype('uint32x2')

            vec_a = tvm.call_pure_intrin('uint8x8', 'reinterpret', vec_ai32)
            vec_b = ins[1].vload([0, 0], "uint8x8")
            vec_c = tvm.const(0, 'uint32x2')

            vdot = tvm.call_llvm_intrin('uint32x2',
                                        'llvm.aarch64.neon.udot.v2i32.v8i8',
                                        tvm.const(2, 'uint32'),
                                        vec_c, vec_a, vec_b)
            if index == 0:
                ib.emit(outs[0].vstore(0, vdot))
            else:
                ib.emit(outs[0].vstore(0, vdot + outs[0].vload([0], 'uint32x2')))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={data:a_buffer, kernel:b_buffer})
