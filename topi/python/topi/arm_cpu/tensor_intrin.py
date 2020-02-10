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

def dot_int8_int8_int32(int32_lanes, dtype='uint'):
    """
    Int8 dot product by every 4 elements using ARM v8.2 udot.
    This function takes two arrays of int8 datatype -- data[4] and
    kernel[int32_lanes][4] -- and computes a dot product of data[4] with every
    4 elements of kernels, resulting in output[int32_lanes] of uint32 datatype.
    The pseudo code is as follows.

    .. code-block:: c

        void dot_int8_int8_int32(int8 data[4], int8 kernel[16][4], int32 output[16]){
            for (int i = 0; i < int32_lanes; i++){
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

    Parameters
    ----------
    int32_lanes: int
        How many int32/uint32 to produce
    dtype: str, optional, {"uint", "int"}
        Whether it works on unsigned int or signed int

    Returns
    -------
    intrin : TensorIntrin
        The ARM uint8 TensorIntrin that can be used in tensorizing schedule
    """
    num_int8_elements = 4  # 4 int8 elements in int32

    data = tvm.placeholder((num_int8_elements,), dtype='%s8' % dtype, name='data')
    kernel = tvm.placeholder((int32_lanes, num_int8_elements), dtype='%s8' % dtype, name='kernel')

    k = tvm.reduce_axis((0, num_int8_elements), name='k')
    C = tvm.compute((int32_lanes,),
                    lambda i: tvm.sum(data[k].astype('%s32' % dtype) *
                                      kernel[i, k].astype('%s32' % dtype),
                                      axis=k), name="C")

    a_buffer = tvm.decl_buffer(data.shape, dtype='%s8' % dtype, name="a_buffer",
                               offset_factor=1,
                               strides=[1])
    b_buffer = tvm.decl_buffer(kernel.shape, dtype='%s8' % dtype, name="b_buffer",
                               offset_factor=1,
                               strides=[tvm.var('s'), 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.const(0, '%s32x%d' % (dtype, int32_lanes))))
                return ib.get()

            dtype_a = '%s8x%d' % (dtype, num_int8_elements)
            dtype_b = '%s8x%d' % (dtype, int32_lanes * num_int8_elements)
            dtype_c = '%s32x%d' % (dtype, int32_lanes)

            a_int8 = ins[0].vload([0], dtype_a)
            re_int32 = tvm.call_pure_intrin('%s32' % dtype, 'reinterpret', a_int8)
            # broadcast a
            vec_ai32 = re_int32.astype(dtype_c)

            vec_a = tvm.call_pure_intrin(dtype_b, 'reinterpret', vec_ai32)
            vec_b = ins[1].vload([0, 0], dtype_b)
            vec_c = outs[0].vload([0], dtype_c)

            inst = 'udot' if dtype == 'uint' else 'sdot'
            inst = 'llvm.aarch64.neon.%s.v%di32.v%di8' % (
                inst, int32_lanes, int32_lanes * num_int8_elements)
            vdot = tvm.call_llvm_intrin(dtype_c,
                                        inst,
                                        tvm.const(2, 'uint32'),
                                        vec_c, vec_a, vec_b)
            ib.emit(outs[0].vstore(0, vdot))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={data:a_buffer, kernel:b_buffer})
