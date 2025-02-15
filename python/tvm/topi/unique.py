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
"""Unique operator"""
from tvm import te, tir


def _calc_adjacent_diff_ir(data, output, binop=tir.Sub):
    """Low level IR to calculate adjacent difference in an 1-D array.

    Parameters
    ----------
    data : Buffer
        Input 1-D Buffer.

    output: Buffer
        A buffer to store adjacent difference, of the same shape as data. The adjacent difference
        is defined as: output[0] = 0, output[i] = binop(data[i], data[i-1])
        where i > 0 and i < len(data).

    binop: function, optional
        A binary associative op to use for calculating adjacent difference. The function takes two
        TIR expressions and produce a new TIR expression. By default it uses tvm.tir.Sub to
        compute the adjacent difference.
    """
    ib = tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    output_ptr = ib.buffer_ptr(output)
    with ib.for_range(0, data.shape[0], kind="parallel") as i:
        with ib.if_scope(i == 0):
            output_ptr[0] = 0
        with ib.else_scope():
            output_ptr[i] = tir.Cast(output.dtype, binop(data_ptr[i], data_ptr[i - 1]))
    return ib.get()


def _calc_adjacent_diff(data, out_dtype="int32", binop=tir.Sub):
    """Function calculate adjacent difference in an 1-D array.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input 1-D tensor.

    output_dtype : str
        The output tensor data type.

    binop: function, optional
        A binary associative op to use for calculating difference. The function takes two
        TIR expressions and produce a new TIR expression. By default it uses tvm.tir.Sub to
        compute the adjacent difference.

    Returns
    -------
    output : tvm.te.Tensor
        1-D tensor storing the adjacent difference of the input tensor. The adjacent difference
        is defined as: output[0] = 0, output[i] = binop(data[i], data[i-1])
        where i > 0 and i < len(data).
    """
    return te.extern(
        [data.shape],
        [data],
        lambda ins, outs: _calc_adjacent_diff_ir(ins[0], outs[0], binop=binop),
        dtype=[out_dtype],
        name="_calc_adjacent_diff",
        tag="_calc_adjacent_diff_cpu",
    )
