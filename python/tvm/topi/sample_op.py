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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""SampleOp operator"""
from ..tir import decl_buffer, ir_builder, Cast, AssertStmt, StringImm, Evaluate
from ..te import extern, hybrid


# @hybrid.script
# def _sample_op(sample_input):
#     # data_buf = decl_buffer((sample_input[0],), "int64", "data_buf", data_alignment=8)

#     # out_buf = decl_buffer(sample_input.shape[0], sample_input.dtype, "out_buf")

#     out = output_tensor((3, ), "int64")
#     for i in range(sample_input[0]):
#         out[i] = int64(sample_input[i])
#     return out

# def sample_op(sample_input):
#     return _sample_op(sample_input)


def sample_op(sample_input, out, out2):

    def gen_ir(sample_input_ptr, out_ptr, out2_ptr):
        # pylint: disable=invalid-name
        ib = ir_builder.create()

        sample_input = ib.buffer_ptr(sample_input_ptr)
        out = ib.buffer_ptr(out_ptr)
        out2 = ib.buffer_ptr(out2_ptr)

        with ib.for_range(0, sample_input[0]) as i:
            # out[i] = sample_input[i]
            out[i] = Cast("int64", 1)

        with ib.for_range(0, sample_input[1]) as i:
            out2[i] = Cast("int64", 1)

        return ib.get()

    out_buf = decl_buffer(out, "int64", "out_buf")
    out2_buf = decl_buffer(out2, "int64", "out2_buf")
    return extern(
        [out, out2],
        [sample_input],
        lambda ins, outs: gen_ir(ins[0], outs[0], outs[1]),
        dtype="int64",
        out_buffers=[out_buf, out2_buf],
        name="sample_op",
        tag="sample_op",
    )
