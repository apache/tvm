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
"""Utilities for changing datatypes of models."""

import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing.inception_v3 import get_workload

tgt = "llvm"

def setup():
    # You must first load the library containing the datatype implementation.
    # In this case, we have built the test functions used below right into TVM.
    # CDLL("libmybfloat16.so", RTLD_GLOBAL)

    tvm.datatype.register("bfloat", 129)

    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "Cast",
        "llvm", "bfloat", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16ToFloat_wrapper"), "Cast",
        "llvm", "float", "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Add_wrapper"), "Add", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Sub_wrapper"), "Sub", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
        "llvm", "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Mul_wrapper"), "Mul", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Div_wrapper"), "Div", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Max_wrapper"), "Max", "llvm",
        "bfloat")

def test_change_dtype_inception_v3():
    setup()

    expr, params = get_workload()

    ex = relay.create_executor("graph")

    def convert_ndarray(dst_dtype, array):
        x = relay.var(str(array) + dst_dtype, shape=array.shape)
        cast = relay.Function([x], x.astype(dst_dtype))
        return ex.evaluate(cast)(array)

    def change_dtype(src, dst, expr, params):
        cdtype = relay.frontend.ChangeDatatype(src, dst)
        expr = cdtype.visit(expr)
        expr = relay.ir_pass.infer_type(expr)
        #raise "pause"
        params = dict(
            (p, convert_ndarray(dst, params[p])) for p in params)
        return expr, params

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16' # Change me to posit.
    expr, params = change_dtype(src_dtype, dst_dtype, expr, params)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input)

    def print_info(node):
        if not isinstance(node, relay.op.op.Op):
            if ("custom[bfloat]32" not in str(node.checked_type())):
                print(node.checked_type())
    relay.ir_pass.post_order_visit(expr, print_info)

    # Execute the model in the new datatype.
    result = ex.evaluate(expr)(input, **params)


if __name__ == "__main__":
    test_change_dtype_inception_v3()
