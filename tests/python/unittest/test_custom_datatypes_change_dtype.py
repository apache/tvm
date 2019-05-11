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
from tvm.target.datatype import register, register_op, create_lower_func

tgt = "llvm"

def convert_ndarray(dst_dtype, array, executor):
    """Converts an NDArray into the specified datatype"""
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    with tvm.transform.PassContext(config={
        "tir.disable_vectorize": True
    }):
        return executor.evaluate(cast)(array)

def setup():
    # You must first load the library containing the datatype implementation.
    # In this case, we have built the test functions used below right into TVM.
    # CDLL("libmybfloat16.so", RTLD_GLOBAL)

    register("bfloat", 129)

    register_op(
        create_lower_func("FloatToBFloat16_wrapper"), "Cast",
        "llvm", "bfloat", "float")
    register_op(
        create_lower_func("BFloat16ToFloat_wrapper"), "Cast",
        "llvm", "float", "bfloat")
    register_op(
        create_lower_func("BFloat16Add_wrapper"), "Add", "llvm",
        "bfloat")
    register_op(
        create_lower_func("BFloat16Sub_wrapper"), "Sub", "llvm",
        "bfloat")
    register_op(
        create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
        "llvm", "bfloat")
    register_op(
        create_lower_func("BFloat16Mul_wrapper"), "Mul", "llvm",
        "bfloat")
    register_op(
        create_lower_func("BFloat16Div_wrapper"), "Div", "llvm",
        "bfloat")
    register_op(
        create_lower_func("BFloat16Max_wrapper"), "Max", "llvm",
        "bfloat")

def test_change_dtype_simple():
    a = relay.expr.var("a", dtype="float32", shape=[3,1])
    b = relay.expr.var("b", dtype="float32", shape=[3,1])
    c = a + b

    A = tvm.nd.array(np.random.rand(3,1))
    B = tvm.nd.array(np.random.rand(3,1))


    ex = relay.create_executor("graph")
    # Execute the model in the new datatype.
    result = ex.evaluate(c)([("a", A), ("b", B)])

def test_change_dtype_inception_v3():
    setup()

    module, params = get_workload()

    def change_dtype(src, dst, module, params):
        module = relay.frontend.ChangeDatatype(src, dst)(module)
        module = relay.transform.InferType()(module)
        ex = relay.create_executor()
        params = dict(
            (p, convert_ndarray(dst, params[p], ex)) for p in params)
        return expr, params

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'
    module, params = change_dtype(src_dtype, dst_dtype, module, params)

    ex = relay.create_executor("graph")
    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype(src_dtype))
    x = relay.var("x", shape=(3, 299, 299))
    castR = relay.Function([x], x.astype(dst_dtype))
    input = ex.evaluate(castR)(input)
    # Execute the model in the new datatype.
    result = ex.evaluate(expr)(input, **params)


if __name__ == "__main__":
    # test_change_dtype_inception_v3()
    test_change_dtype_simple()
