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
from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.target.datatype import register, register_min_func, register_op, create_lower_func

tgt = "llvm"


def convert_ndarray(dst_dtype, array):
    """Converts an NDArray into the specified datatype"""
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        return relay.create_executor('graph').evaluate(cast)(array)


def change_dtype(src, dst, module, params):
    module = relay.frontend.ChangeDatatype(src, dst)(module)
    module = relay.transform.InferType()(module)
    params = dict((p, convert_ndarray(dst, params[p])) for p in params)
    return module, params


def setup():
    # You must first load the library containing the datatype implementation.
    # In this case, we have built the test functions used below right into TVM.
    # CDLL("libmybfloat16.so", RTLD_GLOBAL)

    register("bfloat", 129)

    register_op(create_lower_func("FloatToBFloat16_wrapper"), "Cast", "llvm",
                "bfloat", "float")
    register_op(create_lower_func("BFloat16ToFloat_wrapper"), "Cast", "llvm",
                "float", "bfloat")
    register_op(create_lower_func("BFloat16Add_wrapper"), "Add", "llvm",
                "bfloat")
    register_op(create_lower_func("BFloat16Sub_wrapper"), "Sub", "llvm",
                "bfloat")
    register_op(create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
                "llvm", "bfloat")
    register_op(create_lower_func("BFloat16Mul_wrapper"), "Mul", "llvm",
                "bfloat")
    register_op(create_lower_func("BFloat16Div_wrapper"), "Div", "llvm",
                "bfloat")
    register_op(create_lower_func("BFloat16Max_wrapper"), "Max", "llvm",
                "bfloat")
    register_min_func(lambda num_bits: -3.38953139e38, "bfloat")


def test_change_dtype_simple():
    shape = (3, 1)
    t = relay.TensorType(shape, 'float32')
    a = relay.var("a", t)
    b = relay.var("b", t)
    func = relay.Function([a, b], a + b)
    module = tvm.IRModule.from_expr(func)

    A = tvm.nd.array(np.random.rand(3, 1).astype('float32'))
    B = tvm.nd.array(np.random.rand(3, 1).astype('float32'))

    ex = relay.create_executor("graph", mod=module)
    # Execute the model in the new datatype.
    result = ex.evaluate()(A, B)

    module_changed, _ = change_dtype('float32', 'custom[bfloat]16', module, [])
    ex = relay.create_executor("graph", mod=module_changed)

    A_converted = convert_ndarray('custom[bfloat]16', A)
    B_converted = convert_ndarray('custom[bfloat]16', B)
    result = ex.evaluate()(A_converted, B_converted)
    print(result.dtype)
    result_converted = convert_ndarray('float32', result)
    print(result_converted)


def test_change_dtype_resnet():
    module, params = get_resnet()

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'
    module, params = change_dtype(src_dtype, dst_dtype, module, params)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input)

    # Execute the model in the new datatype.
    ex = relay.create_executor("graph", mod=module)
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        result = ex.evaluate()(input, **params)


def test_change_dtype_inception_v3():
    module, params = get_inception()

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'
    module, params = change_dtype(src_dtype, dst_dtype, module, params)

    ex = relay.create_executor("graph")
    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input)

    # Execute the model in the new datatype.
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        ex = relay.create_executor('graph', mod=module)
        result = ex.evaluate()(input, **params)



if __name__ == "__main__":
    setup()
    # test_change_dtype_inception_v3()
    test_change_dtype_simple()
    test_change_dtype_resnet()
