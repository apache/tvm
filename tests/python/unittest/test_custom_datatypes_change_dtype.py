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
        tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
        "llvm", "bfloat")

def test_change_dtype_inception_v3():
    setup()

    expr, params = get_workload()

    def change_dtype(src, dst, expr, params):
        cdtype = relay.frontend.ChangeDatatype(src, dst)
        expr = cdtype.visit(expr)
        expr = relay.ir_pass.infer_type(expr)
        import pdb
        pdb.set_trace()
        params = dict(
            (p, tvm.nd.array(params[p].asnumpy().astype(dst))) for p in params)
        return expr, params

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16' # Change me to posit.
    expr, params = change_dtype(src_dtype, dst_dtype, expr, params)

    ex = relay.create_executor("graph")
    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype(src_dtype))
    x = relay.var("x", shape=(3, 299, 299))
    castR = relay.Function([x], x.astype(dst_dtype))
    input = ex.evaluate(castR)(input)
    # Execute the model in the new datatype.
    result = ex.evaluate(expr)(input, **params)
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    test_change_dtype_inception_v3()
