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

"""CMSIS-NN integration tests: softmax"""

import pytest
import sys

import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn
import numpy as np


def count_num_calls(mod):
    class CallCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    counter = CallCounter()
    for var in mod.get_global_vars():
        counter.visit(mod[var.name_hint])
    return counter.count


def make_module(func):
    func = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(func)
    return relay.transform.InferType()(mod)


def make_model(shape, zero_point, scale, in_dtype, out_dtype):
    a = relay.var("a", shape=shape, dtype=in_dtype)
    dequantize = relay.qnn.op.dequantize(
        a,
        input_scale=relay.const(scale, "float32"),
        input_zero_point=relay.const(zero_point, "int32"),
    )
    softmax = relay.nn.softmax(dequantize)
    model = relay.qnn.op.quantize(
        softmax,
        output_scale=relay.const(scale, "float32"),
        output_zero_point=relay.const(zero_point, "int32"),
        out_dtype=out_dtype,
    )
    return model


def test_softmax_int8():
    model = make_model([1, 16, 16, 3], 64, 0.02, "int8", "int8")
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    attrs = [
        cmsisnn_mod[var.name_hint].attrs
        for var in cmsisnn_mod.get_global_vars()
        if cmsisnn_mod[var.name_hint].attrs
    ]
    assert any(attrs), "At least one function with external attributes was expected."

    compilers = [
        key == "Compiler" and value == "cmsisnn" for attr in attrs for key, value in attr.items()
    ]
    assert any(compilers), "Module does not contain function for cmsisnn target."

    assert count_num_calls(orig_mod) == count_num_calls(
        cmsisnn_mod
    ), "Number of calls changed during partitioning"


@pytest.mark.parametrize("in_dtype,out_dtype", [["uint8", "int8"], ["int8", "uint8"]])
def test_softmax_not_int8(in_dtype, out_dtype):
    model = make_model([1, 16, 16, 3], 64, 0.02, in_dtype, out_dtype)
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    attrs = [
        cmsisnn_mod[var.name_hint].attrs
        for var in cmsisnn_mod.get_global_vars()
        if cmsisnn_mod[var.name_hint].attrs
    ]
    assert not any(attrs), "No function should have an external attribute."


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
