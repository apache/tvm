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
import tvm
from tvm import tir
from tvm.ir.transform import PassContext
import itertools
import numpy as np


def build_tir_func(func):
    func = func.with_attr("global_symbol", "main")
    pass_ctx = PassContext.current()
    if pass_ctx.config.get("tir.noalias", True):
        func = func.with_attr("tir.noalias", True)
    mod = tvm.IRModule({"main": func})
    func = tvm.build(mod)
    return func


def test_scalar_add():
    # All these types should be interchangeable with each other
    # E.g. float16 + float32 upconverts the float16 --> float32
    # Meanwhile if an int or float or together the int will be
    # cast to the float type.
    lhs_types = ["float32", "float16", "int32", "int64"]
    rhs_types = ["float32", "float16"]
    for lhs_type, rhs_type in itertools.product(lhs_types, rhs_types):
        # Input vars should be float32, we will cast to test for upcasting between them
        lhs_input = tir.Var("lhs", "float32")
        rhs_input = tir.Var("rhs", "float32")
        lhs = tir.Cast(lhs_type, lhs_input)
        rhs = tir.Cast(rhs_type, rhs_input)
        output = lhs + rhs
        output = tir.ret(output)
        output = tir.Evaluate(output)
        func = tir.PrimFunc([lhs_input, rhs_input], output)
        func = build_tir_func(func)
        out = func(1.0, 2.0)
        assert out == 3.0


def test_control_flow_jump():
    ib = tvm.tir.ir_builder.create()
    a = tir.Var("a", "float32")
    b = tir.Var("b", "float32")
    with ib.if_scope(True):
        ib.emit(tir.Evaluate(tir.ret(a)))
    ib.emit(tir.Evaluate(tir.ret(b)))
    stmt = ib.get()
    func = tir.PrimFunc([a, b], stmt)
    func = build_tir_func(func)
    out = func(1.0, 2.0)
    assert out == 1.0


if __name__ == "__main__":
    test_scalar_add()
    test_control_flow_jump()
