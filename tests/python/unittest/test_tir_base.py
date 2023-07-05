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
import pytest
from tvm import tir
from tvm._ffi.base import TVMError
from tvm.ir.transform import PassContext
import itertools
import pytest


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


def assignment_helper(store_dtype, value_dtype):
    store = tir.Var("store", dtype=store_dtype)
    value = tir.Var("value", dtype=value_dtype)
    tir.Let(store, value, body=store)


def test_fail_implicit_downcasts_same_type():
    # These lists should be sorted
    bits = [8, 16, 32, 64]
    for type in ["float", "int", "uint"]:
        for i in range(len(bits) - 1):
            with pytest.raises(TVMError):
                assignment_helper(
                    store_dtype=f"{type}{bits[i]}", value_dtype=f"{type}{bits[i + 1]}"
                )


def test_cast_between_types():
    # We should only be able to assign values with the same types
    bits = [16, 32]
    types = ["float", "int", "uint"]
    for store_type, store_bits, value_type, value_bits in itertools.product(
        types, bits, types, bits
    ):
        store_dtype = f"{store_type}{store_bits}"
        value_dtype = f"{value_type}{value_bits}"
        if store_dtype == value_dtype:
            assignment_helper(store_dtype, value_dtype)
        else:
            # TODO: we might want to allow casts between uint and int types
            with pytest.raises(TVMError):
                assignment_helper(store_dtype, value_dtype)


def test_ret_const():
    a = tir.const(0)
    b = tir.ret(a)
    b = tir.Evaluate(b)
    func = tir.PrimFunc([], b)
    func = build_tir_func(func)
    out = func()
    assert out == 0


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


def test_exception():
    with pytest.raises(tvm.TVMError):
        x = tir.Var(name=1, dtype="int")


def test_eq_ops():
    a = tir.IntImm("int8", 1)
    with pytest.raises(ValueError):
        assert a != None
    with pytest.raises(ValueError):
        assert not a == None
    b = tir.StringImm("abc")
    assert b != None
    assert not b == None


if __name__ == "__main__":
    test_scalar_add()
    test_ret_const()
    test_control_flow_jump()
    test_exception()
    test_eq_ops()
