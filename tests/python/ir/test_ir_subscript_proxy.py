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

import pytest

import tvm
import tvm.testing


def test_buffer_subscription_accumulates_and_caches():
    buffer = tvm.tirx.decl_buffer((8, 16), "float32")
    prefix = buffer[2]
    full_key = prefix[3]

    assert isinstance(prefix, tvm.ir.SubscriptProxy)
    assert isinstance(full_key, tvm.ir.SubscriptProxy)
    load = full_key.to_expr()
    assert isinstance(load, tvm.tirx.BufferLoad)
    assert [int(index) for index in load.indices] == [2, 3]
    assert full_key.to_expr().same_as(load)

    region = prefix.to_expr()
    assert isinstance(region, tvm.tirx.BufferRegion)
    assert [int(axis.extent) for axis in region.region] == [1, 16]

    # Materializing one immutable prefix does not change later accumulation.
    nested = prefix[3].to_expr()
    assert isinstance(nested, tvm.tirx.BufferLoad)
    assert [int(index) for index in nested.indices] == [2, 3]


def test_buffer_slice_subscription():
    buffer = tvm.tirx.decl_buffer((8,), "float32")

    region = buffer[1:5].to_expr()
    assert isinstance(region, tvm.tirx.BufferRegion)
    assert int(region.region[0].min) == 1
    assert int(region.region[0].extent) == 4

    full_region = buffer[:].to_expr()
    assert isinstance(full_region, tvm.tirx.BufferRegion)
    assert int(full_region.region[0].extent) == 8

    with pytest.raises(ValueError, match="slices with a non-unit step are not supported"):
        buffer[1:7:2].to_expr()


def test_buffer_proxy_operator_surface_realizes_both_operands():
    buffer = tvm.tirx.decl_buffer((8,), "float32")
    lhs = buffer[1]
    rhs = buffer[2]

    assert isinstance(lhs, tvm.ir.ExprOperand)
    result = lhs + rhs
    assert isinstance(result, tvm.tirx.Add)
    assert isinstance(result.a, tvm.tirx.BufferLoad)
    assert isinstance(result.b, tvm.tirx.BufferLoad)
    assert hash(lhs) == hash(lhs)


def test_call_realizes_proxy_arguments_with_late_overload_lookup(monkeypatch):
    from tvm.ir import _tensor_expr_overload

    buffer = tvm.tirx.decl_buffer((8,), "float32")
    callee = tvm.ir.Var("callee", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    captured = []

    def call(receiver, *args, attrs=None):
        captured.append((receiver, args, attrs))
        return receiver

    monkeypatch.setattr(_tensor_expr_overload, "__call__", call)
    assert callee(buffer[1]).same_as(callee)
    assert isinstance(captured[0][1][0], tvm.tirx.BufferLoad)


def test_chaining_after_slice_is_rejected():
    buffer = tvm.tirx.decl_buffer((8, 16), "float32")
    with pytest.raises(TypeError, match="Cannot chain a subscription after a slice"):
        buffer[0:4][1]


def test_unsupported_expression_subscription_fails_eagerly():
    value = tvm.ir.Var("value", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    with pytest.raises(TypeError, match="Type ir.PointerType does not support subscript"):
        value[0]


def test_relax_tuple_is_direct_and_unsupported_tensor_fails_eagerly():
    tuple_value = tvm.relax.Var(
        "tuple_value", tvm.ir.TupleType([tvm.relax.TensorType((2,), "float32")])
    )
    assert isinstance(tuple_value[0], tvm.ir.TupleGetItem)
    (field,) = tuple_value
    assert isinstance(field, tvm.ir.TupleGetItem)

    tensor_value = tvm.relax.Var("tensor_value", tvm.relax.TensorType((2,), "float32"))
    with pytest.raises(TypeError, match="Type relax.TensorType does not support subscript"):
        tensor_value[0]


def test_invalid_subscription_errors():
    buffer = tvm.tirx.decl_buffer((8,), "float32")
    with pytest.raises(ValueError, match="Cannot use and / or / not operator"):
        bool(buffer[0])
    with pytest.raises(TypeError, match="Ellipsis"):
        buffer[...].to_expr()
    with pytest.raises(IndexError, match="Too many indices"):
        buffer[0, 1].to_expr()


if __name__ == "__main__":
    tvm.testing.main()
