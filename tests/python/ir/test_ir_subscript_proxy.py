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


def test_buffer_slice_and_vector_subscription():
    buffer = tvm.tirx.decl_buffer((8,), "float32")

    region = buffer[1:5].to_expr()
    assert isinstance(region, tvm.tirx.BufferRegion)
    assert int(region.region[0].min) == 1
    assert int(region.region[0].extent) == 4

    vector = buffer[1:7:2].to_expr()
    assert isinstance(vector, tvm.tirx.BufferLoad)
    assert isinstance(vector.indices[0], tvm.tirx.Ramp)
    assert int(vector.indices[0].lanes) == 3


def test_expression_getitem_returns_proxy_directly():
    value = tvm.ir.Var("value", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    proxy = value[0]
    assert isinstance(proxy, tvm.ir.SubscriptProxy)
    with pytest.raises(TypeError, match="does not register __subscript_expr_realize__"):
        proxy.to_expr()


def test_relax_tuple_is_direct_and_tensor_is_lazy():
    tuple_value = tvm.relax.Var(
        "tuple_value", tvm.ir.TupleType([tvm.relax.TensorType((2,), "float32")])
    )
    assert isinstance(tuple_value[0], tvm.ir.TupleGetItem)

    tensor_value = tvm.relax.Var("tensor_value", tvm.relax.TensorType((2,), "float32"))
    assert isinstance(tensor_value[0], tvm.ir.SubscriptProxy)


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
