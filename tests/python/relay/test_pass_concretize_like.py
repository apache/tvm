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
"""Tests for the ConcretizeLike pass."""
import pytest
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay.testing import run_infer_type


def test_reshape_like():
    data = relay.var("data", shape=(2, 3, 4), dtype="float32")
    shape_like = relay.var("shape_like", shape=(6, 2, 2), dtype="float32")
    f = relay.Function([data, shape_like], relay.reshape_like(data, shape_like))
    f_expected = relay.Function([data, shape_like], relay.reshape(data, (6, 2, 2)))
    f_expected = run_infer_type(f_expected)

    mod = tvm.IRModule.from_expr(f)
    mod_concrete = relay.transform.ConcretizeLike()(mod)
    assert tvm.ir.structural_equal(mod_concrete["main"], f_expected)


def test_reshape_like_attrs():
    data = relay.var("data", shape=(2, 3, 4), dtype="float32")
    shape_like = relay.var("shape_like", shape=(6, 2, 2), dtype="float32")
    f = relay.Function(
        [data, shape_like], relay.reshape_like(data, shape_like, lhs_begin=2, rhs_begin=1)
    )
    f_expected = relay.Function([data, shape_like], relay.reshape(data, (2, 3, 2, 2)))
    f_expected = run_infer_type(f_expected)

    mod = tvm.IRModule.from_expr(f)
    mod_concrete = relay.transform.ConcretizeLike()(mod)
    assert tvm.ir.structural_equal(mod_concrete["main"], f_expected)


def test_zeros_like():
    dtype = "int32"
    shape_like = relay.var("shape_like", shape=(3, 4, 5), dtype=dtype)
    f = relay.Function([shape_like], relay.zeros_like(shape_like))
    f_expected = relay.Function([shape_like], relay.zeros((3, 4, 5), dtype))
    f_expected = run_infer_type(f_expected)

    mod = tvm.IRModule.from_expr(f)
    mod_concrete = relay.transform.ConcretizeLike()(mod)
    assert tvm.ir.structural_equal(mod_concrete["main"], f_expected)


def test_ones_like():
    dtype = "int32"
    shape_like = relay.var("shape_like", shape=(3, 4, 5), dtype=dtype)
    f = relay.Function([shape_like], relay.ones_like(shape_like))
    f_expected = relay.Function([shape_like], relay.ones((3, 4, 5), dtype))
    f_expected = run_infer_type(f_expected)

    mod = tvm.IRModule.from_expr(f)
    mod_concrete = relay.transform.ConcretizeLike()(mod)
    assert tvm.ir.structural_equal(mod_concrete["main"], f_expected)


def test_collapse_sum_like():
    data = relay.var("data", shape=(3, 3, 3), dtype="float32")
    shape_like = relay.var("shape_like", shape=(3,), dtype="float32")
    f = relay.Function([data, shape_like], relay.collapse_sum_like(data, shape_like))
    f_expected = relay.Function([data, shape_like], relay.collapse_sum_to(data, (3,)))
    f_expected = run_infer_type(f_expected)

    mod = tvm.IRModule.from_expr(f)
    mod_concrete = relay.transform.ConcretizeLike()(mod)
    assert tvm.ir.structural_equal(mod_concrete["main"], f_expected)


def test_broadcast_to_like():
    data = relay.var("data", shape=(3,), dtype="float32")
    shape_like = relay.var("shape_like", shape=(3, 3, 3), dtype="float32")
    f = relay.Function([data, shape_like], relay.broadcast_to_like(data, shape_like))
    f_expected = relay.Function([data, shape_like], relay.broadcast_to(data, (3, 3, 3)))
    f_expected = run_infer_type(f_expected)

    mod = tvm.IRModule.from_expr(f)
    mod_concrete = relay.transform.ConcretizeLike()(mod)
    assert tvm.ir.structural_equal(mod_concrete["main"], f_expected)


def test_multiple():
    x = relay.var("x", shape=(2, 3), dtype="float32")
    y = relay.var("x", shape=(3,), dtype="float32")
    l = x + y

    dl = relay.ones_like(l)
    dx = relay.zeros_like(x)
    dy = relay.zeros_like(y)
    dx = dx + relay.collapse_sum_like(dl, dx)
    dy = dy + relay.collapse_sum_like(dl, dy)
    ret = relay.Tuple([dx, dy])
    f = relay.Function([x, y], ret)

    dl_c = relay.ones((2, 3), "float32")
    dx_c = relay.zeros((2, 3), "float32")
    dy_c = relay.zeros((3,), "float32")
    dx_c = dx_c + relay.collapse_sum_to(dl_c, (2, 3))
    dy_c = dy_c + relay.collapse_sum_to(dl_c, (3,))
    ret_c = relay.Tuple([dx_c, dy_c])
    f_expected = relay.Function([x, y], ret_c)
    f_expected = run_infer_type(f_expected)

    mod = tvm.IRModule.from_expr(f)
    mod_concrete = relay.transform.ConcretizeLike()(mod)
    assert tvm.ir.structural_equal(mod_concrete["main"], f_expected)


if __name__ == "__main__":
    pytest.main([__file__])
