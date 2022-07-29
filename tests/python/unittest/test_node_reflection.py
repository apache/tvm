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
"""Test node reflection"""

import tvm
import tvm.testing
import pytest
from tvm import te
import numpy as np


def test_const_saveload_json():
    # save load json
    # pylint: disable=invalid-name
    x = tvm.tir.const(1, "int32")
    y = tvm.tir.const(10, "int32")
    z = x + y
    z = z + z
    json_str = tvm.ir.save_json(z)
    zz = tvm.ir.load_json(json_str)
    tvm.ir.assert_structural_equal(zz, z, map_free_vars=True)


def _test_infinity_value(value, dtype):
    x = tvm.tir.const(value, dtype)
    json_str = tvm.ir.save_json(x)
    tvm.ir.assert_structural_equal(x, tvm.ir.load_json(json_str))


def test_infinity_value():
    _test_infinity_value(float("inf"), "float64")
    _test_infinity_value(float("-inf"), "float64")
    _test_infinity_value(float("inf"), "float32")
    _test_infinity_value(float("-inf"), "float32")


def _test_minmax_value(value):
    json_str = tvm.ir.save_json(value)
    tvm.ir.assert_structural_equal(value, tvm.ir.load_json(json_str))


def test_minmax_value():
    _test_minmax_value(tvm.tir.min_value("float32"))
    _test_minmax_value(tvm.tir.max_value("float32"))


def test_make_smap():
    """Test make smap"""
    # save load json
    x = tvm.tir.const(1, "int32")
    y = tvm.tir.const(10, "int32")
    z = tvm.tir.Add(x, y)
    smap = tvm.runtime.convert({"z": z, "x": x})
    json_str = tvm.ir.save_json(tvm.runtime.convert([smap]))
    arr = tvm.ir.load_json(json_str)
    assert len(arr) == 1
    assert arr[0]["z"].a == arr[0]["x"]
    tvm.ir.assert_structural_equal(arr, [smap], map_free_vars=True)


def test_make_node():
    """Test make node"""
    # pylint: disable=invalid-name
    x = tvm.ir.make_node("IntImm", dtype="int32", value=10, span=None)
    assert isinstance(x, tvm.tir.IntImm)
    assert x.value == 10
    a = te.placeholder((10,), name="a")
    aa = tvm.ir.make_node(
        "Tensor", shape=a.shape, dtype=a.dtype, op=a.op, value_index=a.value_index
    )
    assert aa.op == a.op
    assert aa.value_index == a.value_index

    tvm.ir.make_node("IntImm", dtype=tvm.runtime.String("int32"), value=10, span=None)


def test_make_sum():
    # pylint: disable=invalid-name
    a = te.placeholder((2, 10), name="a")
    k = te.reduce_axis((0, 10), "k")
    b = te.compute((2,), lambda i: te.sum(a[i, k], axis=k), name="b")
    json_str = tvm.ir.save_json(b)
    bb = tvm.ir.load_json(json_str)
    assert b.op.body[0].combiner is not None
    assert bb.op.body[0].combiner is not None


def test_env_func():
    """Test environment function"""

    @tvm.register_func("test.env_func")
    def test(x):  # pylint: disable=unused-variable
        return x + 1

    tvm.get_global_func("test.env_func")
    x = tvm.ir.EnvFunc.get("test.env_func")
    assert x.name == "test.env_func"
    json_str = tvm.ir.save_json([x])
    y = tvm.ir.load_json(json_str)[0]
    assert y.name == x.name
    assert y(1) == 2
    assert y.func(1) == 2

    x = tvm.ir.make_node("attrs.TestAttrs", name="xx", padding=(3, 4), func=y)
    assert x.name == "xx"
    assert x.padding[0].value == 3
    assert x.padding[1].value == 4
    assert x.axis == 10
    x = tvm.ir.load_json(tvm.ir.save_json(x))
    assert isinstance(x.func, tvm.ir.EnvFunc)
    assert x.func(10) == 11


def test_string():
    """test string"""
    # pylint: disable=invalid-name
    # non printable str, need to store by b64
    s1 = tvm.runtime.String("xy\x01z")
    s2 = tvm.ir.load_json(tvm.ir.save_json(s1))
    tvm.ir.assert_structural_equal(s1, s2)

    # printable str, need to store by repr_str
    s1 = tvm.runtime.String("xyz")
    s2 = tvm.ir.load_json(tvm.ir.save_json(s1))
    tvm.ir.assert_structural_equal(s1, s2)


def test_pass_config():
    """Test pass config"""
    cfg = tvm.transform.PassContext(
        opt_level=1,
        config={
            "tir.UnrollLoop": {
                "auto_max_step": 10,
            }
        },
    )

    assert cfg.config["tir.UnrollLoop"].auto_max_step == 10
    # default option
    assert cfg.config["tir.UnrollLoop"].explicit_unroll != 0

    # schema checking for specific config key
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"tir.UnrollLoop": {"invalid": 1}})

    # schema check for un-registered config
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"invalid-opt": True})

    # schema check for wrong type
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"tir.UnrollLoop": 1})


def test_dict():
    x = tvm.tir.const(1)  # a class that has Python-defined methods
    # instances should see the full class dict
    assert set(dir(x.__class__)) <= set(dir(x))


def test_ndarray():
    dev = tvm.cpu(0)
    tvm_arr = tvm.nd.array(np.random.rand(4), device=dev)
    tvm_arr2 = tvm.ir.load_json(tvm.ir.save_json(tvm_arr))
    tvm.ir.assert_structural_equal(tvm_arr, tvm_arr2)
    np.testing.assert_array_equal(tvm_arr.numpy(), tvm_arr2.numpy())


def test_ndarray_dict():
    dev = tvm.cpu(0)
    m_1 = {
        "key1": tvm.nd.array(np.random.rand(4), device=dev),
        "key2": tvm.nd.array(np.random.rand(4), device=dev),
    }
    m_2 = tvm.ir.load_json(tvm.ir.save_json(m_1))
    tvm.ir.assert_structural_equal(m_1, m_2)


def test_alloc_const():
    """Test allocate constant"""
    dev = tvm.cpu(0)
    dtype = "float32"
    shape = (16,)
    buf = tvm.tir.decl_buffer(shape, dtype)
    np_data = np.random.rand(*shape).astype(dtype)
    data = tvm.nd.array(np_data, device=dev)
    body = tvm.tir.Evaluate(0)
    alloc_const = tvm.tir.AllocateConst(buf.data, dtype, shape, data, body)
    alloc_const2 = tvm.ir.load_json(tvm.ir.save_json(alloc_const))
    tvm.ir.assert_structural_equal(alloc_const, alloc_const2)
    np.testing.assert_array_equal(np_data, alloc_const2.data.numpy())


if __name__ == "__main__":
    tvm.testing.main()
