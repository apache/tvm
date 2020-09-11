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
from tvm import te


def test_const_saveload_json():
    # save load json
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


def test_make_smap():
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
    x = tvm.ir.make_node("IntImm", dtype="int32", value=10)
    assert isinstance(x, tvm.tir.IntImm)
    assert x.value == 10
    A = te.placeholder((10,), name="A")
    AA = tvm.ir.make_node(
        "Tensor", shape=A.shape, dtype=A.dtype, op=A.op, value_index=A.value_index
    )
    assert AA.op == A.op
    assert AA.value_index == A.value_index

    y = tvm.ir.make_node("IntImm", dtype=tvm.runtime.String("int32"), value=10)


def test_make_sum():
    A = te.placeholder((2, 10), name="A")
    k = te.reduce_axis((0, 10), "k")
    B = te.compute((2,), lambda i: te.sum(A[i, k], axis=k), name="B")
    json_str = tvm.ir.save_json(B)
    BB = tvm.ir.load_json(json_str)
    assert B.op.body[0].combiner is not None
    assert BB.op.body[0].combiner is not None


def test_env_func():
    @tvm.register_func("test.env_func")
    def test(x):
        return x + 1

    f = tvm.get_global_func("test.env_func")
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
    # non printable str, need to store by b64
    s1 = tvm.runtime.String("xy\x01z")
    s2 = tvm.ir.load_json(tvm.ir.save_json(s1))
    tvm.ir.assert_structural_equal(s1, s2)

    # printable str, need to store by repr_str
    s1 = tvm.runtime.String("xyz")
    s2 = tvm.ir.load_json(tvm.ir.save_json(s1))
    tvm.ir.assert_structural_equal(s1, s2)


def test_pass_config():
    cfg = tvm.transform.PassContext(
        opt_level=1,
        config={
            "tir.UnrollLoop": {
                "auto_max_step": 10,
            }
        },
    )
    cfg.opt_level == 1

    assert cfg.config["tir.UnrollLoop"].auto_max_step == 10
    # default option
    assert cfg.config["tir.UnrollLoop"].explicit_unroll == True

    # schema checking for specific config key
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"tir.UnrollLoop": {"invalid": 1}})

    # schema check for un-registered config
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"inavlid-opt": True})

    # schema check for wrong type
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"tir.UnrollLoop": 1})


def test_dict():
    x = tvm.tir.const(1)  # a class that has Python-defined methods
    # instances should see the full class dict
    assert set(dir(x.__class__)) <= set(dir(x))


if __name__ == "__main__":
    test_string()
    test_env_func()
    test_make_node()
    test_make_smap()
    test_const_saveload_json()
    test_make_sum()
    test_pass_config()
    test_dict()
    test_infinity_value()
