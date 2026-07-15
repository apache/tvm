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
# ruff: noqa: E712, F401, F841
import copy
import json
import sys

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import te


def test_const_saveload_json():
    # save load json
    x = tvm.tirx.const(1, "int32")
    y = tvm.tirx.const(10, "int32")
    z = x + y
    z = z + z
    json_str = tvm.ir.save_json(z)
    zz = tvm.ir.load_json(json_str)
    tvm.ir.assert_structural_equal(zz, z, map_free_vars=True)


def test_save_json_metadata_version():
    obj = tvm.runtime.convert([1, 2])
    json_str = tvm.ir.save_json(obj)
    assert json.loads(json_str)["metadata"]["tvm_version"] == tvm.__version__
    assert list(tvm.ir.load_json(json_str)) == [1, 2]


_LEGACY_RELAX_VAR_JSON = """{
  "root_index": 9,
  "nodes": [
    {"type": "ir.SourceName", "data": "legacy_relax.py"},
    {"type": "ir.Span", "data": {"source_name": 0, "line": 3, "column": 3,
      "end_line": 5, "end_column": 11}},
    {"type": "None"},
    {"type": "ir.PrimType", "data": {"span": 2, "dtype": "int64"}},
    {"type": "ffi.Array", "data": [3, 3]},
    {"type": "ir.TupleType", "data": {"span": 2, "fields": 4}},
    {"type": "ffi.String", "data": "legacy"},
    {"type": "relax.expr.Var", "data": {"span": 1, "ty": 3, "name_hint": 6}},
    {"type": "ffi.Array", "data": [7, 7]},
    {"type": "relax.expr.Tuple", "data": {"span": 1, "ty": 5, "fields": 8}}
  ],
  "metadata": {"tvm_version": "0.26.dev0"}
}"""

_LEGACY_TIRX_VAR_JSON = """{
  "root_index": 6,
  "nodes": [
    {"type": "ir.SourceName", "data": "legacy_tirx.py"},
    {"type": "ir.Span", "data": {"source_name": 0, "line": 7, "column": 2,
      "end_line": 9, "end_column": 14}},
    {"type": "None"},
    {"type": "ir.PrimType", "data": {"span": 2, "dtype": "int64"}},
    {"type": "ffi.String", "data": "legacy"},
    {"type": "tirx.Var", "data": {"span": 1, "ty": 3, "name": 4}},
    {"type": "tirx.Add", "data": {"span": 1, "ty": 3, "a": 5, "b": 5}}
  ],
  "metadata": {"tvm_version": "0.26.dev0"}
}"""


@pytest.mark.parametrize(
    ("legacy_json", "legacy_type", "var_index"),
    [
        (_LEGACY_RELAX_VAR_JSON, "relax.expr.Var", 7),
        (_LEGACY_TIRX_VAR_JSON, "tirx.Var", 5),
    ],
)
def test_var_exact_base_legacy_json_graph_rewrite(legacy_json, legacy_type, var_index):
    from tvm.ir.json_compact import upgrade_json

    original = json.loads(legacy_json)
    expected = copy.deepcopy(original)
    expected["nodes"][var_index]["type"] = "ir.Var"
    if legacy_type == "tirx.Var":
        expected["nodes"][var_index]["data"]["name_hint"] = expected["nodes"][var_index][
            "data"
        ].pop("name")

    upgraded = json.loads(upgrade_json(legacy_json))
    assert upgraded == expected
    assert upgraded["root_index"] == original["root_index"]
    assert len(upgraded["nodes"]) == len(original["nodes"])


def _check_legacy_var(var, source_name, line, end_line, column, end_column):
    assert type(var) is tvm.ir.Var
    assert var.name == "legacy"
    assert var.ty == tvm.ir.PrimType("int64")
    assert var.span.source_name.name == source_name
    assert var.span.line == line
    assert var.span.end_line == end_line
    assert var.span.column == column
    assert var.span.end_column == end_column


def test_var_exact_base_legacy_relax_json_load():
    restored = tvm.ir.load_json(_LEGACY_RELAX_VAR_JSON)
    assert isinstance(restored, tvm.relax.Tuple)
    assert restored.fields[0].same_as(restored.fields[1])
    _check_legacy_var(restored.fields[0], "legacy_relax.py", 3, 5, 3, 11)
    assert restored.span.same_as(restored.fields[0].span)
    assert {node["type"] for node in json.loads(tvm.ir.save_json(restored))["nodes"]}.isdisjoint(
        {"relax.expr.Var", "tirx.Var"}
    )


def test_var_exact_base_legacy_tirx_json_load():
    restored = tvm.ir.load_json(_LEGACY_TIRX_VAR_JSON)
    assert isinstance(restored, tvm.tirx.Add)
    assert restored.a.same_as(restored.b)
    _check_legacy_var(restored.a, "legacy_tirx.py", 7, 9, 2, 14)
    assert restored.span.same_as(restored.a.span)
    assert {node["type"] for node in json.loads(tvm.ir.save_json(restored))["nodes"]}.isdisjoint(
        {"relax.expr.Var", "tirx.Var"}
    )


def test_dataflow_var_json_is_not_migrated_to_canonical_var():
    dataflow_var = tvm.relax.DataflowVar("value", tvm.ir.PrimType("int64"))
    graph = json.loads(tvm.ir.save_json(dataflow_var))
    root = graph["nodes"][graph["root_index"]]
    assert root["type"] == "relax.expr.DataflowVar"
    restored = tvm.ir.load_json(json.dumps(graph))
    assert type(restored) is tvm.relax.DataflowVar


def _test_infinity_value(value, dtype):
    x = tvm.tirx.const(value, dtype)
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
    _test_minmax_value(tvm.tirx.min_value("float32"))
    _test_minmax_value(tvm.tirx.max_value("float32"))


def test_make_smap():
    # save load json
    x = tvm.tirx.const(1, "int32")
    y = tvm.tirx.const(10, "int32")
    z = tvm.tirx.Add(x, y)
    smap = tvm.runtime.convert({"z": z, "x": x})
    json_str = tvm.ir.save_json(tvm.runtime.convert([smap]))
    arr = tvm.ir.load_json(json_str)
    assert len(arr) == 1
    assert arr[0]["z"].a == arr[0]["x"]
    tvm.ir.assert_structural_equal(arr, [smap], map_free_vars=True)


def test_make_node():
    x = tvm.ir.make_node("ir.IntImm", ty=tvm.ir.PrimType("int32"), value=10, span=None)
    assert isinstance(x, tvm.tirx.IntImm)
    assert x.value == 10
    A = te.placeholder((10,), name="A")
    AA = tvm.ir.make_node(
        "te.Tensor", shape=A.shape, dtype=A.dtype, op=A.op, value_index=A.value_index
    )
    assert AA.op == A.op
    assert AA.value_index == A.value_index

    y = tvm.ir.make_node(
        "ir.IntImm", ty=tvm.ir.PrimType(tvm_ffi.core.String("int32")), value=10, span=None
    )
    assert isinstance(y, tvm.tirx.IntImm)
    assert y.value == 10


def test_make_sum():
    A = te.placeholder((2, 10), name="A")
    k = te.reduce_axis((0, 10), "k")
    B = te.compute((2,), lambda i: te.sum(A[i, k], axis=k), name="B")
    json_str = tvm.ir.save_json(B)
    BB = tvm.ir.load_json(json_str)
    assert B.op.body[0].combiner is not None
    assert BB.op.body[0].combiner is not None


def test_string():
    # non printable str, need to store by b64
    s1 = tvm_ffi.core.String("xy\x01z")
    s2 = tvm.ir.load_json(tvm.ir.save_json(s1))
    tvm.ir.assert_structural_equal(s1, s2)

    # printable str, need to store by repr_str
    s1 = tvm_ffi.core.String("xyz")
    s2 = tvm.ir.load_json(tvm.ir.save_json(s1))
    tvm.ir.assert_structural_equal(s1, s2)


def test_pass_config():
    cfg = tvm.transform.PassContext(
        opt_level=1,
        config={
            "tirx.UnrollLoop": {
                "auto_max_step": 10,
            }
        },
    )
    cfg.opt_level == 1

    assert cfg.config["tirx.UnrollLoop"].auto_max_step == 10
    # default option
    assert cfg.config["tirx.UnrollLoop"].explicit_unroll == True

    # schema checking for specific config key
    with pytest.raises(TypeError):
        cfg = tvm.transform.PassContext(config={"tirx.UnrollLoop": {"invalid": 1}})

    # schema check for un-registered config
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"inavlid-opt": True})

    # schema check for wrong type
    with pytest.raises(AttributeError):
        cfg = tvm.transform.PassContext(config={"tirx.UnrollLoop": 1})


def test_dict():
    x = tvm.tirx.const(1)  # a class that has Python-defined methods
    # instances should see the full class dict
    assert set(dir(x.__class__)) <= set(dir(x))


def test_tensor():
    dev = tvm.cpu(0)
    tvm_arr = tvm.runtime.tensor(np.random.rand(4), device=dev)
    tvm_arr2 = tvm.ir.load_json(tvm.ir.save_json(tvm_arr))
    tvm.ir.assert_structural_equal(tvm_arr, tvm_arr2)
    np.testing.assert_array_equal(tvm_arr.numpy(), tvm_arr2.numpy())


def test_tensor_dict():
    dev = tvm.cpu(0)
    m1 = {
        "key1": tvm.runtime.tensor(np.random.rand(4), device=dev),
        "key2": tvm.runtime.tensor(np.random.rand(4), device=dev),
    }
    m2 = tvm.ir.load_json(tvm.ir.save_json(m1))
    tvm.ir.assert_structural_equal(m1, m2)


def test_free_var_equal():
    x = tvm.tirx.Var("x", ty="int32")
    y = tvm.tirx.Var("y", ty="int32")
    z = tvm.tirx.Var("z", ty="int32")
    v1 = x + y
    v1 = y + z
    tvm.ir.assert_structural_equal(x, z, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
