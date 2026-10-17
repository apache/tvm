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
import copy
import json

import pytest
import tvm_ffi

import tvm_tirx

_LEGACY_RELAX_VAR_JSON = """{
  "root_index": 9,
  "nodes": [
    {"type": "ir.SourceName", "data": "legacy_relax.py"},
    {"type": "ir.SourceLoc", "data": {"source_name": 0, "start_line": 3, "start_column": 3,
      "end_line": 5, "end_column": 11}},
    {"type": "ir.UnknownLoc", "data": null},
    {"type": "ir.PrimType", "data": {"loc": 2, "dtype": "int64"}},
    {"type": "ffi.Array", "data": [3, 3]},
    {"type": "ir.TupleType", "data": {"loc": 2, "fields": 4}},
    {"type": "ffi.String", "data": "legacy"},
    {"type": "relax.expr.Var", "data": {"loc": 1, "ty": 3, "name_hint": 6}},
    {"type": "ffi.Array", "data": [7, 7]},
    {"type": "ir.Tuple", "data": {"loc": 1, "ty": 5, "fields": 8}}
  ],
  "metadata": {"tvm_version": "0.26.dev0"}
}"""

_LEGACY_TIRX_VAR_JSON = """{
  "root_index": 6,
  "nodes": [
    {"type": "ir.SourceName", "data": "legacy_tirx.py"},
    {"type": "ir.SourceLoc", "data": {"source_name": 0, "start_line": 7, "start_column": 2,
      "end_line": 9, "end_column": 14}},
    {"type": "ir.UnknownLoc", "data": null},
    {"type": "ir.PrimType", "data": {"loc": 2, "dtype": "int64"}},
    {"type": "ffi.String", "data": "legacy"},
    {"type": "tirx.Var", "data": {"loc": 1, "ty": 3, "name": 4}},
    {"type": "tirx.Add", "data": {"loc": 1, "ty": 3, "a": 5, "b": 5}}
  ],
  "metadata": {"tvm_version": "0.26.dev0"}
}"""


def _make_pre_name_field_json(type_key):
    graph = json.loads(_LEGACY_TIRX_VAR_JSON)
    graph["root_index"] = 5
    var = graph["nodes"][5]
    var["type"] = type_key
    var["data"]["name_hint"] = var["data"].pop("name")
    return json.dumps(graph)


_PRE_NAME_FIELD_IR_VAR_JSON = _make_pre_name_field_json("ir.Var")
_PRE_NAME_FIELD_DATAFLOW_VAR_JSON = _make_pre_name_field_json("relax.expr.DataflowVar")


@pytest.mark.parametrize(
    ("legacy_json", "expected_type", "var_index"),
    [
        (_LEGACY_RELAX_VAR_JSON, "ir.Var", 7),
        (_LEGACY_TIRX_VAR_JSON, "ir.Var", 5),
        (_PRE_NAME_FIELD_IR_VAR_JSON, "ir.Var", 5),
        (_PRE_NAME_FIELD_DATAFLOW_VAR_JSON, "relax.expr.DataflowVar", 5),
    ],
)
def test_var_name_legacy_json_graph_rewrite(legacy_json, expected_type, var_index):
    from tvm_tirx.ir.json_compact import upgrade_json

    original = json.loads(legacy_json)
    expected = copy.deepcopy(original)
    expected["nodes"][var_index]["type"] = expected_type
    fields = expected["nodes"][var_index]["data"]
    if "name_hint" in fields:
        fields["name"] = fields.pop("name_hint")

    upgraded = json.loads(upgrade_json(legacy_json))
    assert upgraded == expected
    assert upgraded["root_index"] == original["root_index"]
    assert len(upgraded["nodes"]) == len(original["nodes"])


def _check_legacy_var(
    var, source_name, line, end_line, column, end_column, expected_type=tvm_tirx.ir.Var
):
    assert type(var) is expected_type
    assert var.name == "legacy"
    assert var.ty == tvm_tirx.ir.PrimType("int64")
    assert var.loc.source_name.name == source_name
    assert var.loc.start_line == line
    assert var.loc.end_line == end_line
    assert var.loc.start_column == column
    assert var.loc.end_column == end_column


def test_var_exact_base_legacy_tirx_json_load():
    restored = tvm_tirx.ir.load_json(_LEGACY_TIRX_VAR_JSON.replace("tirx.Add", "prim.Add"))
    assert isinstance(restored, tvm_tirx.ir.prim.Add)
    assert restored.a.same_as(restored.b)
    _check_legacy_var(restored.a, "legacy_tirx.py", 7, 9, 2, 14)
    assert restored.loc.same_as(restored.a.loc)
    assert {
        node["type"] for node in json.loads(tvm_tirx.ir.save_json(restored))["nodes"]
    }.isdisjoint({"relax.expr.Var", "tirx.Var", "tirx.Add"})


def test_var_exact_base_legacy_relax_json_load():
    restored = tvm_tirx.ir.load_json(_LEGACY_RELAX_VAR_JSON)
    assert isinstance(restored, tvm_tirx.ir.Tuple)
    assert restored.fields[0].same_as(restored.fields[1])
    _check_legacy_var(restored.fields[0], "legacy_relax.py", 3, 5, 3, 11)
    assert restored.loc.same_as(restored.fields[0].loc)
    assert {
        node["type"] for node in json.loads(tvm_tirx.ir.save_json(restored))["nodes"]
    }.isdisjoint({"relax.expr.Var", "tirx.Var"})


@pytest.mark.parametrize(
    ("legacy_json", "expected_type"),
    [
        (_PRE_NAME_FIELD_IR_VAR_JSON, tvm_tirx.ir.Var),
    ],
)
def test_var_name_legacy_json_load(legacy_json, expected_type):
    restored = tvm_tirx.ir.load_json(legacy_json)
    _check_legacy_var(restored, "legacy_tirx.py", 7, 9, 2, 14, expected_type)
    graph = json.loads(tvm_tirx.ir.save_json(restored))
    fields = graph["nodes"][graph["root_index"]]["data"]
    assert "name" in fields
    assert "name_hint" not in fields


@pytest.mark.parametrize(
    "value",
    [
        tvm_tirx.ir.IntImm("int32", 1) + tvm_tirx.ir.IntImm("int32", 10),
        *[
            tvm_tirx.ir.FloatImm(dtype, value)
            for dtype in ("float32", "float64")
            for value in (float("inf"), float("-inf"))
        ],
        tvm_tirx.ir.prim.min_value(ty="float32"),
        tvm_tirx.ir.prim.max_value(ty="float32"),
        tvm_ffi.core.String("xyz"),
        tvm_ffi.core.String("xy\x01z"),
    ],
)
def test_json_primitive_roundtrip(value):
    tvm_tirx.ir.assert_structural_equal(value, tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(value)))


def test_json_containers():
    values = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json([1, 2, 3.5, True]))
    assert list(values) == [1, 2, 3.5, True]
    assert type(values[3]) is bool
    a = tvm_tirx.ir.Var("a", tvm_tirx.ir.PrimType("int32"))
    b = tvm_tirx.ir.Var("b", tvm_tirx.ir.PrimType("int32"))
    objects = tvm_ffi.convert({a: 2, b: 3})
    assert a in objects and b in dict(objects.items())
    assert len(objects) == 2
    assert tvm_tirx.ir.prim.Add(a, tvm_tirx.ir.IntImm("int32", 1)) not in objects
    assert set(objects) == set(objects.keys()) == {a, b}
    assert set(objects.values()) == {2, 3}
    restored = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(objects))
    assert {key.name: value for key, value in restored.items()} == {"a": 2, "b": 3}
    x = tvm_tirx.ir.IntImm("int32", 1)
    z = tvm_tirx.ir.prim.Add(x, tvm_tirx.ir.IntImm("int32", 10))
    graph = [{"z": z, "x": x}]
    saved = tvm_tirx.ir.save_json(graph)
    restored = tvm_tirx.ir.load_json(saved)
    assert len(restored) == 1
    assert restored[0]["z"].a.same_as(restored[0]["x"])
    tvm_tirx.ir.assert_structural_equal(restored, graph)
    assert json.loads(saved)["metadata"]["tvm_version"] == tvm_tirx.__version__


@pytest.mark.parametrize("dtype", ["int32", tvm_ffi.core.String("int32")])
def test_make_node_default_location(dtype):
    value = tvm_tirx.ir.make_node("ir.IntImm", ty=tvm_tirx.ir.PrimType(dtype), value=10)
    assert isinstance(value, tvm_tirx.ir.IntImm)
    assert value.value == 10
    assert set(dir(value.__class__)) <= set(dir(value))
    assert value.loc.same_as(tvm_tirx.ir.UnknownLoc())
    assert tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(value)).loc.same_as(tvm_tirx.ir.UnknownLoc())
