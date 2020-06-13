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
from tvm import relay
from tvm import te
import json

def test_type_var():
    # type var in 0.6
    nodes = [
        {"type_key": ""},
        {"type_key": "relay.TypeVar",
         "attrs": {"kind": "0", "span": "0", "var": "2"}},
        {"type_key": "Variable",
         "attrs": {"dtype": "int32", "name": "in0"}},
        ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    tvar = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tvar, tvm.ir.TypeVar)
    assert tvar.name_hint == "in0"
    nodes[1]["type_key"] = "relay.GlobalTypeVar"
    tvar = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tvar, tvm.ir.GlobalTypeVar)
    assert tvar.name_hint == "in0"

def test_var():
    # type var in 0.6
    nodes = [
        {"type_key": ""},
        {"type_key": "relay.Var",
         "attrs": {
             "_checked_type_": "0",
             "span": "0",
             "type_annotation": "0",
             "vid": "2"
         }
        },
        {"type_key": "relay.Id",
         "attrs": {"name_hint": "a3"}},
        {"type_key": "relay.TensorType",
         "attrs": {
             "dtype": "float32",
             "shape": "4",
             "span": "0"
         }
        },
        {"type_key": "Array",
         "data": [5, 6]
        },
        {"type_key": "IntImm",
         "attrs": {"dtype": "int32", "value": "16"}},
        {"type_key": "IntImm",
         "attrs": {"dtype": "int32", "value": "8"}}
        ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    tvar = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tvar, relay.Var)
    assert tvar.name_hint == "a3"

def test_incomplete_type():
    nodes = [
        {"type_key": ""},
        {"type_key": "relay.IncompleteType",
         "attrs": {"kind": "0", "span": "0"}}]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    tvar = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tvar, tvm.ir.IncompleteType)


def test_func_tuple_type():
    nodes = [
        {"type_key": ""},
        {"type_key": "relay.FuncType",
         "attrs": {
             "arg_types": "2",
             "ret_type": "3",
             "span": "0",
             "type_constraints": "6",
             "type_params": "5"
         }
        },
        {"type_key": "Array"},
        {"type_key": "relay.TupleType",
         "attrs": { "fields": "4", "span": "0" }},
        {"type_key": "Array"},
        {"type_key": "Array"},
        {"type_key": "Array"}
    ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    tvar = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tvar, tvm.ir.FuncType)


def test_global_var():
    nodes = [
        {"type_key": ""},
        {"type_key": "relay.GlobalVar",
         "attrs": {
             "_checked_type_": "0",
             "name_hint": "x",
             "span": "0"
         }
        }
    ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    tvar = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tvar, tvm.ir.GlobalVar)
    nodes = [
        {"type_key": ""},
        {"type_key": "GlobalVar",
         "attrs": {
             "_checked_type_": "0",
             "name_hint": "x",
             "span": "0"
         }
        }
    ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    tvar = tvm.ir.load_json(json.dumps(data))
    assert isinstance(tvar, tvm.ir.GlobalVar)


def test_op():
    nodes = [
        {"type_key": ""},
        {"type_key": "relay.Op",
         "global_key": "nn.conv2d"}
    ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    op = tvm.ir.load_json(json.dumps(data))
    assert op == relay.op.get("nn.conv2d")


def test_tir_var():
    nodes = [
        {"type_key": ""},
        {"type_key": "Variable",
         "attrs": {"dtype": "int32", "name": "x"}},
        {"type_key": "SizeVar",
         "attrs": {"dtype": "int32", "name": "y"}},
    ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    x = tvm.ir.load_json(json.dumps(data))
    assert isinstance(x, tvm.tir.Var)
    assert x.name == "x"
    data["root"] = 2
    y = tvm.ir.load_json(json.dumps(data))
    assert isinstance(y, tvm.tir.SizeVar)
    assert y.name == "y"


def test_str_map():
    nodes = [
        {'type_key': ''},
        {'type_key': 'StrMap', 'keys': ['z', 'x'], 'data': [2, 3]},
        {'type_key': 'IntImm', 'attrs': {'dtype': 'int32', 'value': '2'}},
        {'type_key': 'Max', 'attrs': {'a': '4', 'b': '10', 'dtype': 'int32'}},
        {'type_key': 'Add', 'attrs': {'a': '5', 'b': '9', 'dtype': 'int32'}},
        {'type_key': 'Add', 'attrs': {'a': '6', 'b': '8', 'dtype': 'int32'}},
        {'type_key': 'tir.Var', 'attrs': {'dtype': 'int32', 'name': '7', 'type_annotation': '0'}},
        {'type_key': 'runtime.String', 'repr_str': 'x'},
        {'type_key': 'IntImm', 'attrs': {'dtype': 'int32', 'value': '1'}},
        {'type_key': 'IntImm', 'attrs': {'dtype': 'int32', 'value': '2'}},
        {'type_key': 'IntImm', 'attrs': {'dtype': 'int32', 'value': '100'}}
    ]
    data = {
        "root" : 1,
        "nodes": nodes,
        "attrs": {"tvm_version": "0.6.0"},
        "b64ndarrays": [],
    }
    x = tvm.ir.load_json(json.dumps(data))
    assert(isinstance(x, tvm.ir.container.Map))
    assert(len(x) == 2)
    assert('x' in x)
    assert('z' in x)
    assert(bool(x['z'] == 2))


if __name__ == "__main__":
    test_op()
    test_type_var()
    test_var()
    test_incomplete_type()
    test_func_tuple_type()
    test_global_var()
    test_tir_var()
    test_str_map()
