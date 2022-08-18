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
"""
This file tests the FFI binding of script.printer.VarTable.
These only make sure parameter can be passed to the C++ functions
correctly. The test for the functionality of VarTable is in C++.
"""

from tvm.runtime import ObjectPath
from tvm.script.printer.doc import LiteralDoc
from tvm.script.printer.frame import VarDefFrame
from tvm.script.printer.var_table import VarTable
from tvm.tir import Var


def test_define():
    var_table = VarTable()
    var_name = "a"
    var_obj = Var(var_name, dtype="int32")
    object_path = ObjectPath.root().attr("a")
    frame = VarDefFrame()

    id_doc = var_table.define(var_obj, var_name, object_path, frame)

    assert id_doc.name == "a"
    assert list(id_doc.source_paths) == [object_path]

    id_doc = var_table.get_var_doc(var_obj, object_path)

    assert id_doc.name == "a"
    assert list(id_doc.source_paths) == [object_path]


def test_define_by_doc():
    var_table = VarTable()
    var_name = "a"
    var_obj = Var(var_name, dtype="int32")
    object_path = ObjectPath.root().attr("a")
    frame = VarDefFrame()

    var_table.define_by_doc(var_obj, lambda: LiteralDoc(var_name), frame)

    var_doc = var_table.get_var_doc(var_obj, object_path)

    assert isinstance(var_doc, LiteralDoc)
    assert var_doc.value == var_name
    assert list(var_doc.source_paths) == [object_path]


def test_is_var_defined():
    var_table = VarTable()
    a = Var("a", dtype="int32")
    object_path = ObjectPath.root().attr("a")
    frame = VarDefFrame()

    var_table.define(a, "a", object_path, frame)

    assert var_table.is_var_defined(a)
    assert a in var_table


def test_var_out_of_scope():
    var_table = VarTable()
    var_name = "a"
    var_obj = Var(var_name, dtype="int32")
    object_path = ObjectPath.root().attr("a")
    frame = VarDefFrame()

    var_table.define(var_obj, var_name, object_path, frame)

    with frame:
        assert var_obj in var_table

    assert var_obj not in var_table
    assert var_table.get_var_doc(var_obj, object_path) is None
