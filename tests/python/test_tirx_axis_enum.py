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
"""Axis enum identity and import behavior across compiler and runtime builds."""

import os
import subprocess
import sys

import pytest
from tvm_ffi.dataclasses import Enum

import tvm
from tvm.tirx.layout import Axis, S, TileLayout, laneid


def test_axis_enum_identity_and_attributes():
    axis = Axis.laneid
    assert isinstance(axis, Enum)
    assert axis.same_as(Axis.get("laneid"))
    assert axis.same_as(laneid)
    assert axis.name == "laneid"
    assert axis in Axis.all_entries()
    assert hash(axis) == hash(Axis.get("laneid"))
    assert Axis.def_attr("thread")[axis] is True
    assert Axis.m.is_memory()
    assert Axis.bx.get_scope().name == "thread"
    assert Axis.bx.get_subscope().name == "cta"
    assert TileLayout(S[4 : 1 @ axis]).shard[0].axis.same_as(axis)


def test_axis_unknown_name_and_json_roundtrip():
    axis = Axis.get("axis_enum_unknown_test")
    assert axis.same_as(Axis.get("axis_enum_unknown_test"))
    assert axis.get_scope() is None
    assert axis.get_subscope() is None
    with pytest.raises(Exception, match="has no thread classification"):
        axis.is_thread()
    with pytest.raises(Exception, match="has no thread classification"):
        axis.is_memory()
    assert tvm.ir.load_json(tvm.ir.save_json(axis)).same_as(axis)
    assert tvm.ir.load_json(tvm.ir.save_json(Axis.laneid)).same_as(Axis.laneid)


def test_axis_runtime_only_import():
    env = os.environ.copy()
    env["TVM_USE_RUNTIME_LIB"] = "1"
    subprocess.run(
        [sys.executable, "-c", "import tvm; import tvm.tirx.layout; assert tvm.base._RUNTIME_ONLY"],
        env=env,
        check=True,
    )
