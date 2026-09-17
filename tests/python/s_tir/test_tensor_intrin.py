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
"""Tensor intrinsic ownership and runtime compatibility."""

import json

import pytest

import tvm
import tvm.testing
from tvm import s_tir, tirx


def test_tensor_intrin_serialization():
    func = tirx.PrimFunc([], tirx.Evaluate(0))
    intrin = s_tir.TensorIntrin(func, func)
    graph = json.loads(tvm.ir.save_json(intrin))
    assert any(node.get("type") == "s_tir.TensorIntrin" for node in graph["nodes"])
    for legacy in (False, True):
        if legacy:
            for node in graph["nodes"]:
                if node.get("type") == "s_tir.TensorIntrin":
                    node["type"] = "tirx.TensorIntrin"
        restored = tvm.ir.load_json(json.dumps(graph))
        assert isinstance(restored, s_tir.TensorIntrin)
        assert restored.desc.same_as(restored.impl)
        tvm.ir.assert_structural_equal(restored.desc, func)


def test_tensor_intrin_registration():
    func = tirx.PrimFunc([], tirx.Evaluate(0))
    name = "test_s_tir_tensor_intrin_registration"
    s_tir.TensorIntrin.register(name, func, func, override=True)
    assert s_tir.TensorIntrin.get(name).desc.same_as(func)
    with pytest.raises(ValueError, match="already been registered"):
        s_tir.TensorIntrin.register(name, func, func)
    replacement = tirx.PrimFunc([], tirx.Evaluate(1))
    s_tir.TensorIntrin.register(name, func, replacement, override=True)
    assert s_tir.TensorIntrin.get(name).impl.same_as(replacement)


def test_tensor_intrin_constructor_constraints():
    empty = tirx.PrimFunc([], tirx.Evaluate(0))
    scalar = tirx.PrimFunc([tirx.Var("x", "int32")], tirx.Evaluate(0))
    with pytest.raises(ValueError, match="number of parameters"):
        s_tir.TensorIntrin(empty, scalar)
    with pytest.raises(ValueError, match="description.*handle only"):
        s_tir.TensorIntrin(scalar, scalar)
    pointer = tirx.PrimFunc(
        [tirx.Var("p", tvm.ir.PointerType(tvm.ir.PrimType("float32")))], tirx.Evaluate(0)
    )
    with pytest.raises(ValueError, match="implementation.*handle only"):
        s_tir.TensorIntrin(pointer, scalar)


if __name__ == "__main__":
    tvm.testing.main()
