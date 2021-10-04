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
from tvm import te
import numpy


@pytest.fixture
def mod_without_attrs():
    ib = tvm.tir.ir_builder.create()
    A = tvm.tir.decl_buffer(name="A", shape=[1])
    stmt = ib.get()
    return tvm.IRModule.from_expr(tvm.tir.PrimFunc([A], stmt))


@pytest.fixture
def mod(mod_without_attrs):
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(
        mod_without_attrs
    )
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    return mod


def test_fails_if_not_global_symbol(mod_without_attrs):
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(
        mod_without_attrs
    )
    with pytest.raises(tvm.TVMError, match="Expect PrimFunc to have the global_symbol attribute"):
        f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]


def test_fails_if_no_target(mod_without_attrs):
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod_without_attrs)
    with pytest.raises(tvm.TVMError, match="Require the target attribute"):
        f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]


@tvm.testing.parametrize_targets("c", "llvm", "cuda")
def test_device_setup(mod, target, dev):
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target(target)))(mod)
    f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 1
    assert f.params[0].name == "arg0"
    assert f.body.node == "default"
    assert f.body.attr_key == "device_id"
    assert f.body.value == 0
    assert f.body.body.node == "default"
    assert f.body.body.attr_key == "device_type"
    assert f.body.body.value == dev.device_type


def test_no_buffers_no_device_setup():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A], stmt))
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 1
    assert f.body.var.name == "A"
    assert f.body.value.name == "arg0"


def test_argument_mapping(mod):
    f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 1
    assert f.params[0].name == "arg0"
    assert f.body.body.body.var.name == "A"
    assert f.body.body.body.value.name == "arg0"


def test_argument_mapping_multiple():
    ib = tvm.tir.ir_builder.create()
    A = tvm.tir.decl_buffer(name="A", shape=[1])
    B = tvm.tir.decl_buffer(name="B", shape=[1])

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, B], stmt))
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 2
    assert f.params[0].name == "arg0"
    assert f.params[1].name == "arg1"
    assert f.body.body.body.var.name == "A"
    assert f.body.body.body.value.name == "arg0"
    assert f.body.body.body.body.var.name == "B"
    assert f.body.body.body.body.value.name == "arg1"


def test_argument_mapping_multiple_matching():
    ib = tvm.tir.ir_builder.create()
    A = tvm.tir.decl_buffer(name="A", shape=[1])
    B = tvm.tir.decl_buffer(name="B", shape=[1])
    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, A], stmt))
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)

    f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 2
    assert f.params[0].name == "arg0"
    assert f.params[1].name == "arg1"
    assert f.body.body.body.var.name == "A"
    assert f.body.body.body.value.name == "arg0"
    assert f.body.body.body.body.condition.a.name == "A"
    assert f.body.body.body.body.condition.b.name == "arg1"


def test_body():
    ib = tvm.tir.ir_builder.create()
    A = tvm.tir.decl_buffer(name="A", shape=[1])
    B = tvm.tir.decl_buffer(name="B", shape=[1])
    C = ib.buffer_ptr(A.data)

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, B, C], stmt))
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    f = tvm.tir.transform.MakeUnpackedAPI()(mod)["main"]
    assert len(f.params) == 3
    assert f.params[0].name == "arg0"
    assert f.params[1].name == "arg1"
    assert f.params[2].name == "arg2"
    assert f.body.body.body.var.name == "A"
    assert f.body.body.body.value.name == "arg2"
    assert f.body.body.body.body.var.name == "B"
    assert f.body.body.body.body.value.name == "arg1"
    assert f.body.body.body.body.body.condition.a.name == "A"
    assert f.body.body.body.body.body.condition.b.name == "arg0"


if __name__ == "__main__":
    pytest.main([__file__])
