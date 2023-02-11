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
from tvm import relax
from tvm.ir import structural_equal
from tvm.ir.base import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R


def test_to_non_dataflow():
    @tvm.script.ir_module
    class TestToNonDataflow:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.var("int64"), T.var("int64")
            with R.dataflow():
                lv0 = R.call_tir("test.op.identity", (x,), R.Tensor((m, n), dtype="float32"))
                gv0 = R.call_tir("test.op.identity", (lv0,), R.Tensor((m, n), dtype="float32"))
                R.output(gv0)
            return gv0

    mod = TestToNonDataflow

    old_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal old_vars
            old_vars.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)
    x, lv0, gv0 = old_vars

    new_mod = relax.transform.ToNonDataflow()(mod)

    new_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var):
            nonlocal new_vars
            new_vars.append(e)

    relax.analysis.post_order_visit(new_mod["foo"], fvisit)

    assert x == new_vars[0]
    assert lv0 != new_vars[1]
    assert isinstance(lv0, relax.DataflowVar)
    assert not isinstance(new_vars[1], relax.DataflowVar)

    assert isinstance(gv0, relax.Var)
    assert isinstance(new_vars[2], relax.Var)
    assert gv0 == new_vars[2]


def test_call_tir_rewrite():
    @tvm.script.ir_module
    class TestCallTIRRewrite:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.var("int64"), T.var("int64")
            gv0 = R.call_tir("test.op.identity", (x,), R.Tensor((m, n), dtype="float32"))
            return gv0

    mod = TestCallTIRRewrite

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, relax.Call)
    assert s0.op.name == "relax.call_tir"

    # after rewrite
    new_mod = relax.transform.CallTIRRewrite()(mod)
    func = new_mod["foo"]

    block = func.body.blocks[0]
    assert not isinstance(block, relax.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, relax.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    assert structural_equal(s1.args[0], s0.sinfo_args[0].shape)
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


def test_vm_builtin_lower():
    @tvm.script.ir_module
    class TestVMBuiltinLower:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor:
            m, n = T.var("int64"), T.var("int64")
            alloc = R.builtin.alloc_tensor((m, n), runtime_device_index=0, dtype="float32")
            _ = R.call_packed(
                "test.op.identity", x, alloc, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    mod = TestVMBuiltinLower

    # after vm builtin lowering
    new_mod = relax.transform.VMBuiltinLower()(mod)
    func = new_mod["foo"]

    assert isinstance(new_mod, tvm.IRModule)
    assert isinstance(func, tvm.relax.expr.Function)

    block = func.body.blocks[0]
    s1 = block.bindings[0].value
    assert isinstance(s1, relax.Call)
    assert s1.op.name == "relax.vm.alloc_storage"
    s2 = block.bindings[1].value
    assert isinstance(s2, relax.Call)
    s3 = block.bindings[2].value
    assert isinstance(s3, relax.Call)
    assert isinstance(s3.op, relax.ExternFunc)
    assert s3.op.global_symbol == "test.op.identity"


if __name__ == "__main__":
    pytest.main([__file__])
