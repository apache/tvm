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
import tvm.testing
from tvm import te


def test_prim_func_pass():
    @tvm.tir.transform.prim_func_pass(opt_level=1)
    class TestReplaceFunc:
        """Simple test function to replace one argument to another."""

        def __init__(self, new_func):
            self.new_func = new_func

        def transform_function(self, func, mod, ctx):
            return self.new_func

    x = te.var("x")
    y = te.var("y")
    b = tvm.tir.decl_buffer((x,), "float32")
    stmt = tvm.tir.LetStmt(x, 10, tvm.tir.Evaluate(x + 1))

    func = tvm.tir.PrimFunc([x, y, b], stmt)

    new_func = tvm.tir.PrimFunc([x, y, b], tvm.tir.Evaluate(0))

    mod = tvm.IRModule({"main": func})
    mod = TestReplaceFunc(new_func)(mod)

    assert tvm.ir.structural_equal(mod["main"].body, new_func.body)


def test_cow_pass():
    def fapply(f):
        assert tvm.testing.object_use_count(f) == 1
        return f

    pidentity = tvm.tir.transform.Apply(fapply)
    x = te.var("x")
    func = tvm.tir.PrimFunc([x], tvm.tir.Evaluate(x)).with_attr("target_bits", 32)
    func_hash = func.__hash__()
    mod = tvm.IRModule({"main": func})
    del func
    # copy on write
    mod_hash = mod.__hash__()
    mod = tvm.transform.Sequential([pidentity, tvm.tir.transform.NarrowDataType(32)])(mod._move())
    assert mod_hash == mod.__hash__()
    assert func_hash == mod["main"].__hash__()


if __name__ == "__main__":
    test_cow_pass()
    test_prim_func_pass()
