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
from tvm.ir.transform.instrument import PrintAfterAll, PrintBeforeAll
from tvm.script import ir as I
from tvm.script import tirx as T


def test_function_pass(capsys):
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(x: T.int32, y: T.int32, B: T.Tensor((16,), "float32")):
            value = T.bind(10)
            T.evaluate(value + 1)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(x: T.int32, y: T.int32, B: T.Tensor((16,), "float32")):
            T.evaluate(0)

    @tvm.tirx.transform.function_pass(opt_level=1)
    class TestReplaceFunc:
        def __init__(self, new_func):
            self.new_func = new_func

        def transform_function(self, func, mod, ctx):
            return self.new_func

    with tvm.ir.transform.PassContext(instruments=[PrintBeforeAll(), PrintAfterAll()]):
        After = TestReplaceFunc(Expected["main"])(Before)
    output = capsys.readouterr().out
    assert "Before Running Pass:" in output
    assert "After Running Pass:" in output
    assert 'name="TestReplaceFunc"' in output
    tvm.ir.assert_structural_equal(After, Expected)


def test_cow_pass():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(x: T.int32):
            T.func_attr({"target_bits": 32})
            T.evaluate(x)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(x: T.int32):
            T.func_attr({"target_bits": 32})
            T.evaluate(x)

    def fapply(func):
        assert tvm.testing.object_use_count(func) == 1
        return func

    func_hash = hash(Before["main"])
    mod_hash = hash(Before)
    After = tvm.ir.transform.Sequential(
        [tvm.tirx.transform.Apply(fapply), tvm.tirx.transform.NarrowDataType(32)]
    )(Before._move())
    tvm.ir.assert_structural_equal(After, Expected)
    assert mod_hash == hash(After)
    assert func_hash == hash(After["main"])


if __name__ == "__main__":
    tvm.testing.main()
