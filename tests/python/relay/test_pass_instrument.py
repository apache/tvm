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
import tvm.relay
from tvm.relay import op
from tvm.ir.instrument import PassTimingInstrument, pass_instrument


def test_pass_timing_instrument():
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    pass_timing = PassTimingInstrument()
    with tvm.transform.PassContext(instruments=[pass_timing]):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

        profiles = pass_timing.render()
        assert "AnnotateSpans" in profiles
        assert "ToANormalForm" in profiles
        assert "InferType" in profiles


def test_custom_instrument(capsys):
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    @pass_instrument
    class MyTest:
        def set_up(self):
            print("set up")

        def tear_down(self):
            print("tear down")

        def run_before_pass(self, mod, info):
            print("run before " + info.name)
            return True

        def run_after_pass(self, mod, info):
            print("run after " + info.name)

    with tvm.transform.PassContext(instruments=[MyTest()]):
        mod = tvm.relay.transform.InferType()(mod)

    output = "set up\n" "run before InferType\n" "run after InferType\n" "tear down\n"
    assert capsys.readouterr().out == output


def test_disable_pass(capsys):
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    @pass_instrument
    class CustomPI:
        def run_before_pass(self, mod, info):
            # Only run pass name contains "InferType"
            if "InferType" not in info.name:
                return False

            print(info.name)
            return True

    with tvm.transform.PassContext(instruments=[CustomPI()]):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

    assert capsys.readouterr().out == "InferType\n"


def test_multiple_instrument(capsys):
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    @pass_instrument
    class SkipPass:
        def __init__(self, skip_pass_name):
            self.skip_pass_name = skip_pass_name

        def run_before_pass(self, mod, info):
            if self.skip_pass_name in info.name:
                return False
            return True

    skip_annotate = SkipPass("AnnotateSpans")
    skip_anf = SkipPass("ToANormalForm")

    @pass_instrument
    class PrintPassName:
        def run_before_pass(self, mod, info):
            print(info.name)
            return True

    print_pass_name = PrintPassName()

    with tvm.transform.PassContext(instruments=[skip_annotate, skip_anf, print_pass_name]):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

    assert capsys.readouterr().out == "InferType\n"


def test_instrument_pass_counts(capsys):
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    @pass_instrument
    class PassesCounter:
        def __init__(self):
            self.run_before_count = 0
            self.run_after_count = 0

        def __clear(self):
            self.run_before_count = 0
            self.run_after_count = 0

        def set_up(self):
            self.__clear()

        def tear_down(self):
            self.__clear()

        def run_before_pass(self, mod, info):
            self.run_before_count = self.run_before_count + 1
            return True

        def run_after_pass(self, mod, info):
            self.run_after_count = self.run_after_count + 1

    passes_counter = PassesCounter()
    with tvm.transform.PassContext(instruments=[passes_counter]):
        tvm.relay.build(mod, "llvm")
        assert passes_counter.run_after_count != 0
        assert passes_counter.run_after_count == passes_counter.run_before_count

    # Out of pass context scope, should be reset
    assert passes_counter.run_before_count == 0
    assert passes_counter.run_after_count == 0
