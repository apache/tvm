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
""" Instrument test cases.
"""
import pytest
import tvm
import tvm.relay
from tvm.relay import op
from tvm.ir.instrument import PassTimingInstrument, pass_instrument


def get_test_model():
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    return tvm.IRModule.from_expr(e3 + e2)


def test_pass_timing_instrument():
    pass_timing = PassTimingInstrument()

    # Override current PassContext's instruments
    tvm.transform.PassContext.current().override_instruments([pass_timing])

    mod = get_test_model()
    mod = tvm.relay.transform.AnnotateSpans()(mod)
    mod = tvm.relay.transform.ToANormalForm()(mod)
    mod = tvm.relay.transform.InferType()(mod)

    profiles = pass_timing.render()
    assert "AnnotateSpans" in profiles
    assert "ToANormalForm" in profiles
    assert "InferType" in profiles

    # Reset current PassContext's instruments to None
    tvm.transform.PassContext.current().override_instruments(None)

    mod = get_test_model()
    mod = tvm.relay.transform.AnnotateSpans()(mod)
    mod = tvm.relay.transform.ToANormalForm()(mod)
    mod = tvm.relay.transform.InferType()(mod)

    profiles = pass_timing.render()
    assert profiles == ""


def test_custom_instrument(capsys):
    @pass_instrument
    class MyTest:
        def enter_pass_ctx(self):
            print("enter ctx")

        def exit_pass_ctx(self):
            print("exit ctx")

        def run_before_pass(self, mod, info):
            print("run before " + info.name)

        def run_after_pass(self, mod, info):
            print("run after " + info.name)

    mod = get_test_model()
    with tvm.transform.PassContext(instruments=[MyTest()]):
        mod = tvm.relay.transform.InferType()(mod)

    assert (
        "enter ctx\n"
        "run before InferType\n"
        "run after InferType\n"
        "exit ctx\n" == capsys.readouterr().out
    )


def test_disable_pass(capsys):
    @pass_instrument
    class CustomPI:
        def should_run(self, mod, info):
            # Only run pass name contains "InferType"
            if "InferType" not in info.name:
                return False
            return True

        def run_before_pass(self, mod, info):
            print(info.name)

    mod = get_test_model()
    with tvm.transform.PassContext(instruments=[CustomPI()]):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

    assert capsys.readouterr().out == "InferType\n"


def test_multiple_instrument(capsys):
    @pass_instrument
    class SkipPass:
        def __init__(self, skip_pass_name):
            self.skip_pass_name = skip_pass_name

        def should_run(self, mod, info):
            if self.skip_pass_name in info.name:
                return False
            return True

    skip_annotate = SkipPass("AnnotateSpans")
    skip_anf = SkipPass("ToANormalForm")

    @pass_instrument
    class PrintPassName:
        def run_before_pass(self, mod, info):
            print(info.name)

    mod = get_test_model()
    print_pass_name = PrintPassName()
    with tvm.transform.PassContext(instruments=[skip_annotate, skip_anf, print_pass_name]):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

    assert capsys.readouterr().out == "InferType\n"


def test_instrument_pass_counts(capsys):
    @pass_instrument
    class PassesCounter:
        def __init__(self):
            self.run_before_count = 0
            self.run_after_count = 0

        def __clear(self):
            self.run_before_count = 0
            self.run_after_count = 0

        def enter_pass_ctx(self):
            self.__clear()

        def exit_pass_ctx(self):
            self.__clear()

        def run_before_pass(self, mod, info):
            self.run_before_count = self.run_before_count + 1

        def run_after_pass(self, mod, info):
            self.run_after_count = self.run_after_count + 1

    mod = get_test_model()
    passes_counter = PassesCounter()
    with tvm.transform.PassContext(instruments=[passes_counter]):
        tvm.relay.build(mod, "llvm")
        assert passes_counter.run_after_count != 0
        assert passes_counter.run_after_count == passes_counter.run_before_count

    # Out of pass context scope, should be reset
    assert passes_counter.run_before_count == 0
    assert passes_counter.run_after_count == 0


def test_enter_pass_ctx_expection(capsys):
    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            print(self.id + " enter ctx")

        def exit_pass_ctx(self):
            print(self.id + " exit ctx")

    @pass_instrument
    class PIBroken(PI):
        def __init__(self, id):
            super().__init__(id)

        def enter_pass_ctx(self):
            print(self.id + " enter ctx")
            raise RuntimeError("Just a dummy error")

    with pytest.raises(tvm.error.TVMError):
        with tvm.transform.PassContext(instruments=[PI("%1"), PIBroken("%2"), PI("%3")]):
            pass

    assert "%1 enter ctx\n" "%2 enter ctx\n" == capsys.readouterr().out


def test_pass_exception(capsys):
    @pass_instrument
    class PI:
        def enter_pass_ctx(self):
            print("enter_pass_ctx")

        def exit_pass_ctx(self):
            print("exit_pass_ctx")

        def should_run(self, mod, info):
            print("should_run")
            return True

        def run_before_pass(self, mod, info):
            print("run_before_pass")

        def run_after_pass(self, mod, info):
            print("run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        print("transform pass")
        raise RuntimeError("Just a dummy error")
        return mod

    mod = get_test_model()
    with pytest.raises(tvm.error.TVMError):
        with tvm.transform.PassContext(instruments=[PI()]):
            mod = transform(mod)

    assert (
        "enter_pass_ctx\n"
        "should_run\n"
        "run_before_pass\n"
        "transform pass\n"
        "exit_pass_ctx\n" == capsys.readouterr().out
    )


def test_should_run_exception(capsys):
    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            print(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            print(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            print(self.id + " should_run")
            raise RuntimeError("Just a dummy error")
            return True

        def run_before_pass(self, mod, info):
            print(self.id + " run_before_pass")

        def run_after_pass(self, mod, info):
            print(self.id + " run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        print("transform pass")
        return mod

    mod = get_test_model()
    with pytest.raises(tvm.error.TVMError):
        with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
            mod = transform(mod)

    assert (
        "%1 enter_pass_ctx\n"
        "%2 enter_pass_ctx\n"
        "%1 should_run\n"
        "%1 exit_pass_ctx\n"
        "%2 exit_pass_ctx\n" == capsys.readouterr().out
    )


def test_run_before_exception(capsys):
    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            print(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            print(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            print(self.id + " should_run")
            return True

        def run_before_pass(self, mod, info):
            print(self.id + " run_before_pass")
            raise RuntimeError("Just a dummy error")

        def run_after_pass(self, mod, info):
            print(self.id + " run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        print("transform pass")
        return mod

    mod = get_test_model()
    with pytest.raises(tvm.error.TVMError):
        with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
            mod = transform(mod)

    assert (
        "%1 enter_pass_ctx\n"
        "%2 enter_pass_ctx\n"
        "%1 should_run\n"
        "%2 should_run\n"
        "%1 run_before_pass\n"
        "%1 exit_pass_ctx\n"
        "%2 exit_pass_ctx\n" == capsys.readouterr().out
    )


def test_run_after_exception(capsys):
    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            print(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            print(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            print(self.id + " should_run")
            return True

        def run_before_pass(self, mod, info):
            print(self.id + " run_before_pass")

        def run_after_pass(self, mod, info):
            print(self.id + " run_after_pass")
            raise RuntimeError("Just a dummy error")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        print("transform pass")
        return mod

    x, y = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xy"]
    mod = tvm.IRModule.from_expr(tvm.relay.add(x, y))

    with pytest.raises(tvm.error.TVMError):
        with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
            mod = transform(mod)

    assert (
        "%1 enter_pass_ctx\n"
        "%2 enter_pass_ctx\n"
        "%1 should_run\n"
        "%2 should_run\n"
        "%1 run_before_pass\n"
        "%2 run_before_pass\n"
        "transform pass\n"
        "%1 run_after_pass\n"
        "%1 exit_pass_ctx\n"
        "%2 exit_pass_ctx\n" == capsys.readouterr().out
    )


def test_instrument_call_sequence(capsys):
    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            print(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            print(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            print("  " + self.id + " should_run")
            return True

        def run_before_pass(self, mod, info):
            print("  " + self.id + " run_before_pass")

        def run_after_pass(self, mod, info):
            print("  " + self.id + " run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform1(mod, ctx):
        print("    transform1 pass")
        return mod

    @tvm.transform.module_pass(opt_level=2)
    def transform2(mod, ctx):
        print("    transform2 pass")
        return mod

    mod = get_test_model()
    with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
        mod = transform1(mod)
        mod = transform2(mod)

    assert (
        "%1 enter_pass_ctx\n"
        "%2 enter_pass_ctx\n"
        "  %1 should_run\n"
        "  %2 should_run\n"
        "  %1 run_before_pass\n"
        "  %2 run_before_pass\n"
        "    transform1 pass\n"
        "  %1 run_after_pass\n"
        "  %2 run_after_pass\n"
        "  %1 should_run\n"
        "  %2 should_run\n"
        "  %1 run_before_pass\n"
        "  %2 run_before_pass\n"
        "    transform2 pass\n"
        "  %1 run_after_pass\n"
        "  %2 run_after_pass\n"
        "%1 exit_pass_ctx\n"
        "%2 exit_pass_ctx\n" == capsys.readouterr().out
    )
