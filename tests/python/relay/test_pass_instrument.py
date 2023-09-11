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


instrument_definition_type = tvm.testing.parameter("decorator", "subclass")


def test_custom_instrument(instrument_definition_type):
    class BaseTest:
        def __init__(self):
            self.events = []

        def enter_pass_ctx(self):
            self.events.append("enter ctx")

        def exit_pass_ctx(self):
            self.events.append("exit ctx")

        def run_before_pass(self, mod, info):
            self.events.append("run before " + info.name)

        def run_after_pass(self, mod, info):
            self.events.append("run after " + info.name)

    if instrument_definition_type == "decorator":
        MyTest = pass_instrument(BaseTest)

    elif instrument_definition_type == "subclass":

        class MyTest(BaseTest, tvm.ir.instrument.PassInstrument):
            def __init__(self):
                BaseTest.__init__(self)
                tvm.ir.instrument.PassInstrument.__init__(self)

    mod = get_test_model()
    my_test = MyTest()
    with tvm.transform.PassContext(instruments=[my_test]):
        mod = tvm.relay.transform.InferType()(mod)

    assert (
        "enter ctx"
        "run before InferType"
        "run after InferType"
        "exit ctx" == "".join(my_test.events)
    )


def test_disable_pass():
    @pass_instrument
    class CustomPI:
        def __init__(self):
            self.events = []

        def should_run(self, mod, info):
            # Only run pass name contains "InferType"
            if "InferType" not in info.name:
                return False
            return True

        def run_before_pass(self, mod, info):
            self.events.append(info.name)

    mod = get_test_model()
    custom_pi = CustomPI()
    with tvm.transform.PassContext(instruments=[custom_pi]):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

    assert "InferType" == "".join(custom_pi.events)


def test_multiple_instrument():
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
        def __init__(self):
            self.events = []

        def run_before_pass(self, mod, info):
            self.events.append(info.name)

    mod = get_test_model()
    print_pass_name = PrintPassName()
    with tvm.transform.PassContext(instruments=[skip_annotate, skip_anf, print_pass_name]):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

    assert "InferType" == "".join(print_pass_name.events)


def test_instrument_pass_counts():
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


def test_list_pass_configs():
    configs = tvm.transform.PassContext.list_configs()

    assert len(configs) > 0
    assert "relay.backend.use_auto_scheduler" in configs.keys()
    assert configs["relay.backend.use_auto_scheduler"]["type"] == "IntImm"


def test_enter_pass_ctx_exception():
    events = []

    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            events.append(self.id + " enter ctx")

        def exit_pass_ctx(self):
            events.append(self.id + " exit ctx")

    @pass_instrument
    class PIBroken(PI):
        def __init__(self, id):
            super().__init__(id)

        def enter_pass_ctx(self):
            events.append(self.id + " enter ctx")
            raise RuntimeError("Just a dummy error")

    pass_ctx = tvm.transform.PassContext(instruments=[PI("%1"), PIBroken("%2"), PI("%3")])
    with pytest.raises(RuntimeError) as cm:
        with pass_ctx:
            pass
        assert "Just a dummy error" in str(cm.execption)

    assert "%1 enter ctx" "%2 enter ctx" "%1 exit ctx" == "".join(events)

    # Make sure we get correct PassContext
    cur_pass_ctx = tvm.transform.PassContext.current()
    assert pass_ctx != cur_pass_ctx
    assert not cur_pass_ctx.instruments


def test_enter_pass_ctx_exception_global():
    @pass_instrument
    class PIBroken:
        def enter_pass_ctx(self):
            raise RuntimeError("Just a dummy error")

    cur_pass_ctx = tvm.transform.PassContext.current()
    with pytest.raises(RuntimeError) as cm:
        cur_pass_ctx.override_instruments([PIBroken()])
        assert "Just a dummy error" in str(cm.exception)
    assert not cur_pass_ctx.instruments


def test_exit_pass_ctx_exception():
    events = []

    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def exit_pass_ctx(self):
            events.append(self.id + " exit ctx")

    @pass_instrument
    class PIBroken(PI):
        def __init__(self, id):
            super().__init__(id)

        def exit_pass_ctx(self):
            events.append(self.id + " exit ctx")
            raise RuntimeError("Just a dummy error")

    pass_ctx = tvm.transform.PassContext(instruments=[PI("%1"), PIBroken("%2"), PI("%3")])
    with pytest.raises(RuntimeError) as cm:
        with pass_ctx:
            pass
        assert "Just a dummy error" in str(cm.exception)

    assert "%1 exit ctx" "%2 exit ctx" == "".join(events)

    # Make sure we get correct PassContext
    cur_pass_ctx = tvm.transform.PassContext.current()
    assert pass_ctx != cur_pass_ctx
    assert not cur_pass_ctx.instruments


def test_exit_pass_ctx_exception_global():
    @pass_instrument
    class PIBroken:
        def exit_pass_ctx(self):
            raise RuntimeError("Just a dummy error")

    cur_pass_ctx = tvm.transform.PassContext.current()
    with pytest.raises(RuntimeError) as cm:
        cur_pass_ctx.override_instruments([PIBroken()])
        cur_pass_ctx.override_instruments([PIBroken()])
        assert "Just a dummy error" in str(cm.exception)
    assert not cur_pass_ctx.instruments


def test_pass_exception():
    events = []

    @pass_instrument
    class PI:
        def enter_pass_ctx(self):
            events.append("enter_pass_ctx")

        def exit_pass_ctx(self):
            events.append("exit_pass_ctx")

        def should_run(self, mod, info):
            events.append("should_run")
            return True

        def run_before_pass(self, mod, info):
            events.append("run_before_pass")

        def run_after_pass(self, mod, info):
            events.append("run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        events.append("transform pass")
        raise RuntimeError("Just a dummy error")
        return mod

    mod = get_test_model()
    with pytest.raises(RuntimeError) as cm:
        with tvm.transform.PassContext(instruments=[PI()]):
            mod = transform(mod)
        assert "Just a dummy error" in str(cm.exception)

    assert (
        "enter_pass_ctx"
        "should_run"
        "run_before_pass"
        "transform pass"
        "exit_pass_ctx" == "".join(events)
    )


def test_should_run_exception():
    events = []

    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            events.append(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            events.append(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            events.append(self.id + " should_run")
            raise RuntimeError("Just a dummy error")
            return True

        def run_before_pass(self, mod, info):
            events.append(self.id + " run_before_pass")

        def run_after_pass(self, mod, info):
            events.append(self.id + " run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        events.append("transform pass")
        return mod

    mod = get_test_model()
    with pytest.raises(RuntimeError) as cm:
        with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
            mod = transform(mod)
        assert "Just a dummy error" in str(cm.exception)

    assert (
        "%1 enter_pass_ctx"
        "%2 enter_pass_ctx"
        "%1 should_run"
        "%1 exit_pass_ctx"
        "%2 exit_pass_ctx" == "".join(events)
    )


def test_run_before_exception():
    events = []

    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            events.append(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            events.append(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            events.append(self.id + " should_run")
            return True

        def run_before_pass(self, mod, info):
            events.append(self.id + " run_before_pass")
            raise RuntimeError("Just a dummy error")

        def run_after_pass(self, mod, info):
            events.append(self.id + " run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        events.append("transform pass")
        return mod

    mod = get_test_model()
    with pytest.raises(RuntimeError) as cm:
        with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
            mod = transform(mod)
        assert "Just a dummy error" in str(cm.exception)

    assert (
        "%1 enter_pass_ctx"
        "%2 enter_pass_ctx"
        "%1 should_run"
        "%2 should_run"
        "%1 run_before_pass"
        "%1 exit_pass_ctx"
        "%2 exit_pass_ctx" == "".join(events)
    )


def test_run_after_exception():
    events = []

    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            events.append(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            events.append(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            events.append(self.id + " should_run")
            return True

        def run_before_pass(self, mod, info):
            events.append(self.id + " run_before_pass")

        def run_after_pass(self, mod, info):
            events.append(self.id + " run_after_pass")
            raise RuntimeError("Just a dummy error")

    @tvm.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
        events.append("transform pass")
        return mod

    x, y = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xy"]
    mod = tvm.IRModule.from_expr(tvm.relay.add(x, y))

    with pytest.raises(RuntimeError) as cm:
        with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
            mod = transform(mod)
        assert "Just a dummy error" in str(cm.exception)

    assert (
        "%1 enter_pass_ctx"
        "%2 enter_pass_ctx"
        "%1 should_run"
        "%2 should_run"
        "%1 run_before_pass"
        "%2 run_before_pass"
        "transform pass"
        "%1 run_after_pass"
        "%1 exit_pass_ctx"
        "%2 exit_pass_ctx" == "".join(events)
    )


def test_instrument_call_sequence():
    events = []

    @pass_instrument
    class PI:
        def __init__(self, id):
            self.id = id

        def enter_pass_ctx(self):
            events.append(self.id + " enter_pass_ctx")

        def exit_pass_ctx(self):
            events.append(self.id + " exit_pass_ctx")

        def should_run(self, mod, info):
            events.append("  " + self.id + " should_run")
            return True

        def run_before_pass(self, mod, info):
            events.append("  " + self.id + " run_before_pass")

        def run_after_pass(self, mod, info):
            events.append("  " + self.id + " run_after_pass")

    @tvm.transform.module_pass(opt_level=2)
    def transform1(mod, ctx):
        events.append("    transform1 pass")
        return mod

    @tvm.transform.module_pass(opt_level=2)
    def transform2(mod, ctx):
        events.append("    transform2 pass")
        return mod

    mod = get_test_model()
    with tvm.transform.PassContext(instruments=[PI("%1"), PI("%2")]):
        mod = transform1(mod)
        mod = transform2(mod)

    assert (
        "%1 enter_pass_ctx"
        "%2 enter_pass_ctx"
        "  %1 should_run"
        "  %2 should_run"
        "  %1 run_before_pass"
        "  %2 run_before_pass"
        "    transform1 pass"
        "  %1 run_after_pass"
        "  %2 run_after_pass"
        "  %1 should_run"
        "  %2 should_run"
        "  %1 run_before_pass"
        "  %2 run_before_pass"
        "    transform2 pass"
        "  %1 run_after_pass"
        "  %2 run_after_pass"
        "%1 exit_pass_ctx"
        "%2 exit_pass_ctx" == "".join(events)
    )
