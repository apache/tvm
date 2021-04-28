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
from tvm.ir.instrument import PassesTimeInstrument, PassInstrument, PassInstrumentor


def test_pass_time_instrument():
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    time_instrument = PassesTimeInstrument()
    with tvm.transform.PassContext(pass_instrumentor=PassInstrumentor([time_instrument])):
        mod = tvm.relay.transform.AnnotateSpans()(mod)
        mod = tvm.relay.transform.ToANormalForm()(mod)
        mod = tvm.relay.transform.InferType()(mod)

        profiles = time_instrument.render()
        assert "AnnotateSpans" in profiles
        assert "ToANormalForm" in profiles
        assert "InferType" in profiles


def test_custom_instrument(capsys):
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    def custom_pi():
        pi = PassInstrument("MyTest")

        @pi.register_set_up
        def set_up():
            print("set up")

        @pi.register_tear_down
        def tear_down():
            print("tear down")

        @pi.register_run_before_pass
        def run_before_pass(mod, info):
            print("run before " + info.name)
            return True

        @pi.register_run_after_pass
        def run_after_pass(mod, info):
            print("run after " + info.name)

        return pi

    with tvm.transform.PassContext(pass_instrumentor=PassInstrumentor([custom_pi()])):
        mod = tvm.relay.transform.InferType()(mod)

    output = "set up\n" "run before InferType\n" "run after InferType\n" "tear down\n"
    assert capsys.readouterr().out == output


def test_disable_pass(capsys):
    x, y, z = [tvm.relay.var(c, shape=(3, 4), dtype="float32") for c in "xyz"]
    e1 = op.add(x, y)
    e2 = op.subtract(x, z)
    e3 = op.multiply(e1, e1 / e2)
    mod = tvm.IRModule.from_expr(e3 + e2)

    def custom_pi():
        pi = PassInstrument("MyTest")

        @pi.register_run_before_pass
        def run_before_pass(mod, info):
            # Only run pass name contains "InferType"
            if "InferType" not in info.name:
                return False

            print(info.name)
            return True

        return pi

    with tvm.transform.PassContext(pass_instrumentor=PassInstrumentor([custom_pi()])):
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

    def custom_pi(skip_pass_name):
        def create_custom_pi():
            pi = PassInstrument("Don't care")

            @pi.register_run_before_pass
            def run_before_pass(mod, info):
                if skip_pass_name in info.name:
                    return False

                return True

            return pi

        return create_custom_pi()

    skip_annotate = custom_pi("AnnotateSpans")
    skip_anf = custom_pi("ToANormalForm")

    print_pass_name = PassInstrument("PrintPassName")

    @print_pass_name.register_run_before_pass
    def run_before_pass(mod, info):
        print(info.name)
        return True

    with tvm.transform.PassContext(
        pass_instrumentor=PassInstrumentor([skip_annotate, skip_anf, print_pass_name])
    ):
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

    class PassesCounter(PassInstrument):
        def __init__(self):
            super().__init__("PassesCounter")
            super().register_set_up(self.__set_up)
            super().register_tear_down(self.__tear_down)
            super().register_run_before_pass(self.__run_before_pass)
            super().register_run_after_pass(self.__run_after_pass)
            self.__clear()

        def __clear(self):
            self.run_before_count = 0
            self.run_after_count = 0

        def __set_up(self):
            self.__clear()

        def __tear_down(self):
            self.__clear()

        def __run_before_pass(self, mod, info):
            self.run_before_count = self.run_before_count + 1
            return True

        def __run_after_pass(self, mod, info):
            self.run_after_count = self.run_after_count + 1

    passes_counter = PassesCounter()
    with tvm.transform.PassContext(pass_instrumentor=PassInstrumentor([passes_counter])):
        tvm.relay.build(mod, "llvm")
        assert passes_counter.run_after_count != 0
        assert passes_counter.run_after_count == passes_counter.run_before_count

    # Out of pass context scope, should be reset
    assert passes_counter.run_before_count == 0
    assert passes_counter.run_after_count == 0
