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
"""Ordered transformations, scoped controls, and canonical instrumentation."""

import importlib

import pytest

import tvm
from tvm import transform
from tvm.transform import instrument


def record_pass(events, name, level=0):
    @transform.module_pass(opt_level=level, name=name)
    def run(mod, _ctx):
        events.append(name)
        return mod

    return run


def test_nested_sequence_and_context_controls():
    events = []
    pipeline = transform.Sequential(
        [
            record_pass(events, "first"),
            transform.Sequential(
                [
                    record_pass(events, "high", 3),
                    record_pass(events, "disabled"),
                ]
            ),
            record_pass(events, "last"),
        ]
    )
    with transform.PassContext(opt_level=0):
        pipeline(tvm.IRModule())
    assert events == ["first", "disabled", "last"]
    events.clear()
    with transform.PassContext(
        opt_level=0,
        required_pass=["high", "disabled", "not_in_pipeline"],
        disabled_pass=["disabled"],
    ):
        pipeline(tvm.IRModule())
    assert events == ["first", "high", "last"]
    assert pipeline.passes[1].passes[0].info.name == "high"


def test_instrument_identity_and_callbacks():
    core = importlib.import_module("tvm.transform.core")
    assert transform.PassInstrument is core.PassInstrument
    assert transform.pass_instrument is core.pass_instrument
    assert importlib.import_module("tvm.transform.instrument") is instrument
    assert isinstance(instrument.PassTimingInstrument(), transform.PassInstrument)
    events = []

    @transform.pass_instrument
    class Recorder:
        def enter_pass_ctx(self):
            events.append("enter")

        def should_run(self, mod, info):
            events.append("check:" + info.name)
            return info.name != "skip"

        def run_before_pass(self, mod, info):
            events.append("before:" + info.name)

        def run_after_pass(self, mod, info):
            events.append("after:" + info.name)

        def exit_pass_ctx(self):
            events.append("exit")

    recorder = Recorder()
    assert isinstance(recorder, transform.PassInstrument)
    context = transform.PassContext(instruments=[recorder])
    assert isinstance(context.instruments[0], transform.PassInstrument)
    with context:
        record_pass(events, "run")(tvm.IRModule())
        record_pass(events, "skip")(tvm.IRModule())
    assert events == ["enter", "check:run", "before:run", "run", "after:run", "check:skip", "exit"]


def test_instrument_exception_and_required_pass():
    events = []

    class Skip(transform.PassInstrument):
        def should_run(self, mod, info):
            events.append("check")
            return False

        def exit_pass_ctx(self):
            events.append("exit")

    with transform.PassContext(required_pass=["required"], instruments=[Skip()]):
        record_pass(events, "required")(tvm.IRModule())
    assert events == ["required", "exit"]

    @transform.module_pass(opt_level=0)
    def broken(mod, ctx):
        raise ValueError("pass failed")

    class ExitRecorder(transform.PassInstrument):
        def exit_pass_ctx(self):
            events.append("exception exit")

    with pytest.raises(ValueError, match="pass failed"):
        with transform.PassContext(instruments=[ExitRecorder()]):
            broken(tvm.IRModule())
    assert events[-1] == "exception exit"


def test_concrete_timing_and_dump(tmp_path):
    events = []
    timing = instrument.PassTimingInstrument()
    dump = instrument.DumpIR(tmp_path)
    with transform.PassContext(instruments=[timing, dump]):
        record_pass(events, "record")(tvm.IRModule())
        assert "record:" in timing.render()
    assert (tmp_path / "000_record.py").read_text()


def test_registered_configs_and_validation():
    assert transform.PassContext.list_configs()["tirx.UnrollLoop"]["type"] == (
        "tirx.transform.UnrollLoopConfig"
    )
    with transform.PassContext(config={"tirx.UnrollLoop": {"auto_max_step": 3}}) as ctx:
        config = ctx.config["tirx.UnrollLoop"]
        assert config.auto_max_step == 3
        assert config.auto_max_depth == 8
    with pytest.raises(AttributeError):
        transform.PassContext(config={"tirx.disable_vectorize": "invalid"})
    with pytest.raises(AttributeError):
        transform.PassContext(config={"unknown.configuration": True})
    with pytest.raises((AttributeError, TypeError)):
        transform.PassContext(config={"tirx.UnrollLoop": {"unknown_field": 3}})
